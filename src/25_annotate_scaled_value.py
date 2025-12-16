import json
import os
import torch
import math
import re
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.multiprocessing as mp

# SymPy関連
from latex2sympy2 import latex2sympy
from sympy import simplify

# ==========================================
# 1. 設定パラメータ
# ==========================================
MODEL_ID = "Qwen/Qwen2.5-Math-7B-Instruct"

# 入力: 生成直後の生データ
INPUT_FILE = "data/numinamath_gen_30k.jsonl"
# 出力: 価値関数学習用データ
OUTPUT_FILE = "data/p_scaled_value_train_30k_chat.jsonl"

# ★重要: 不正解パスの割引係数 (0.1 = 10%の部分点)
# 正解パスは 1.0 (そのまま)
INCORRECT_SCALE = 0.1

# その他設定
BATCH_SIZE = 8
TRIGGER_PHRASE = "\nThe final answer is \\boxed{"
TARGET_MAX_STEPS = 15
MIN_MERGE_CHARS = 50

# GPU設定
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


# ==========================================
# 2. ユーティリティ (正解判定・ステップ分割)
# ==========================================
def extract_answer_content(text):
    if not text: return None
    matches = re.findall(r"\\boxed\{(.*?)\}", text)
    if not matches: return None
    return matches[-1].strip()

def robust_float_check(pred, gold):
    try:
        def to_float(s):
            s = str(s).replace(r"\frac", "").replace("{", "(").replace("}", ")").replace("^", "**")
            s = s.replace(r"\left", "").replace(r"\right", "").replace(",", "")
            return float(eval(s))
        if not any(c.isalpha() for c in str(pred)) and not any(c.isalpha() for c in str(gold)):
            return abs(to_float(pred) - to_float(gold)) < 1e-6
    except:
        pass
    return False

def check_equivalence_sympy(pred_str, gold_str):
    if not pred_str or not gold_str: return False
    if pred_str.strip() == gold_str.strip(): return True
    try:
        sym_pred = latex2sympy(pred_str)
        sym_gold = latex2sympy(gold_str)
        if simplify(sym_pred - sym_gold) == 0:
            return True
    except Exception:
        return robust_float_check(pred_str, gold_str)
    return False

def reduce_step_count(steps, target_max=15, min_chars=50):
    if len(steps) <= target_max: return steps
    merged_steps = []
    buffer = ""
    for i, step in enumerate(steps):
        if i == 0: buffer = step; continue
        if len(step) < min_chars or len(buffer) < min_chars: buffer += "\n" + step
        else: merged_steps.append(buffer); buffer = step
    if buffer: merged_steps.append(buffer)
    while len(merged_steps) > target_max:
        new_merged = []
        for i in range(0, len(merged_steps), 2):
            if i + 1 < len(merged_steps): new_merged.append(merged_steps[i] + "\n" + merged_steps[i+1])
            else: new_merged.append(merged_steps[i])
        merged_steps = new_merged
        if len(merged_steps) <= 1: break
    return merged_steps

def get_adaptive_steps(path_text):
    steps = [s.strip() for s in re.split(r'\n\s*\n', path_text) if s.strip()]
    if len(steps) < 3:
        alt_steps = [s.strip() for s in path_text.split('\n') if s.strip()]
        if len(alt_steps) >= 3: steps = alt_steps
    if len(steps) > TARGET_MAX_STEPS:
        raw_lines = [s.strip() for s in path_text.split('\n') if s.strip()]
        steps = reduce_step_count(raw_lines, target_max=TARGET_MAX_STEPS, min_chars=MIN_MERGE_CHARS)
    return steps


# ==========================================
# 3. 確率計算ロジック (Value Probing)
# ==========================================
def calculate_joint_prob(model, tokenizer, context_texts, target_answer, device):
    """
    文脈(context_texts)を与えた状態で、ターゲット回答(target_answer)が生成される確率を計算する。
    戻り値: 0.0 ~ 1.0 の確率値のリスト
    """
    full_texts = [ctx + TRIGGER_PHRASE + target_answer for ctx in context_texts]
    prefix_texts = [ctx + TRIGGER_PHRASE for ctx in context_texts]
    
    # トークナイズ
    inputs = tokenizer(full_texts, return_tensors="pt", padding=True, padding_side="right").to(device)
    prefix_inputs = tokenizer(prefix_texts, return_tensors="pt", padding=True, padding_side="right")
    prefix_lengths = prefix_inputs.attention_mask.sum(dim=1).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Logitsのシフト (Next Token Predictionのため)
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = inputs.input_ids[:, 1:].contiguous()
    
    probs = []
    for i in range(len(full_texts)):
        start_idx = prefix_lengths[i] - 1
        end_idx = inputs.attention_mask[i].sum() - 1
        
        if start_idx >= end_idx:
            probs.append(0.0)
            continue
            
        # ターゲット部分(Answer)のみのLogitsを抽出
        ans_logits = shift_logits[i, start_idx:end_idx, :]
        ans_labels = shift_labels[i, start_idx:end_idx]
        
        # Log Softmax で対数確率を取得
        token_log_probs = torch.log_softmax(ans_logits, dim=-1)
        # 正解トークンの対数確率を取得
        target_token_log_probs = token_log_probs.gather(1, ans_labels.unsqueeze(1)).squeeze(1)
        
        # 合計して exp を取り、結合確率(0~1)に戻す
        total_log_prob = target_token_log_probs.sum().item()
        prob = math.exp(total_log_prob)
        probs.append(prob)
        
    return probs


# ==========================================
# 4. ワーカープロセス
# ==========================================
def worker_process(rank, gpu_id, lines_chunk, output_temp_file):
    print(f"[GPU {gpu_id}] Worker started.")
    device = f"cuda:{gpu_id}"
    
    # モデルロード
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        torch_dtype=DTYPE, 
        device_map=device, 
        attn_implementation="eager"
    )
    model.eval()

    annotated_data = []
    iterator = tqdm(lines_chunk, desc=f"GPU {gpu_id}", position=rank) if len(lines_chunk) > 0 else []

    for line in iterator:
        # スコープエラー対策: 変数初期化
        try: 
            record = json.loads(line)
            source_id = record["source_id"]
            problem = record["problem"]
            ground_truth_raw = record["ground_truth"]
            paths = record["generated_paths"]
            source_name = record.get("source", "unknown")
        except: 
            continue
            
        default_target = extract_answer_content(ground_truth_raw)
        if not default_target: continue

        # パスごとの処理
        for path_idx, path in enumerate(paths):
            # 1. ステップ分割
            steps = get_adaptive_steps(path)
            if not steps: continue

            # 2. 正解判定 & Target決定
            generated_answer = extract_answer_content(path)
            is_correct_outcome = False
            probing_target = default_target
            
            if generated_answer:
                if check_equivalence_sympy(generated_answer, default_target):
                    is_correct_outcome = True
                    # 表記ゆれ対応: モデル自身の生成した正解をターゲットにする方が自然
                    if generated_answer != default_target:
                        probing_target = generated_answer
                else:
                    is_correct_outcome = False

            # 3. コンテキスト構築
            # Problemから始まり、1ステップずつ追加していく
            # 修正部分
            messages = [
                {"role": "system", "content": "Please reason step by step and put your final answer within \\boxed{}."},
                {"role": "user", "content": problem}
            ]
            
            # apply_chat_template で "<|im_start|>assistant\n" までを含めたプロンプトを作成
            base_prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # ベースをチャット形式にする
            context_list = [base_prompt] 
            current_text = base_prompt
            #context_list = [problem] 
            #current_text = problem
            for step in steps:
                #current_text += "\n" + step
                current_text += step + "\n"
                context_list.append(current_text)
            
            # 4. 確率計算 (Value Probing)
            step_probs = []
            for i in range(0, len(context_list), BATCH_SIZE):
                batch_contexts = context_list[i : i + BATCH_SIZE]
                batch_probs = calculate_joint_prob(
                    model, tokenizer, batch_contexts, probing_target, device
                )
                step_probs.extend(batch_probs)
            
            path_id = f"{source_id}_path_{path_idx}"

            # 5. ラベル生成 (Scaled Value Logic)
            # step_probs[0] は Problem単体の確率なので、t=1 (Step1完了後) から保存する
            for t in range(1, len(step_probs)):
                current_prob = step_probs[t]
                
                # ★ここが核心★
                if is_correct_outcome:
                    # 正解パス: 進捗率(確率)をそのまま学習させる
                    # 例: 0.1 -> 0.5 -> 0.99
                    label = current_prob
                else:
                    # 不正解パス: 形は維持しつつ、高さだけ0.1倍に潰す
                    # 例: 0.1 -> 0.5 -> 0.99 (幻覚)  =>  0.01 -> 0.05 -> 0.099 (矯正)
                    label = current_prob * INCORRECT_SCALE
                
                # データ保存
                new_rec = {
                    "source_id": source_id,
                    "source": source_name,
                    "problem": problem,
                    "path_id": path_id,
                    "step_index": t,
                    "step_text": steps[t-1],
                    "full_text": context_list[t],   # 入力
                    "label": label,                 # 教師ラベル
                    "raw_prob": current_prob,       # 分析用(元の確率)
                    "is_outcome_correct": is_correct_outcome
                }
                annotated_data.append(new_rec)

    # 一時ファイル書き出し
    with open(output_temp_file, "w", encoding="utf-8") as f:
        for item in annotated_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"[GPU {gpu_id}] Finished. Saved chunk.")


# ==========================================
# 5. メインエントリー
# ==========================================
def main():
    if not os.path.exists(INPUT_FILE):
        print("Input file not found.")
        return

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0: num_gpus = 1
    
    print(f"Starting SCALED VALUE Annotation on {num_gpus} GPUs...")
    print(f"  Incorrect Scale: {INCORRECT_SCALE} (10%)")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        all_lines = f.readlines()
    
    chunk_size = math.ceil(len(all_lines) / num_gpus)
    chunks = [all_lines[i:i + chunk_size] for i in range(0, len(all_lines), chunk_size)]
    
    processes = []
    temp_files = []
    mp.set_start_method('spawn', force=True)

    for rank, chunk in enumerate(chunks):
        if not chunk: continue
        gpu_id = rank % num_gpus
        temp_file = f"{OUTPUT_FILE}.part{rank}"
        temp_files.append(temp_file)
        p = mp.Process(target=worker_process, args=(rank, gpu_id, chunk, temp_file))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
        
    print("Merging output files...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                with open(temp_file, "r", encoding="utf-8") as infile:
                    for line in infile:
                        outfile.write(line)
                os.remove(temp_file)
    
    print(f"Done! Dataset saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()