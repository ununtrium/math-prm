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
# 出力: 最終学習用データ
OUTPUT_FILE = "data/annotated_train_data_30k_v2.0.jsonl"

# --- 報酬設計パラメータ ---
POWER = 2.0           # 二乗強調 (大きな変動を重視)
CORRECT_BOOST = 1.0   # 正解パスの加点倍率
PARTIAL_SCALE = 0.1   # 不正解パスのプラス評価（部分点）の割引率

# ★追加: 正解パスへの基礎点 (地道な進歩を救う)
# 0.1 程度あれば、ノイズ(0.0)や不正解(-0.05)と区別できる
CORRECT_BASE_REWARD = 0.1

# --- ペナルティ設定 (Stagnation) ---
STAGNATION_THRESH = 0.002  # 0.2%未満の変動は「停滞」とみなす
BASE_PENALTY      = 0.05   # 停滞1回目のペナルティ
STAGNATION_GROWTH = 0.05   # 停滞が続くごとの追加ペナルティ

# --- その他設定 ---
BATCH_SIZE = 8        # VRAMに合わせて調整
TARGET_MAX_STEPS = 15 
MIN_MERGE_CHARS = 50  
TRIGGER_PHRASE = "\nThe final answer is \\boxed{"

# GPU設定
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


# ==========================================
# 2. 正解判定ロジック (SymPy)
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


# ==========================================
# 3. ステップ処理ロジック
# ==========================================
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
# 4. 確率計算ロジック (Probing)
# ==========================================
def calculate_joint_log_prob(model, tokenizer, context_texts, target_answer, device):
    full_texts = [ctx + TRIGGER_PHRASE + target_answer for ctx in context_texts]
    prefix_texts = [ctx + TRIGGER_PHRASE for ctx in context_texts]
    
    inputs = tokenizer(full_texts, return_tensors="pt", padding=True, padding_side="right").to(device)
    prefix_inputs = tokenizer(prefix_texts, return_tensors="pt", padding=True, padding_side="right")
    prefix_lengths = prefix_inputs.attention_mask.sum(dim=1).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = inputs.input_ids[:, 1:].contiguous()
    
    batch_log_probs = []
    for i in range(len(full_texts)):
        start_idx = prefix_lengths[i] - 1
        end_idx = inputs.attention_mask[i].sum() - 1
        if start_idx >= end_idx:
            batch_log_probs.append(-100.0)
            continue
        ans_logits = shift_logits[i, start_idx:end_idx, :]
        ans_labels = shift_labels[i, start_idx:end_idx]
        token_log_probs = torch.log_softmax(ans_logits, dim=-1)
        target_token_log_probs = token_log_probs.gather(1, ans_labels.unsqueeze(1)).squeeze(1)
        batch_log_probs.append(target_token_log_probs.sum().item())
        
    return batch_log_probs


# ==========================================
# 5. ワーカープロセス (修正版: スコープ対策 & メタデータ保存)
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
        attn_implementation="eager" # Flash Attnがあるなら "flash_attention_2"
    )
    model.eval()

    annotated_data = []
    iterator = tqdm(lines_chunk, desc=f"GPU {gpu_id}", position=rank) if len(lines_chunk) > 0 else []

    for line in iterator:
        # 変数のスコープ対策: tryブロック内で変数を初期化
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

        # パスごとのループ (enumerateを使用)
        for path_idx, path in enumerate(paths):
            # 1. ステップ分割
            steps = get_adaptive_steps(path)
            if not steps: continue

            # 2. 正解判定 & Hindsight Relabeling
            generated_answer = extract_answer_content(path)
            is_correct_outcome = False
            probing_target = default_target
            
            if generated_answer:
                if check_equivalence_sympy(generated_answer, default_target):
                    is_correct_outcome = True
                    # 表記が違っても正解なら、生成された答えをターゲットにする
                    if generated_answer != default_target:
                        probing_target = generated_answer
                else:
                    is_correct_outcome = False

            # 3. コンテキスト構築
            context_list = [problem] 
            current_text = problem
            for step in steps:
                current_text += "\n" + step
                context_list.append(current_text)
            
            # 4. 確率計算 (Probing)
            all_log_probs = []
            for i in range(0, len(context_list), BATCH_SIZE):
                batch_contexts = context_list[i : i + BATCH_SIZE]
                batch_log_probs = calculate_joint_log_prob(
                    model, tokenizer, batch_contexts, probing_target, device
                )
                all_log_probs.extend(batch_log_probs)

            # 5. Raw Delta計算 & 報酬付与
            probs = [math.exp(lp) for lp in all_log_probs]
            # probs[0]はBase Probなのでスキップ
            
            stagnation_streak = 0 # 停滞カウンター
            path_id = f"{source_id}_path_{path_idx}" # Path ID

            for t in range(1, len(probs)):
                # 純粋な差分 (Raw Delta)
                raw_delta = probs[t] - probs[t-1]
                
                # --- 停滞判定 ---
                if abs(raw_delta) < STAGNATION_THRESH: # 0.2%未満なら停滞
                    stagnation_streak += 1
                else:
                    stagnation_streak = 0
                
                # ペナルティ計算
                penalty = 0.0
                if stagnation_streak > 0:
                    penalty = BASE_PENALTY + ((stagnation_streak - 1) * STAGNATION_GROWTH)

                # --- 最終ラベル決定 (Asymmetric Rectification) ---
                final_label = 0.0
                
                if is_correct_outcome:
                    if raw_delta > 0:
                        # [正解パス & プラス]: スパイクを強調
                        val = (raw_delta * CORRECT_BOOST) ** POWER
                        final_label = CORRECT_BASE_REWARD + val
                    else:
                        # [正解パス & マイナス]: 免罪 (0.0)
                        # 停滞ペナルティも適用しない（迷いを許容）
                        final_label = CORRECT_BASE_REWARD * 0.5
                else:
                    # [不正解パス]
                    # 基礎点 (符号付き2乗)
                    if raw_delta > 0:
                        # 部分正解: 0.1倍に割引
                        base_val = PARTIAL_SCALE * ((raw_delta) ** POWER)
                    else:
                        # ミス: マイナス評価
                        base_val = -1 * (abs(raw_delta) ** POWER)
                    
                    # ペナルティを引く (不正解パスのみ停滞を罰する)
                    final_label = base_val - penalty

                # データ保存 (メタデータ付き)
                new_rec = {
                    "source_id": source_id,
                    "source": source_name,
                    "problem": problem,
                    "path_id": path_id,
                    "step_index": t,
                    "step_text": steps[t-1],
                    "full_text": context_list[t],   # 学習用入力
                    "label": final_label,           # 学習用ラベル
                    "raw_delta": raw_delta,         # 分析用
                    "is_outcome_correct": is_correct_outcome
                }
                
                annotated_data.append(new_rec)

    # 一時ファイル書き出し
    with open(output_temp_file, "w", encoding="utf-8") as f:
        for item in annotated_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"[GPU {gpu_id}] Finished. Saved chunk.")


# ==========================================
# 6. メインエントリー (並列実行)
# ==========================================
def main():
    if not os.path.exists(INPUT_FILE):
        print("Input file not found.")
        return

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0: num_gpus = 1
    
    print(f"Starting Annotation on {num_gpus} GPUs...")
    print(f"  Threshold: {STAGNATION_THRESH}")
    print(f"  Penalty:   Base={BASE_PENALTY}, Growth={STAGNATION_GROWTH}")

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