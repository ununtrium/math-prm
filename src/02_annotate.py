import json
import os
import re
import torch
import math
import time
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.multiprocessing as mp

# ==========================================
# ★変更点: ライブラリのインポート
# ==========================================
from latex2sympy2 import latex2sympy
from sympy import simplify

# ==========================================
# 1. 設定パラメータ (変更なし)
# ==========================================
MODEL_ID = "Qwen/Qwen2.5-Math-7B-Instruct"
INPUT_FILE = "data/numinamath_gen_30k.jsonl"
OUTPUT_FILE = "data/annotated_train_data_30k.jsonl"

GAMMA = 0.9           
OUTCOME_REWARD = 1.0  
SCALE_FACTOR = 1.0    

CLIP_MAX = 2.0
CLIP_MIN = -2.0

TARGET_MAX_STEPS = 15 
MIN_MERGE_CHARS = 50  

BATCH_SIZE = 64        
TRIGGER_PHRASE = "\nThe final answer is \\boxed{"

# ==========================================
# 2. 正解判定関数 (ライブラリ使用により大幅短縮)
# ==========================================

from sympy import simplify, nsimplify

def check_equivalence(pred_str, gold_str):
    """
    latex2sympy2 を使用し、小数vs分数や浮動小数点誤差も吸収する堅牢なチェック
    """
    if not pred_str or not gold_str: return False
    
    # 1. 文字列としての完全一致 (高速化)
    if pred_str.strip() == gold_str.strip(): return True

    try:
        # latex2sympy で SymPy オブジェクト化
        sym_pred = latex2sympy(pred_str)
        sym_gold = latex2sympy(gold_str)
        
        # 差分をとる
        diff = sym_pred - sym_gold

        # ---------------------------------------------------------
        # 判定A: 記号的な完全一致 (simplify)
        # ---------------------------------------------------------
        if simplify(diff) == 0:
            return True

        # ---------------------------------------------------------
        # 判定B: 有理化して比較 (nsimplify)
        # 0.1 と 1/10 を同一視するために、小数を分数に変換してから簡約
        # ---------------------------------------------------------
        if nsimplify(diff) == 0:
            return True
            
        # ---------------------------------------------------------
        # 判定C: 数値的な近似一致 (evalf)
        # 最終的に数値計算して、差が誤差(epsilon)以下なら正解とする
        # ---------------------------------------------------------
        # 変数(x, y)が含まれていない場合のみ実行
        if not diff.free_symbols:
            # 数値評価 (10桁精度などで計算)
            val = diff.evalf()
            if abs(val) < 1e-7:
                return True

    except Exception:
        # パースエラー等の場合
        return robust_float_check(pred_str, gold_str)

    return False
    
def robust_float_check(pred, gold):
    """
    SymPyがコケた時のための最後の砦（数値変換比較）
    自作の簡易パーサーだが、latex2sympy2が失敗するようなケース(崩れたLaTeXなど)用
    """
    try:
        def to_float(s):
            # 簡易置換
            s = s.replace(r"\frac", "").replace("{", "(").replace("}", ")").replace("^", "**")
            s = s.replace(r"\left", "").replace(r"\right", "")
            return float(eval(s))
        
        # アルファベット(変数)が含まれていない場合のみ実行
        if not any(c.isalpha() for c in pred) and not any(c.isalpha() for c in gold):
            return abs(to_float(pred) - to_float(gold)) < 1e-6
    except:
        pass
    return False

# ==========================================
# 3. テキスト処理・ステップ分割関数 (変更なし)
# ==========================================
def extract_answer_content(text):
    if not text: return None
    matches = re.findall(r"\\boxed\{(.*?)\}", text)
    if not matches: return None
    return matches[-1].strip()

def reduce_step_count(steps, target_max=15, min_chars=50):
    if len(steps) <= target_max: return steps
    merged_steps = []
    buffer = ""
    for i, step in enumerate(steps):
        if i == 0: buffer = step; continue
        if len(step) < min_chars or len(buffer) < min_chars:
            buffer += "\n" + step
        else:
            merged_steps.append(buffer); buffer = step
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
# 4. 確率計算ロジック (変更なし)
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
# 5. ワーカープロセス (変更なし)
# ==========================================
def worker_process(rank, gpu_id, lines_chunk, output_temp_file):
    print(f"[GPU {gpu_id}] Worker started. Processing {len(lines_chunk)} lines.")
    
    # GPU設定
    device = f"cuda:{gpu_id}"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        torch_dtype=dtype, 
        device_map=device, 
        attn_implementation="eager"
    )
    model.eval()

    annotated_data = []
    iterator = tqdm(lines_chunk, desc=f"GPU {gpu_id}", position=rank) if len(lines_chunk) > 0 else []

    for line in iterator:
        try: record = json.loads(line)
        except: continue
            
        source_id = record.get("source_id")
        source_name = record.get("source", "unknown")
        problem = record["problem"]
        ground_truth_raw = record["ground_truth"]
        paths = record["generated_paths"]
        
        default_target = extract_answer_content(ground_truth_raw)
        if not default_target: continue

        for path in paths:
            steps = get_adaptive_steps(path)
            if not steps: continue

            # --- 正解判定 (ライブラリ使用) & Hindsight Relabeling ---
            generated_answer = extract_answer_content(path)
            is_correct_outcome = False
            probing_target = default_target
            
            if generated_answer:
                # ★変更: check_equivalence (latex2sympy2) を使用
                if check_equivalence(generated_answer, default_target):
                    is_correct_outcome = True
                    if generated_answer != default_target:
                        probing_target = generated_answer
                else:
                    is_correct_outcome = False

            context_list = [problem] 
            current_text = problem
            for step in steps:
                current_text += "\n" + step
                context_list.append(current_text)
            
            all_log_probs = []
            for i in range(0, len(context_list), BATCH_SIZE):
                batch_contexts = context_list[i : i + BATCH_SIZE]
                batch_log_probs = calculate_joint_log_prob(
                    model, tokenizer, batch_contexts, probing_target, device
                )
                all_log_probs.extend(batch_log_probs)

            probs = [math.exp(lp) for lp in all_log_probs]
            raw_deltas = []
            for t in range(1, len(probs)):
                delta = (probs[t] - probs[t-1]) * SCALE_FACTOR
                raw_deltas.append(delta)

            if is_correct_outcome:
                running_add = OUTCOME_REWARD 
            else:
                running_add = 0.0

            discounted_values = [0.0] * len(raw_deltas)
            for t in reversed(range(len(raw_deltas))):
                if t == len(raw_deltas) - 1 and is_correct_outcome:
                    current_val = OUTCOME_REWARD
                else:
                    current_val = raw_deltas[t] + GAMMA * running_add
                discounted_values[t] = current_val
                running_add = current_val

            for t, val in enumerate(discounted_values):
                label = max(CLIP_MIN, min(CLIP_MAX, val))
                annotated_data.append({
                    "source_id": source_id,
                    "source": source_name,
                    "problem": problem,
                    "step_text": steps[t],
                    "full_text": context_list[t+1],
                    "label": label,
                    "raw_score": val,
                    "is_outcome_correct": is_correct_outcome
                })

    with open(output_temp_file, "w", encoding="utf-8") as f:
        for item in annotated_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"[GPU {gpu_id}] Finished. Saved {len(annotated_data)} samples.")

# ==========================================
# 6. メインエントリーポイント (変更なし)
# ==========================================
def main():
    if not os.path.exists(INPUT_FILE):
        print("Input file not found.")
        return

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("No GPU detected. Using CPU (slow).")
        num_gpus = 1
    
    print(f"Detected {num_gpus} GPUs. Starting parallel annotation...")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        all_lines = f.readlines()
    
    total_lines = len(all_lines)
    chunk_size = math.ceil(total_lines / num_gpus)
    chunks = [all_lines[i:i + chunk_size] for i in range(0, total_lines, chunk_size)]
    
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
        
    print("All workers finished. Merging files...")
    
    total_samples = 0
    with open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                with open(temp_file, "r", encoding="utf-8") as infile:
                    for line in infile:
                        outfile.write(line)
                        total_samples += 1
                os.remove(temp_file)
    
    print(f"Done! Total {total_samples} annotated samples saved to {OUTPUT_FILE}.")

if __name__ == "__main__":
    main()