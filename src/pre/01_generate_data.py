import json
import random
import os
import re
import torch
from datasets import load_dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# ==========================================
# 設定パラメータ (Configuration)
# ==========================================
MODEL_ID = "Qwen/Qwen2.5-Math-7B-Instruct"

# 出力設定
OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "numinamath_gen_30k.jsonl")

# 実験設定
NUM_PROBLEMS = 30000      # 収集する問題数
N_PATHS = 8              # 1問あたりの生成パス数
SEED = 42                # 再現性のための乱数シード

# GPU設定
GPU_UTILIZATION = 0.9    # VRAM使用率
TENSOR_PARALLEL = 1      # GPU枚数


# ==========================================
# フィルタリング関数
# ==========================================
def is_suitable_for_prm(solution_text):
    """
    PRM学習に適した問題かどうかを判定するフィルタリング関数。
    """
    if not solution_text:
        return False
        
    # \boxed{...} を探す
    matches = re.findall(r"\\boxed\{(.*?)\}", solution_text)
    
    if not matches:
        return False  # 正解フォーマットがないものは除外
    
    final_answer = matches[-1].strip()
    
    # 基準1: 文字数制限 (100文字)
    if len(final_answer) > 100:
        return False
    if len(final_answer) == 0:
        return False

    # 基準2: 改行コードが含まれている場合は除外
    if "\\\\" in final_answer or "\n" in final_answer:
        return False

    # 基準3: 等号が複数ある場合は除外
    if final_answer.count("=") > 1:
        return False
        
    # 基準4: 特殊な環境定義を除外
    if "\\begin" in final_answer: 
        return False

    return True


# ==========================================
# メイン処理
# ==========================================
def main():
    # 0. 初期化
    random.seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"--- Settings ---")
    print(f"Model: {MODEL_ID}")
    print(f"Target Problems: {NUM_PROBLEMS}")
    print(f"Paths per Problem: {N_PATHS}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"----------------")

    # 1. NuminaMathデータのロード
    print("Loading NuminaMath dataset (this may take a while)...")
    try:
        dataset = load_dataset("AI-MO/NuminaMath-CoT", split="train")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    total_len = len(dataset)
    print(f"Total problems in raw dataset: {total_len}")

    # 2. フィルタリング付きランダムサンプリング
    print(f"Searching for {NUM_PROBLEMS} suitable problems...")
    
    indices = list(range(total_len))
    random.shuffle(indices)
    
    selected_problems = []
    
    for idx in indices:
        item = dataset[idx]
        solution = item['solution']
        
        # フィルタリング実行
        if is_suitable_for_prm(solution):
            selected_problems.append({
                "source_id": idx,
                "source": item.get('source', 'unknown'), # ★ここを追加: ソース情報の取得
                "problem": item['problem'],
                "ground_truth": solution
            })
        
        if len(selected_problems) >= NUM_PROBLEMS:
            break
            
    print(f"Collected {len(selected_problems)} problems suitable for PRM training.")
    
    if len(selected_problems) < NUM_PROBLEMS:
        print("Warning: Could not find enough suitable problems.")

    # 3. vLLMの初期化
    print("Initializing vLLM Engine...")
    llm = LLM(
        model=MODEL_ID,
        tensor_parallel_size=TENSOR_PARALLEL,
        gpu_memory_utilization=GPU_UTILIZATION,
        trust_remote_code=True,
        seed=SEED,
        dtype="bfloat16"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # 4. プロンプト作成
    print("Preparing prompts...")
    prompts = []
    system_prompt = "Please reason step by step and put your final answer within \\boxed{}."
    
    for item in selected_problems:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": item['problem']}
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        prompts.append(prompt_text)

    # 5. 生成パラメータ設定
    sampling_params = SamplingParams(
        n=N_PATHS,
        temperature=0.7,
        max_tokens=2048,
        top_p=0.95,
        seed=SEED
    )

    # 6. 生成実行
    print(f"Generating {N_PATHS} paths for each problem using vLLM...")
    outputs = llm.generate(prompts, sampling_params)

    # 7. JSONL形式で保存
    print(f"Saving results to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item, output in zip(selected_problems, outputs):
            generated_texts = [o.text for o in output.outputs]
            
            # 保存用レコード作成
            record = {
                "source_id": item['source_id'],
                "source": item['source'],             # ★ここを追加: ソース情報の保存
                "problem": item['problem'],
                "ground_truth": item['ground_truth'],
                "generated_paths": generated_texts
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Done! Saved {len(selected_problems)} records.")

if __name__ == "__main__":
    main()