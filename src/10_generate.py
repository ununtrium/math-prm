import argparse
import json
import random
import os
import sys

# srcディレクトリにパスを通す（実行場所によるエラー回避のため）
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from datasets import load_dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from src.utils import is_suitable_for_prm, build_prompt  # 作成したutilsをインポート

def main():
    parser = argparse.ArgumentParser(description="Generate solutions using vLLM")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-Math-7B-Instruct")
    parser.add_argument("--output_file", type=str, default="data/numinamath_gen_30k.jsonl")
    parser.add_argument("--num_problems", type=int, default=30000)
    parser.add_argument("--n_paths", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tensor_parallel", type=int, default=1)
    parser.add_argument("--gpu_utilization", type=float, default=0.9)
    args = parser.parse_args()

    # 0. 初期化
    random.seed(args.seed)
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    print(f"--- Settings ---")
    print(f"Model: {args.model_id}")
    print(f"Target Problems: {args.num_problems}")
    print(f"Paths per Problem: {args.n_paths}")
    print(f"Output: {args.output_file}")
    print(f"----------------")

    # 1. NuminaMathデータのロード
    print("Loading NuminaMath dataset...")
    try:
        # dataset全体をロード (メモリ効率のためストリーミングはせず、インデックスアクセスを使用)
        dataset = load_dataset("AI-MO/NuminaMath-CoT", split="train")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    total_len = len(dataset)
    print(f"Total problems in raw dataset: {total_len}")

    # 2. フィルタリング付きランダムサンプリング
    print(f"Searching for {args.num_problems} suitable problems...")
    
    # インデックスをシャッフルしてランダムアクセス
    indices = list(range(total_len))
    random.shuffle(indices)
    
    selected_problems = []
    
    # 必要な数が見つかるまでデータセットを走査
    for idx in indices:
        item = dataset[idx]
        solution = item['solution']
        
        # utilsのフィルタリング関数を使用
        if is_suitable_for_prm(solution):
            selected_problems.append({
                "source_id": idx,
                "source": item.get('source', 'unknown'),
                "problem": item['problem'],
                "ground_truth": solution
            })
        
        if len(selected_problems) >= args.num_problems:
            break
            
    print(f"Collected {len(selected_problems)} problems suitable for PRM training.")
    
    if len(selected_problems) < args.num_problems:
        print(f"Warning: Could not find enough suitable problems. Using {len(selected_problems)} problems.")

    # 3. vLLMの初期化
    print("Initializing vLLM Engine...")
    llm = LLM(
        model=args.model_id,
        tensor_parallel_size=args.tensor_parallel,
        gpu_memory_utilization=args.gpu_utilization,
        trust_remote_code=True,
        seed=args.seed,
        dtype="bfloat16"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    # 4. プロンプト作成
    print("Preparing prompts...")
    prompts = []
    for item in selected_problems:
        # utilsのプロンプト作成関数を使用
        prompt_text = build_prompt(tokenizer, item['problem'])
        prompts.append(prompt_text)

    # 5. 生成パラメータ設定
    sampling_params = SamplingParams(
        n=args.n_paths,
        temperature=0.7,
        max_tokens=2048,
        top_p=0.95,
        seed=args.seed
    )

    # 6. 生成実行
    print(f"Generating {args.n_paths} paths for each problem using vLLM...")
    outputs = llm.generate(prompts, sampling_params)

    # 7. JSONL形式で保存
    print(f"Saving results to {args.output_file}...")
    with open(args.output_file, "w", encoding="utf-8") as f:
        for item, output in zip(selected_problems, outputs):
            generated_texts = [o.text for o in output.outputs]
            
            # 保存用レコード作成
            record = {
                "source_id": item['source_id'],
                "source": item['source'],
                "problem": item['problem'],
                "ground_truth": item['ground_truth'],
                "generated_paths": generated_texts  # generated_samples でも可（統一されていればOK）
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Done! Saved {len(selected_problems)} records.")

if __name__ == "__main__":
    main()