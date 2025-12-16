import subprocess
import os
import json
import numpy as np
from datetime import datetime

# ==========================================
# ★設定: ここを環境に合わせて変更してください
# ==========================================
MODELS = {
    # ORMのパス (1.5Bなのか7Bなのかに合わせて指定)
    "ORM": "models/orm_7b_30k_v1.0", 
    
    # PRMのパス (7Bのチェックポイント)
    "PRM": "models/prm_7b_30k_v3.0/checkpoint-16174" 
}

SEEDS = [42, 123, 2024]
GEN_MODEL = "Qwen/Qwen2.5-Math-7B-Instruct"
OUTPUT_DIR = "data/experiments/beamsearch_results"
SCRIPT_PATH = "src/31_step_beam_search_vllm_final.py" # 上記コードのパス

# ==========================================
# メイン処理
# ==========================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    report = {}

    print(f"Starting Benchmark at {datetime.now()}")
    print(f"Models: {list(MODELS.keys())}")
    print(f"Seeds: {SEEDS}")
    print("-" * 50)

    for model_name, model_path in MODELS.items():
        print(f"\n>>> Evaluating Model: {model_name} ({model_path})")
        accuracies = []
        
        for seed in SEEDS:
            output_json = os.path.join(OUTPUT_DIR, f"{model_name}_seed{seed}.json")
            
            # すでに実行済みならスキップ (再開用)
            if os.path.exists(output_json):
                print(f"  Seed {seed}: Found existing result.")
                with open(output_json, 'r') as f:
                    data = json.load(f)
                    accuracies.append(data['accuracy'])
                continue

            print(f"  Running Seed {seed}...")
            
            # サブプロセスで実行 (vLLMのメモリを毎回クリアするため)
            cmd = [
                "python", SCRIPT_PATH,
                "--gen_model", GEN_MODEL,
                "--prm_model", model_path,
                "--seed", str(seed),
                "--output_file", output_json,
                "--beam_width", "3",
                "--num_candidates", "5",
                "--gpu_memory_utilization", "0.6",
                "--max_model_len", "4096"
            ]
            
            try:
                subprocess.run(cmd, check=True)
                
                # 結果読み込み
                with open(output_json, 'r') as f:
                    data = json.load(f)
                    accuracies.append(data['accuracy'])
                    print(f"  -> Accuracy: {data['accuracy']:.2%}")
                    
            except subprocess.CalledProcessError:
                print(f"  -> Error occurred for seed {seed}. Skipping.")
        
        # 集計
        if accuracies:
            mean_acc = np.mean(accuracies)
            std_acc = np.std(accuracies)
            report[model_name] = {
                "mean": mean_acc,
                "std": std_acc,
                "raw": accuracies
            }
        else:
            print(f"  No results for {model_name}.")

    # ==========================================
    # 最終レポート出力
    # ==========================================
    print("\n\n" + "="*60)
    print("FINAL BENCHMARK REPORT (MATH-500 Beam Search)")
    print("="*60)
    print(f"{'Model':<10} | {'Mean Acc':<10} | {'Std Dev':<10} | {'Raw Seeds'}")
    print("-" * 60)
    
    for name, stats in report.items():
        mean_str = f"{stats['mean']:.2%}"
        std_str = f"±{stats['std']:.2%}"
        raw_str = ", ".join([f"{x:.2%}" for x in stats['raw']])
        print(f"{name:<10} | {mean_str:<10} | {std_str:<10} | {raw_str}")
        
    print("="*60)

if __name__ == "__main__":
    main()