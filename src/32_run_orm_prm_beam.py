import subprocess
import os
import json
import time
import numpy as np
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

# ==========================================
# ★設定: パスを書き換えてください
# ==========================================
MODELS = {
    # 1.5Bなのか7Bなのか確認して正しいパスを指定してください
    "ORM": "models/orm_1.5b_30k_v1.0",           
    "PRM": "models/prm_1.5b_30k_v3.0" 
}

SEEDS = list(range(10)) # 0〜9の10回
GEN_MODEL = "Qwen/Qwen2.5-Math-7B-Instruct"
OUTPUT_DIR = "data/experiments/benchmark_1.5b_10seeds_beamsearch_width5_candi5"
SCRIPT_PATH = "src/31_step_beam_search_vllm_final.py"

# 使用可能なGPU数
NUM_GPUS = 8 

# ==========================================
# 1つの実験を実行する関数
# ==========================================
def run_experiment(task):
    model_name, model_path, seed, gpu_id = task
    
    output_json = os.path.join(OUTPUT_DIR, f"{model_name}_seed{seed}.json")
    
    # --- 修正箇所: 既に結果があれば、読み込んで返す ---
    if os.path.exists(output_json):
        try:
            with open(output_json, 'r') as f:
                data = json.load(f)
            acc = data['accuracy']
            print(f"[GPU {gpu_id}] Skipping {model_name} Seed {seed} (Exists) -> Loaded Acc: {acc:.2%}")
            return (model_name, seed, acc)
        except Exception as e:
            print(f"[GPU {gpu_id}] Warning: Failed to load existing file {output_json}, re-running. Error: {e}")
            # 読み込み失敗時は再実行へ進む
    # ------------------------------------------------

    print(f"[GPU {gpu_id}] Starting {model_name} Seed {seed}...")
    
    # 環境変数をセットしてGPUを指定
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"

    cmd = [
        "python", SCRIPT_PATH,
        "--gen_model", GEN_MODEL,
        "--prm_model", model_path,
        "--seed", str(seed),
        "--output_file", output_json,
        "--beam_width", "5",
        "--num_candidates", "5",
        "--gpu_memory_utilization", "0.5",
        "--max_model_len", "4096"
    ]
    
    try:
        # ログを個別のファイルに出す
        log_file = os.path.join(OUTPUT_DIR, f"log_{model_name}_seed{seed}.txt")
        with open(log_file, "w") as f:
            subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)
        
        # 精度を読み取って返す
        with open(output_json, 'r') as f:
            data = json.load(f)
            acc = data['accuracy']
            print(f"[GPU {gpu_id}] Finished {model_name} Seed {seed} -> Acc: {acc:.2%}")
            return (model_name, seed, acc)
            
    except Exception as e:
        print(f"[GPU {gpu_id}] Error in {model_name} Seed {seed}: {e}")
        return (model_name, seed, "Error")

# ==========================================
# メイン処理: キュー管理
# ==========================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 全タスクリスト作成
    all_tasks = []
    for model_name, model_path in MODELS.items():
        for seed in SEEDS:
            all_tasks.append((model_name, model_path, seed))
    
    print(f"Total tasks: {len(all_tasks)}")
    
    results_map = {name: [] for name in MODELS.keys()}
    
    # ProcessPoolExecutorで並列実行
    with ProcessPoolExecutor(max_workers=NUM_GPUS) as executor:
        futures = []
        
        for i, (m_name, m_path, seed) in enumerate(all_tasks):
            # GPU IDをローテーションで割り当て (0~7)
            gpu_id = i % NUM_GPUS
            futures.append(executor.submit(run_experiment, (m_name, m_path, seed, gpu_id)))
            
        # 結果回収
        for future in as_completed(futures):
            m_name, seed, res = future.result()
            if res is not None and res != "Error":
                results_map[m_name].append(res)

    # 最終レポート
    print("\n" + "="*60)
    print("FINAL 10-SEED BENCHMARK REPORT")
    print("="*60)
    print(f"{'Model':<10} | {'Mean Acc':<10} | {'Std Dev':<10} | {'Raw Seeds'}")
    print("-" * 60)
    
    for name, accs in results_map.items():
        if not accs:
            print(f"{name:<10} | N/A        | N/A        | No results")
            continue
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        # ソートはせず、実行順に関わらずリスト化
        raw_str = ", ".join([f"{x:.1%}" for x in accs])
        print(f"{name:<10} | {mean_acc:.2%}     | ±{std_acc:.2%}     | {raw_str}")
        
    print("="*60)

if __name__ == "__main__":
    main()