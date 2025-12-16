import argparse
import subprocess
import os
import json
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

# ==========================================
# ★設定: モデルパス (ここは固定でもOKですが、必要なら引数化も可能です)
# ==========================================
MODELS = {
    "ORM": "models/orm_1.5b_30k_v1.0",           
    "PRM": "models/prm_1.5b_30k_v3.0" 
}
GEN_MODEL = "Qwen/Qwen2.5-Math-7B-Instruct"
SCRIPT_PATH = "src/31_step_beam_search_vllm_final.py"

# ==========================================
# 引数解析
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="Run Benchmark with flexible args")
    
    # 必須: シード (カンマ区切りで複数指定可、例: "0" または "0,4,6")
    parser.add_argument("--seeds", type=str, required=True, help="Seeds to run (comma separated, e.g., '0,4,6')")
    
    # 設定パラメータ (デフォルト値あり)
    parser.add_argument("--beam_width", type=int, default=5, help="Beam search width")
    parser.add_argument("--num_candidates", type=int, default=5, help="Number of candidates per step")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.6, help="vLLM GPU memory utilization")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use for parallelism")
    
    return parser.parse_args()

# ==========================================
# 実験実行関数
# ==========================================
def run_experiment(task):
    model_name, model_path, seed, gpu_id, args = task
    
    output_json = os.path.join(args.output_dir, f"{model_name}_seed{seed}.json")
    
    # スキップ判定
    if os.path.exists(output_json):
        try:
            with open(output_json, 'r') as f:
                data = json.load(f)
            acc = data['accuracy']
            print(f"[GPU {gpu_id}] Skipping {model_name} Seed {seed} (Exists) -> Loaded Acc: {acc:.2%}")
            return (model_name, seed, acc)
        except Exception as e:
            print(f"[GPU {gpu_id}] Warning: Failed to load existing file {output_json}, re-running. Error: {e}")

    print(f"[GPU {gpu_id}] Starting {model_name} Seed {seed}...")
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"

    cmd = [
        "python", SCRIPT_PATH,
        "--gen_model", GEN_MODEL,
        "--prm_model", model_path,
        "--seed", str(seed),
        "--output_file", output_json,
        "--beam_width", str(args.beam_width),       # 引数から受取
        "--num_candidates", str(args.num_candidates), # 引数から受取
        "--gpu_memory_utilization", str(args.gpu_memory_utilization), # 引数から受取
        "--max_model_len", "4096"
    ]
    
    try:
        log_file = os.path.join(args.output_dir, f"log_{model_name}_seed{seed}.txt")
        with open(log_file, "w") as f:
            subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)
        
        with open(output_json, 'r') as f:
            data = json.load(f)
            acc = data['accuracy']
            print(f"[GPU {gpu_id}] Finished {model_name} Seed {seed} -> Acc: {acc:.2%}")
            return (model_name, seed, acc)
            
    except Exception as e:
        print(f"[GPU {gpu_id}] Error in {model_name} Seed {seed}: {e}")
        return (model_name, seed, "Error")

# ==========================================
# メイン処理
# ==========================================
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # シード文字列 "0,4,6" をリスト [0, 4, 6] に変換
    seed_list = [int(s) for s in args.seeds.split(",")]
    
    print(f"Running Seeds: {seed_list}")
    print(f"Config: Width={args.beam_width}, Cands={args.num_candidates}, Mem={args.gpu_memory_utilization}")
    print(f"Output: {args.output_dir}")
    
    all_tasks = []
    for model_name, model_path in MODELS.items():
        for seed in seed_list:
            # タスクに args 自体も含める
            all_tasks.append((model_name, model_path, seed, args))
    
    print(f"Total tasks: {len(all_tasks)}")
    
    results_map = {name: [] for name in MODELS.keys()}
    
    with ProcessPoolExecutor(max_workers=args.num_gpus) as executor:
        futures = []
        for i, (m_name, m_path, seed, arg_obj) in enumerate(all_tasks):
            gpu_id = i % args.num_gpus
            futures.append(executor.submit(run_experiment, (m_name, m_path, seed, gpu_id, arg_obj)))
            
        for future in as_completed(futures):
            m_name, seed, res = future.result()
            if res is not None and res != "Error":
                results_map[m_name].append(res)

    print("\n" + "="*60)
    print(f"FINAL REPORT (Seeds: {args.seeds})")
    print("="*60)
    for name, accs in results_map.items():
        if not accs:
            print(f"{name:<10} | N/A        | N/A        | No results")
            continue
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        raw_str = ", ".join([f"{x:.1%}" for x in accs])
        print(f"{name:<10} | {mean_acc:.2%}     | ±{std_acc:.2%}     | {raw_str}")
    print("="*60)

if __name__ == "__main__":
    main()