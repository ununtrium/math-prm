import os
import json
import glob
import matplotlib.pyplot as plt
import numpy as np

# 論文用にフォントサイズを調整
plt.rcParams.update({'font.size': 14})

def collect_step_counts_simple(base_dir="results"):
    """
    各ベンチマークから代表的な1モデルのみをサンプリングして、ステップ数分布を計算する。
    """
    benchmark_data = {}
    
    if not os.path.exists(base_dir):
        print(f"Error: Directory '{base_dir}' not found.")
        return None

    # 1. Benchmark層をループ
    for benchmark in sorted(os.listdir(base_dir)):
        bench_path = os.path.join(base_dir, benchmark)
        if not os.path.isdir(bench_path): continue
        
        counts = []
        found_representative = False
        
        # 2. 最初に見つかった Instruct モデルのフォルダを1つだけ選ぶ
        for gen_model in os.listdir(bench_path):
            if "Instruct" not in gen_model: continue
            gen_model_path = os.path.join(bench_path, gen_model)
            
            # 3. 最初に見つかった Teacher モデルのフォルダを1つだけ選ぶ
            for teacher_model in os.listdir(gen_model_path):
                target_dir = os.path.join(gen_model_path, teacher_model)
                if not os.path.isdir(target_dir): continue
                
                # samples_16 などの下の jsonl を全て取得
                files = glob.glob(os.path.join(target_dir, "**", "seed_*.jsonl"), recursive=True)
                
                if files:
                    print(f"Sampling benchmark: {benchmark} (Source: {teacher_model})")
                    for file_path in files:
                        with open(file_path, "r", encoding="utf-8") as f:
                            for line in f:
                                item = json.loads(line)
                                for r in item.get("responses", []):
                                    step_scores = r.get("step_scores", [])
                                    if step_scores:
                                        # 有効なステップ数を抽出
                                        valid_len = len([s for s in step_scores if s is not None])
                                        if valid_len > 0:
                                            counts.append(valid_len)
                    found_representative = True
                    break # Teacher層を抜ける
            if found_representative:
                break # Generator層を抜ける
        
        if counts:
            benchmark_data[benchmark.upper()] = counts
            
    return benchmark_data

def plot_distributions(benchmark_data, output_file="step_distribution_simple.png"):
    benchmarks = sorted(benchmark_data.keys())
    n = len(benchmarks)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 4))
    if n == 1: axes = [axes]
    
    for i, bench in enumerate(benchmarks):
        data = benchmark_data[bench]
        avg, med = np.mean(data), np.median(data)
        axes[i].hist(data, bins=np.arange(min(data), max(data)+2)-0.5, color='gray', alpha=0.7, edgecolor='black')
        axes[i].set_title(f"{bench}\nAvg: {avg:.1f}, Med: {med:.0f}")
        axes[i].set_xlabel("Steps")
        if i == 0: axes[i].set_ylabel("Frequency")
        
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Saved: {output_file}")

if __name__ == "__main__":
    data = collect_step_counts_simple()
    if data:
        plot_distributions(data)