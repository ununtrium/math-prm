import os
import json
import glob
import pandas as pd
import argparse
import numpy as np
from collections import Counter

# ==========================================
# スコア計算ロジック
# ==========================================
def calculate_path_score(step_scores, method="min"):
    """ステップごとのスコアリストからパス全体のスコアを計算する"""
    if not step_scores: return -float('inf')
    
    # Noneを除外（念のため）
    valid_scores = [s for s in step_scores if s is not None]
    if not valid_scores: return -float('inf')

    if method == "min":
        return min(valid_scores)
    elif method == "mean":
        return np.mean(valid_scores)
    elif method == "last":
        return valid_scores[-1]
    elif method == "sum": 
        return sum(valid_scores)
    else:
        return min(valid_scores)

def get_majority_vote(responses):
    """Majority Voteによる予測結果の正誤を返す"""
    answers = [r["extracted"] for r in responses if r.get("extracted")]
    if not answers:
        return False
    vote = Counter(answers).most_common(1)[0][0]
    for r in responses:
        if r.get("extracted") == vote:
            return r.get("is_correct", False)
    return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results", help="Base directory of scored results")
    parser.add_argument("--output_file", type=str, default="best_of_n_report.md", help="Output markdown file name")
    # 特定のGeneratorやSamplesだけに絞り込みたい場合のための引数
    parser.add_argument("--target_gen", type=str, default="Instruct", help="Filter for Generator name (e.g., 'Instruct')")
    args = parser.parse_args()

    AGG_METHODS = ["min", "mean", "last", "sum"]
    
    # 探索
    files = glob.glob(os.path.join(args.results_dir, "*", "*", "*", "*", "seed_*.jsonl"))
    stats = {}

    print(f"Found {len(files)} result files. Processing...")

    for file_path in files:
        parts = file_path.split(os.sep)
        if len(parts) < 5: continue
        
        bench_name = parts[-5]
        gen_model = parts[-4]
        prm_model = parts[-3]
        samples_dir = parts[-2]
        
        # ★フィルタリング追加: 指定されたGenerator以外はスキップ
        if args.target_gen and args.target_gen not in gen_model:
            continue

        if bench_name not in stats: stats[bench_name] = {}
        if gen_model not in stats[bench_name]: stats[bench_name][gen_model] = {}
        if samples_dir not in stats[bench_name][gen_model]: stats[bench_name][gen_model][samples_dir] = {}
        if prm_model not in stats[bench_name][gen_model][samples_dir]: stats[bench_name][gen_model][samples_dir][prm_model] = []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                items = [json.loads(line) for line in f]
                
            total = len(items)
            if total == 0: continue

            seed_metrics = {m: 0 for m in AGG_METHODS}
            seed_metrics["majority_vote"] = 0
            
            for item in items:
                responses = item["responses"]
                if not responses: continue

                if get_majority_vote(responses):
                    seed_metrics["majority_vote"] += 1
                
                for method in AGG_METHODS:
                    best_score = -float('inf')
                    best_is_correct = False
                    for resp in responses:
                        if not resp.get("step_scores"): continue
                        score = calculate_path_score(resp["step_scores"], method)
                        if score > best_score:
                            best_score = score
                            best_is_correct = resp.get("is_correct", False)
                    if best_is_correct:
                        seed_metrics[method] += 1

            final_metrics = {k: (v / total * 100) for k, v in seed_metrics.items()}
            stats[bench_name][gen_model][samples_dir][prm_model].append(final_metrics)

        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    # ==========================================
    # レポート作成 (Best Methodの特定)
    # ==========================================
    display_records = []
    
    for bench in stats:
        for gen in stats[bench]:
            for s_dir in stats[bench][gen]:
                for prm in stats[bench][gen][s_dir]:
                    results_list = stats[bench][gen][s_dir][prm]
                    if not results_list: continue
                    
                    # 各指標の平均値を計算
                    avg_metrics = {k: [] for k in results_list[0].keys()}
                    for res in results_list:
                        for k, v in res.items():
                            avg_metrics[k].append(v)
                    
                    mean_values = {k: np.mean(v) for k, v in avg_metrics.items()}
                    
                    # --- Best Method の特定 ---
                    # min, mean, last, sum の中で最も平均スコアが高いものを探す
                    best_method_name = "N/A"
                    best_method_score = -1.0
                    
                    for method in AGG_METHODS:
                        score = mean_values.get(method, 0)
                        if score > best_method_score:
                            best_method_score = score
                            best_method_name = method
                    
                    # 行データ作成
                    row = {
                        "Benchmark": bench,
                        "Generator": gen,
                        "Samples": s_dir,
                        "PRM": prm,
                        "Seeds": len(results_list),
                        "MajVote": f"{mean_values['majority_vote']:.2f}",
                        # 全てのメソッドの結果も載せるが...
                        "Min": f"{mean_values['min']:.2f}",
                        "Mean": f"{mean_values['mean']:.2f}",
                        "Last": f"{mean_values['last']:.2f}",
                        # ★ここが重要: 最高スコアのメソッドとそのスコア
                        "BEST_METHOD": best_method_name.upper(),
                        "BEST_SCORE": f"{best_method_score:.2f}"
                    }
                    display_records.append(row)

    df = pd.DataFrame(display_records)
    
    if df.empty:
        print("No results found matching the criteria.")
        return

    # 表示カラムの整理
    cols = ["Benchmark", "Samples", "PRM", "BEST_METHOD", "BEST_SCORE", "MajVote"]
    
    # ソート (Benchmark -> Samples -> BEST_SCORE降順)
    df = df.sort_values(by=["Benchmark", "Samples", "BEST_SCORE"], ascending=[True, True, False])

    print("\n=== Best Aggregation Method Summary ===")
    print(df[cols].to_markdown(index=False))
    
    # 保存
    with open(args.output_file, "w", encoding="utf-8") as f:
        f.write("# Best Aggregation Method Summary\n\n")
        f.write(df[cols].to_markdown(index=False))
    print(f"\nReport saved to {args.output_file}")

if __name__ == "__main__":
    main()