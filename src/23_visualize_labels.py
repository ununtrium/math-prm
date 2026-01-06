import json
import argparse
import math
import numpy as np
from tqdm import tqdm

def stable_sigmoid(x):
    """オーバーフロー対策付きシグモイド"""
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    else:
        z = math.exp(x)
        return z / (1 + z)

def print_stacked_histogram(correct_scores, incorrect_scores, bins=10):
    all_scores = correct_scores + incorrect_scores
    if not all_scores:
        print("No data found.")
        return

    min_val, max_val = 0.0, 1.0
    hist_correct, bin_edges = np.histogram(correct_scores, bins=bins, range=(min_val, max_val))
    hist_incorrect, _ = np.histogram(incorrect_scores, bins=bins, range=(min_val, max_val))
    
    max_count = max(hist_correct + hist_incorrect)
    scale = 50.0 / max_count if max_count > 0 else 1.0

    print(f"\n{'Range':<12} | {'Distribution (O=Incorrect, *=Correct)':<50} | {'Inc.':<6} | {'Corr.':<6} | {'Total'}")
    print("-" * 100)

    for i in range(bins):
        c_count = hist_correct[i]
        i_count = hist_incorrect[i]
        total = c_count + i_count
        
        # O = Incorrect, * = Correct
        bar_i = "O" * int(i_count * scale)
        bar_c = "*" * int(c_count * scale)
        
        range_str = f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}"
        print(f"{range_str:<12} | {bar_i}{bar_c:<{50-len(bar_i)}} | {i_count:<6} | {c_count:<6} | {total}")

    print("-" * 100)
    print(f"Total: {len(all_scores)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="data/prm_train_ensemble_2M.jsonl")
    # シミュレーションしたいパラメータ
    parser.add_argument("--alpha", type=float, default=1.0, help="Scale factor")
    parser.add_argument("--tau", type=float, default=-5.0, help="Threshold (shift)")
    args = parser.parse_args()

    print(f"Loading {args.input_file}...")
    print(f"Applying formula: Score = Sigmoid( {args.alpha} * (LogProb - ({args.tau})) )")
    
    correct_scores = []
    incorrect_scores = []
    
    # 元のログ確率分布も確認用
    raw_log_probs = []
    
    with open(args.input_file, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            try:
                rec = json.loads(line)
                
                # 生確率から対数を逆算してもいいし、log_probがあればそれを使う
                # ここでは安全のため log_prob を使う (マージ時に計算済み想定)
                log_prob = rec.get("log_prob", -100.0)
                if log_prob is None: log_prob = -100.0
                
                # 計算
                logit = args.alpha * (log_prob - args.tau)
                score = stable_sigmoid(logit)
                
                is_correct = rec.get("is_outcome_correct", False)
                
                raw_log_probs.append(log_prob)
                
                if is_correct:
                    correct_scores.append(score)
                else:
                    incorrect_scores.append(score)
            except:
                continue

    # 1. 元のログ確率の統計
    print("\n" + "="*60)
    print("1. ORIGINAL LOG_PROB STATS")
    print("="*60)
    print(f"Mean: {np.mean(raw_log_probs):.4f}")
    print(f"Median: {np.median(raw_log_probs):.4f}")
    print(f"Min: {min(raw_log_probs):.4f}, Max: {max(raw_log_probs):.4f}")

    # 2. 新しいスコア分布
    print("\n" + "="*60)
    print(f"2. TRANSFORMED SCORE DISTRIBUTION (alpha={args.alpha}, tau={args.tau})")
    print("="*60)
    
    mean_c = np.mean(correct_scores) if correct_scores else 0
    mean_i = np.mean(incorrect_scores) if incorrect_scores else 0
    
    print(f"Correct Path   : Mean = {mean_c:.4f}, Median = {np.median(correct_scores):.4f}")
    print(f"Incorrect Path : Mean = {mean_i:.4f}, Median = {np.median(incorrect_scores):.4f}")
    print(f"Gap            : {mean_c - mean_i:.4f} (Goal: > 0.1)")
    
    print_stacked_histogram(correct_scores, incorrect_scores)
    
    # 3. アドバイス
    print("\n[Diagnosis]")
    if 0.4 <= np.mean(correct_scores) <= 0.9 and np.mean(incorrect_scores) < 0.4:
        print("EXCELLENT: The distribution is well spread out.")
    elif mean_c - mean_i < 0.1:
        print("WARNING: Separation is still poor. Try changing --tau.")
        print("  - If mostly 0.0: Decrease tau (e.g., -6.0, -7.0)")
        print("  - If mostly 1.0: Increase tau (e.g., -3.0, -2.0)")
    else:
        print("GOOD: Better than Raw Prob.")

if __name__ == "__main__":
    main()