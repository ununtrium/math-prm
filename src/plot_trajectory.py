import os
import json
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt

def load_and_normalize_scores(target_dir, n_bins=20):
    """
    各パスの長さを 0.0(開始)〜1.0(終了) に正規化し、
    n_bins個のビン（区間）に割り振って平均化する。
    """
    files = glob.glob(os.path.join(target_dir, "**", "seed_*.jsonl"), recursive=True)
    
    if not files:
        print(f"No files found in {target_dir}")
        return None, None

    # ビンごとのスコアリスト
    # bins[0] = 0%地点(Start), bins[-1] = 100%地点(End)
    correct_bins = [[] for _ in range(n_bins)]
    incorrect_bins = [[] for _ in range(n_bins)]
    
    print(f"Loading and normalizing paths from {target_dir}...")

    for file_path in files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    item = json.loads(line)
                    responses = item.get("responses", [])
                    
                    for r in responses:
                        step_scores = r.get("step_scores", [])
                        if not step_scores: continue
                        
                        valid_scores = [float(s) for s in step_scores if s is not None]
                        L = len(valid_scores)
                        if L == 0: continue

                        is_correct = r.get("is_correct", False)
                        target_bins = correct_bins if is_correct else incorrect_bins
                        
                        # 正規化処理: 各ステップが全体の何%の位置にあるかを計算
                        for i, score in enumerate(valid_scores):
                            # 進行度 (0.0 〜 1.0)
                            if L == 1:
                                progress = 1.0
                            else:
                                progress = i / (L - 1)
                            
                            # ビンのインデックス決定 (0 〜 n_bins-1)
                            bin_idx = int(progress * (n_bins - 1))
                            
                            # 念のためクリップ
                            bin_idx = max(0, min(bin_idx, n_bins - 1))
                            
                            target_bins[bin_idx].append(score)
                            
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    return correct_bins, incorrect_bins

def aggregate_bins(bins):
    """ビンごとの平均と標準誤差を計算"""
    means = []
    sems = []
    indices = []
    
    for i, scores in enumerate(bins):
        if len(scores) < 10: # サンプルが少なすぎる区間は除外
            means.append(np.nan)
            sems.append(np.nan)
        else:
            means.append(np.mean(scores))
            sems.append(np.std(scores) / np.sqrt(len(scores)))
        indices.append(i / (len(bins) - 1)) # 0.0 ~ 1.0
        
    return indices, means, sems

def plot_normalized_trajectories(c_bins, i_bins, output_file="normalized_trajectory.png"):
    c_x, c_mean, c_sem = aggregate_bins(c_bins)
    i_x, i_mean, i_sem = aggregate_bins(i_bins)

    plt.figure(figsize=(10, 6))
    
    # 正解パス (Blue)
    plt.plot(c_x, c_mean, label='Correct Paths', color='blue', linewidth=2)
    plt.fill_between(c_x, 
                     np.array(c_mean) - np.array(c_sem), 
                     np.array(c_mean) + np.array(c_sem), 
                     color='blue', alpha=0.15)

    # 不正解パス (Red)
    plt.plot(i_x, i_mean, label='Incorrect Paths', color='red', linewidth=2, linestyle='--')
    plt.fill_between(i_x, 
                     np.array(i_mean) - np.array(i_sem), 
                     np.array(i_mean) + np.array(i_sem), 
                     color='red', alpha=0.15)

    plt.xlabel('Reasoning Progress (Normalized)')
    plt.ylabel('Average PRM Score')
    plt.title('PRM Score Trajectory (Normalized Length)')
    
    # 軸の設定 (0%〜100%)
    plt.xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], ['0%', '20%', '40%', '60%', '80%', '100%'])
    plt.ylim(0.0, 1.05)
    
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"\nPlot saved to {output_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_dir", type=str, required=True)
    parser.add_argument("--output", type=str, default="normalized_trajectory.png")
    args = parser.parse_args()

    c_bins, i_bins = load_and_normalize_scores(args.target_dir)
    
    if c_bins is None: return
    
    plot_normalized_trajectories(c_bins, i_bins, args.output)

if __name__ == "__main__":
    main()