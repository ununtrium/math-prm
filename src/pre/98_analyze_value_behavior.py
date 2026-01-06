import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import re
from latex2sympy2 import latex2sympy
from sympy import simplify

# ==========================================
# 設定
# ==========================================
INPUT_DIR = "data/experiments/evaluation_v3.0_new_prm"
OUTPUT_IMG_DIST = "analysis_score_distribution.png"
OUTPUT_IMG_TRAJ = "analysis_score_trajectory.png"

# 対象とするスコアキー
TARGET_KEY = "step_scores_new" # New PRM (v3)

# ==========================================
# ユーティリティ
# ==========================================
def extract_answer_content(text):
    if not text: return None
    matches = re.findall(r"\\boxed\{(.*?)\}", text)
    if matches: return matches[-1].strip()
    return None

def robust_float_check(pred, gold):
    try:
        def to_float(s):
            s = str(s).replace(r"\frac", "").replace("{", "(").replace("}", ")").replace("^", "**")
            return float(eval(s))
        if not any(c.isalpha() for c in str(pred)) and not any(c.isalpha() for c in str(gold)):
            return abs(to_float(pred) - to_float(gold)) < 1e-6
    except: pass
    return False

def check_correctness(pred_str, gold_str):
    if not pred_str or not gold_str: return False
    pred_str = str(pred_str).strip(); gold_str = str(gold_str).strip()
    if pred_str == gold_str: return True
    try:
        sym_pred = latex2sympy(pred_str); sym_gold = latex2sympy(gold_str)
        if simplify(sym_pred - sym_gold) == 0: return True
    except: return robust_float_check(pred_str, gold_str)
    return False

# ==========================================
# メイン解析
# ==========================================
def main():
    files = glob.glob(os.path.join(INPUT_DIR, "trial_*.json"))
    print(f"Analyzing {len(files)} files...")

    correct_step_scores = []
    incorrect_step_scores = []
    
    # 軌跡分析用 (正規化したステップ位置ごとの平均スコア)
    # 0%地点, 10%地点 ... 100%地点
    bins = 10
    traj_correct = [[] for _ in range(bins)]
    traj_incorrect = [[] for _ in range(bins)]

    for fpath in tqdm(files):
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        for item in data:
            gold = item["gold"]
            paths = item["paths"]
            scores_list = item.get(TARGET_KEY, [])
            
            extracted = [extract_answer_content(p) for p in paths]
            
            for i, ans in enumerate(extracted):
                if i >= len(scores_list): continue
                
                step_scores = scores_list[i]
                if not step_scores: continue
                
                is_correct = check_correctness(ans, gold)
                
                # 1. 全スコアの分布用
                if is_correct:
                    correct_step_scores.extend(step_scores)
                else:
                    incorrect_step_scores.extend(step_scores)
                    
                # 2. 軌跡分析用 (長さを0~1に正規化してビン詰め)
                target_traj = traj_correct if is_correct else traj_incorrect
                path_len = len(step_scores)
                
                for step_idx, score in enumerate(step_scores):
                    # 正規化位置 (0.0 ~ 0.99...)
                    norm_pos = step_idx / path_len
                    bin_idx = int(norm_pos * bins)
                    bin_idx = min(bin_idx, bins - 1)
                    target_traj[bin_idx].append(score)

    # --- プロット作成 ---
    
    # 1. スコア分布 (ヒストグラム)
    plt.figure(figsize=(10, 6))
    plt.hist(incorrect_step_scores, bins=50, alpha=0.5, label='Incorrect Paths', color='red', density=True)
    plt.hist(correct_step_scores, bins=50, alpha=0.5, label='Correct Paths', color='green', density=True)
    plt.title("Distribution of Predicted Step Scores (1.5B Inference)")
    plt.xlabel("Predicted Value (Probability)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(OUTPUT_IMG_DIST)
    print(f"Saved distribution plot to {OUTPUT_IMG_DIST}")
    
    # 2. 学習軌跡 (Trajectory)
    plt.figure(figsize=(10, 6))
    
    def get_means_errs(traj_list):
        means = [np.mean(vals) if vals else 0 for vals in traj_list]
        stds = [np.std(vals) if vals else 0 for vals in traj_list]
        # 信頼区間 (簡易)
        errs = [s / np.sqrt(len(v)) if v else 0 for s, v in zip(stds, traj_list)]
        return means, errs

    c_means, c_errs = get_means_errs(traj_correct)
    i_means, i_errs = get_means_errs(traj_incorrect)
    
    x_axis = np.linspace(0, 100, bins)
    
    plt.plot(x_axis, c_means, 'g-o', label='Correct Paths')
    plt.fill_between(x_axis, np.array(c_means)-np.array(c_errs), np.array(c_means)+np.array(c_errs), color='green', alpha=0.2)
    
    plt.plot(x_axis, i_means, 'r-o', label='Incorrect Paths')
    plt.fill_between(x_axis, np.array(i_means)-np.array(i_errs), np.array(i_means)+np.array(i_errs), color='red', alpha=0.2)
    
    plt.title("Average Score Trajectory (Progress vs Value)")
    plt.xlabel("Progress (%)")
    plt.ylabel("Predicted Value")
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(True)
    plt.savefig(OUTPUT_IMG_TRAJ)
    print(f"Saved trajectory plot to {OUTPUT_IMG_TRAJ}")

if __name__ == "__main__":
    main()