import json
import os
import glob
import numpy as np
import math
from collections import Counter
from tqdm import tqdm

# SymPy関連
from latex2sympy2 import latex2sympy
from sympy import simplify

# ==========================================
# 設定
# ==========================================
# src/12 で保存したフォルダを指定
INPUT_DIR = "data/experiments/evaluation_v3.0_new_prm"
TARGET_MODEL_KEY = "step_scores_new" # Delta-PRMを最適化
#TARGET_MODEL_KEY = "step_scores_orm" # ORMを見たい場合はこっち

# 探索パラメータ
METRICS_TO_TRY = ["min", "mean"]
SCALES_TO_TRY = [0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]

# ==========================================
# ユーティリティ
# ==========================================
def extract_answer_content(text):
    if not text: return None
    import re
    matches = re.findall(r"\\boxed\{(.*?)\}", text)
    if matches: return matches[-1].strip()
    return None

def robust_float_check(pred, gold):
    try:
        def to_float(s):
            s = str(s).replace(r"\frac", "").replace("{", "(").replace("}", ")").replace("^", "**")
            s = s.replace(r"\left", "").replace(r"\right", "").replace(",", "")
            return float(eval(s))
        if not any(c.isalpha() for c in str(pred)) and not any(c.isalpha() for c in str(gold)):
            return abs(to_float(pred) - to_float(gold)) < 1e-6
    except:
        pass
    return False

def check_correctness(pred_str, gold_str):
    if not pred_str or not gold_str: return False
    pred_str = str(pred_str).strip()
    gold_str = str(gold_str).strip()
    if pred_str == gold_str: return True
    try:
        sym_pred = latex2sympy(pred_str)
        sym_gold = latex2sympy(gold_str)
        if simplify(sym_pred - sym_gold) == 0:
            return True
    except Exception:
        return robust_float_check(pred_str, gold_str)
    return False

# ==========================================
# メイン処理
# ==========================================
def main():
    if not os.path.exists(INPUT_DIR):
        print(f"Error: Directory {INPUT_DIR} not found.")
        return

    # ファイル検索
    files = glob.glob(os.path.join(INPUT_DIR, "trial_*.json"))
    if not files:
        print("No trial files found.")
        return
    
    print(f"Found {len(files)} trial files. Loading data...")
    
    # 全データをメモリにロード (List of List of Dicts)
    all_trials_data = []
    for fpath in files:
        with open(fpath, "r", encoding="utf-8") as f:
            all_trials_data.append(json.load(f))
            
    print(f"Loaded {len(all_trials_data)} trials. Starting optimization...")
    print("-" * 60)
    print(f"{'METRIC':<6} | {'SCALE':<5} | {'AVG ACCURACY':<15} | {'STD DEV':<10} | {'MIN':<8} | {'MAX':<8}")
    print("-" * 60)

    best_acc = 0.0
    best_config = ""

    # Grid Search
    for metric_type in METRICS_TO_TRY:
        for scale in SCALES_TO_TRY:
            
            trial_accuracies = []
            
            # 各トライアルごとに精度を計算
            for trial_data in all_trials_data:
                correct_count = 0
                total_problems = len(trial_data)
                
                for item in trial_data:
                    gold = item["gold"]
                    paths = item["paths"]
                    step_scores_list = item[TARGET_MODEL_KEY] # Delta or ORM
                    
                    extracted_answers = [extract_answer_content(p) for p in paths]
                    valid_indices = [i for i, a in enumerate(extracted_answers) if a]
                    
                    if not valid_indices: continue
                    
                    # 投票
                    votes = {}
                    for i in valid_indices:
                        ans = extracted_answers[i]
                        steps = step_scores_list[i]
                        
                        # スコア集計
                        if not steps: val = -99.0
                        elif metric_type == "min": val = min(steps)
                        else: val = np.mean(steps)
                        
                        # 重み付け
                        weight = math.exp(val * scale)
                        
                        if ans not in votes: votes[ans] = 0
                        votes[ans] += weight
                    
                    # 判定
                    if votes:
                        best_ans = max(votes, key=votes.get)
                        if check_correctness(best_ans, gold):
                            correct_count += 1
                
                # そのトライアルの精度
                trial_accuracies.append(correct_count / total_problems)
            
            # 統計
            avg_acc = np.mean(trial_accuracies)
            std_acc = np.std(trial_accuracies)
            min_acc = np.min(trial_accuracies)
            max_acc = np.max(trial_accuracies)
            
            # ハイライト表示
            marker = ""
            if avg_acc > best_acc:
                best_acc = avg_acc
                best_config = f"{metric_type.upper()} (Scale={scale})"
                marker = "*"
            
            print(f"{metric_type.upper():<6} | {scale:<5.1f} | {avg_acc:.2%} ± {std_acc:.2%} | {min_acc:.2%} | {max_acc:.2%} {marker}")

    print("-" * 60)
    print(f"🏆 BEST CONFIGURATION (Avg of {len(files)} trials):")
    print(f"   {best_config} -> {best_acc:.2%}")

if __name__ == "__main__":
    main()