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
# 1. 設定パラメータ
# ==========================================
INPUT_DIR = "data/experiments/evaluation_v3.0_new_orm_7b"
TARGET_STEP_SCORES_KEY = "step_scores_new" 

# ==========================================
# 2. ユーティリティ
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
# 3. 集計戦略 (Sumを追加)
# ==========================================
def agg_min(scores): return min(scores) if scores else -99.0
def agg_mean(scores): return np.mean(scores) if scores else -99.0
def agg_last(scores): return scores[-1] if scores else -99.0
def agg_max(scores): return max(scores) if scores else -99.0
def agg_sum(scores): return sum(scores) if scores else -99.0 # ★追加: Math Shepherd流

# ==========================================
# 4. メイン集計処理
# ==========================================
def calculate_metrics_for_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        results = json.load(f)
    
    # カウンター (sumを追加)
    metrics = {
        "pass1": 0, "maj_vote": 0,
        "bon_min": 0, "bon_mean": 0, "bon_last": 0, "bon_max": 0, "bon_sum": 0,
        "weighted_vote": 0
    }
    total_paths = 0
    total_problems = len(results)
    
    for item in results:
        gold = item["gold"]
        paths = item["paths"]
        step_scores_list = item[TARGET_STEP_SCORES_KEY] 
        
        extracted = [extract_answer_content(p) for p in paths]
        valid_idx = [i for i, a in enumerate(extracted) if a]
        
        # Pass@1
        for i, ans in enumerate(extracted):
            if check_correctness(ans, gold): metrics["pass1"] += 1
        total_paths += len(paths)
        
        if not valid_idx: continue

        # Majority Vote
        valid_ans = [extracted[i] for i in valid_idx]
        maj = Counter(valid_ans).most_common(1)[0][0]
        if check_correctness(maj, gold): metrics["maj_vote"] += 1
        
        # 集計スコア計算
        s_min = [agg_min(s) for s in step_scores_list]
        s_mean = [agg_mean(s) for s in step_scores_list]
        s_last = [agg_last(s) for s in step_scores_list]
        s_max = [agg_max(s) for s in step_scores_list]
        s_sum = [agg_sum(s) for s in step_scores_list] # ★追加

        # Best-of-N
        if check_correctness(extracted[np.argmax(s_min)], gold): metrics["bon_min"] += 1
        if check_correctness(extracted[np.argmax(s_mean)], gold): metrics["bon_mean"] += 1
        if check_correctness(extracted[np.argmax(s_last)], gold): metrics["bon_last"] += 1
        if check_correctness(extracted[np.argmax(s_max)], gold): metrics["bon_max"] += 1
        if check_correctness(extracted[np.argmax(s_sum)], gold): metrics["bon_sum"] += 1 # ★追加
        
        # Weighted Vote (Mean base)
        votes = {}
        for i in valid_idx:
            ans = extracted[i]
            score = s_mean[i]
            weight = math.exp(score * 0.5)
            votes[ans] = votes.get(ans, 0) + weight
            
        if check_correctness(max(votes, key=votes.get), gold): 
            metrics["weighted_vote"] += 1

    return {k: v / total_problems for k, v in metrics.items() if k != "pass1"}

def main():
    if not os.path.exists(INPUT_DIR):
        print("Directory not found.")
        return

    files = glob.glob(os.path.join(INPUT_DIR, "trial_*.json"))
    print(f"Found {len(files)} trial files. Aggregating...")
    
    history = {k: [] for k in ["maj_vote", "bon_min", "bon_mean", "bon_last", "bon_max", "bon_sum", "weighted_vote"]}
    
    for fpath in tqdm(files):
        res = calculate_metrics_for_file(fpath)
        for k, v in res.items():
            if k in history: history[k].append(v)
            
    print("\n" + "="*60)
    print(f"  NEW PRM EVALUATION (Avg of {len(files)} Trials)")
    print("="*60)
    
    def p(label, key):
        vals = history[key]
        print(f"{label:<20} | {np.mean(vals):.2%} ± {np.std(vals):.2%}")

    p("Majority Vote", "maj_vote")
    print("-" * 40)
    p("BoN (Min)", "bon_min")
    p("BoN (Mean)", "bon_mean")
    p("BoN (Last)", "bon_last")
    p("BoN (Max)", "bon_max")
    p("BoN (Sum)", "bon_sum")     # ★追加
    print("-" * 40)
    p("Weighted Vote", "weighted_vote")
    print("="*60)

if __name__ == "__main__":
    main()