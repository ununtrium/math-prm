import json
import os
import re
import numpy as np
import math
from collections import Counter
from tqdm import tqdm

from latex2sympy2 import latex2sympy
from sympy import simplify

# ==========================================
# 1. 設定パラメータ
# ==========================================
INPUT_FILE = "data/math500_results_full_scores_30k_v1.0.json"

# ==========================================
# 2. ユーティリティ関数
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
# 3. 集計戦略 (Aggregators)
# ==========================================

def agg_min(scores):
    """Min: 致命的なミスを弾く (Strict)"""
    return min(scores) if scores else -99.0

def agg_mean(scores):
    """Arithmetic Mean: 対数空間での幾何平均に相当 (Robust)"""
    return np.mean(scores) if scores else -99.0

def agg_sum(scores):
    """Sum: 同時確率に相当 (長いパスほど不利になる)"""
    return sum(scores) if scores else -99.0

def agg_geomean_sigmoid(scores):
    """
    Sigmoid Geometric Mean: 
    一度 0~1 の確率に戻してから幾何平均をとる (Strict Probability View)
    """
    if not scores: return -99.0
    # 1. Sigmoid で 0~1 に変換 (スコアが -2~2 なので)
    probs = [1 / (1 + math.exp(-s)) for s in scores]
    
    # 2. 対数をとって平均 (対数空間での算術平均 = 元の空間での幾何平均)
    # log_mean = sum(log(p)) / N
    log_probs = [math.log(p + 1e-9) for p in probs] # 1e-9はゼロ除算防止
    log_mean = sum(log_probs) / len(log_probs)
    
    # 3. 元に戻す (比較だけならlog_meanのままで良いが、念の為)
    return math.exp(log_mean)

def agg_last(scores):
    """Last: 最終ステップのスコアのみ採用 (ORM近似)"""
    return scores[-1] if scores else -99.0

def agg_dynamic(scores):
    """
    Dynamic Score: 平均値 + 標準偏差
    「波のあるパス」を優遇する
    """
    if not scores: return -99.0
    return np.mean(scores) + 0.5 * np.std(scores) # 係数0.5は要調整

def agg_max(scores):
    """
    Max Score: パスの中で最も高かった瞬間のスコアを採用
    「一度でも素晴らしいひらめきがあれば採用」という戦略
    """
    if not scores: return -99.0
    return max(scores)


# ==========================================
# 4. メイン集計処理
# ==========================================

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    print(f"Loading results from {INPUT_FILE}...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        results = json.load(f)
    
    total_problems = len(results)
    
    # 指標カウンター
    metrics = {
        "pass1": 0,
        "maj_vote": 0,
        "bon_min": 0,
        "bon_mean": 0,
        "bon_sum": 0,
        "bon_geomean": 0,  # ★追加
        "weighted_vote_mean": 0,
        "bon_last": 0,
        "bon_dynamic": 0,
        "bon_max": 0,
    }
    
    total_paths = 0

    for item in tqdm(results, desc="Analyzing"):
        gold = item["gold"]
        paths = item["paths"]
        step_scores_list = item["step_scores"]
        
        extracted_answers = [extract_answer_content(p) for p in paths]
        
        # 集計スコアの計算
        scores_min = [agg_min(s) for s in step_scores_list]
        scores_mean = [agg_mean(s) for s in step_scores_list]
        scores_sum = [agg_sum(s) for s in step_scores_list]
        scores_geo = [agg_geomean_sigmoid(s) for s in step_scores_list] # ★追加
        scores_last = [agg_last(s) for s in step_scores_list] # ★追加
        scores_dynamic = [agg_dynamic(s) for s in step_scores_list]
        scores_max = [agg_max(s) for s in step_scores_list]
        
        valid_indices = [i for i, a in enumerate(extracted_answers) if a]
        
        # --- 1. Pass@1 ---
        for i, ans in enumerate(extracted_answers):
            if check_correctness(ans, gold): metrics["pass1"] += 1
        total_paths += len(paths)
        
        if not valid_indices: continue

        # --- 2. Majority Vote ---
        valid_answers = [extracted_answers[i] for i in valid_indices]
        maj_ans = Counter(valid_answers).most_common(1)[0][0]
        if check_correctness(maj_ans, gold): metrics["maj_vote"] += 1
        
        # --- 3. Best-of-N Variations ---
        
        # Min
        if check_correctness(extracted_answers[np.argmax(scores_min)], gold): 
            metrics["bon_min"] += 1
            
        # Mean (Approx GeoMean)
        if check_correctness(extracted_answers[np.argmax(scores_mean)], gold): 
            metrics["bon_mean"] += 1
            
        # Sum (Joint Prob)
        if check_correctness(extracted_answers[np.argmax(scores_sum)], gold): 
            metrics["bon_sum"] += 1
            
        # GeoMean (Sigmoid Strict) ★追加
        if check_correctness(extracted_answers[np.argmax(scores_geo)], gold): 
            metrics["bon_geomean"] += 1

        # --- 4. Weighted Vote (Best Strategy) ---
        # Meanスコアを使って重み付け
        exp_votes = {}
        for i in valid_indices:
            ans = extracted_answers[i]
            score = scores_mean[i] 
            weight = np.exp(score * 2.0)
            if ans not in exp_votes: exp_votes[ans] = 0
            exp_votes[ans] += weight
            
        if check_correctness(max(exp_votes, key=exp_votes.get), gold): 
            metrics["weighted_vote_mean"] += 1

        # Best-of-N Last の判定
        if check_correctness(extracted_answers[np.argmax(scores_last)], gold): 
            metrics["bon_last"] += 1

        if check_correctness(extracted_answers[np.argmax(scores_dynamic)], gold): 
            metrics["bon_dynamic"] += 1

        if check_correctness(extracted_answers[np.argmax(scores_max)], gold): 
            metrics["bon_max"] += 1

    # --- 結果出力 ---
    print("\n" + "="*50)
    print(f"  FINAL AGGREGATION COMPARISON (N={len(paths)})")
    print("="*50)
    print(f"1. Pass@1 (Avg)      : {metrics['pass1']/total_paths:.2%}")
    print(f"2. Majority Vote     : {metrics['maj_vote']/total_problems:.2%}")
    print("-" * 30)
    print(f"3. BoN (Min)         : {metrics['bon_min']/total_problems:.2%}")
    print(f"4. BoN (Sum)         : {metrics['bon_sum']/total_problems:.2%}")
    print(f"5. BoN (Mean)        : {metrics['bon_mean']/total_problems:.2%}  (Log-GeoMean)")
    print(f"6. BoN (Sigmoid-Geo) : {metrics['bon_geomean']/total_problems:.2%}  (Strict GeoMean)")
    print("-" * 30)
    print(f"7. Weighted Vote     : {metrics['weighted_vote_mean']/total_problems:.2%}")
    print("-" * 30)
    print(f"8. BoN (Last)        : {metrics['bon_last']/total_problems:.2%}  (ORM-like)") # ★追加
    print(f"9. BoN (Dynamic)     : {metrics['bon_dynamic']/total_problems:.2%}  (Mean + Std)")
    print(f"10. BoN (max)     : {metrics['bon_max']/total_problems:.2%}  (max)")
    print("="*50)

if __name__ == "__main__":
    main()