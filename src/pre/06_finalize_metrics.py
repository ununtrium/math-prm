import json
import os
import numpy as np
from collections import Counter
from tqdm import tqdm
import math

# 正解判定ライブラリ
from latex2sympy2 import latex2sympy
from sympy import simplify

# ==========================================
# 設定
# ==========================================
INPUT_FILE = "data/math500_results_orm_fixed_set_30k_v1.0.json"

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
# メイン処理: 最適な重み付けを探す
# ==========================================
def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        results = json.load(f)
    
    total_problems = len(results)
    print(f"Optimizing Weighted Voting on {total_problems} problems...")

    # 探索するパラメータ (Temperature Scaling)
    # weight = exp(score * scale)
    # scaleが大きいほど、スコア差が強調される（良いものだけが投票権を持つ）
    scales_to_try = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    
    best_acc = 0
    best_config = ""

    # ベースライン計算
    maj_correct = 0
    for item in results:
        gold = item["gold"]
        paths = item["paths"]
        extracted_answers = [extract_answer_content(p) for p in paths]
        valid_indices = [i for i, a in enumerate(extracted_answers) if a]
        if valid_indices:
            valid_answers = [extracted_answers[i] for i in valid_indices]
            vote = Counter(valid_answers).most_common(1)[0][0]
            if check_correctness(vote, gold): maj_correct += 1
    
    baseline_acc = maj_correct / total_problems
    print(f"Baseline (Majority Vote): {baseline_acc:.2%}")
    print("-" * 40)

    # Grid Search
    for use_metric in ["min", "mean"]: # Minを使うかMeanを使うか
        for scale in scales_to_try:
            current_correct = 0
            
            for item in results:
                gold = item["gold"]
                paths = item["paths"]
                step_scores_list = item["step_scores"]
                
                extracted_answers = [extract_answer_content(p) for p in paths]
                valid_indices = [i for i, a in enumerate(extracted_answers) if a]
                
                if not valid_indices: continue
                
                votes = {}
                for i in valid_indices:
                    ans = extracted_answers[i]
                    steps = step_scores_list[i]
                    
                    # スコアの決定
                    if not steps: val = -99.0
                    elif use_metric == "min": val = min(steps)
                    else: val = np.mean(steps)
                    
                    # 重み計算: exp(val * scale)
                    # Minスコアが高い(=致命的ミスがない)パスに強い投票権を与える
                    weight = math.exp(val * scale)
                    
                    if ans not in votes: votes[ans] = 0
                    votes[ans] += weight
                
                # 最大票の答え
                best_ans = max(votes, key=votes.get)
                if check_correctness(best_ans, gold):
                    current_correct += 1
            
            acc = current_correct / total_problems
            print(f"Metric: {use_metric.upper()}, Scale: {scale:.1f} -> Acc: {acc:.2%}")
            
            if acc > best_acc:
                best_acc = acc
                best_config = f"{use_metric.upper()} (Scale={scale})"

    print("-" * 40)
    print(f"🏆 BEST RESULT: {best_acc:.2%} using {best_config}")
    if best_acc > baseline_acc:
        print(f"✅ Improvement over Majority Vote: +{best_acc - baseline_acc:.2%}")
    else:
        print(f"⚠️ Could not beat Majority Vote significantly.")

if __name__ == "__main__":
    main()