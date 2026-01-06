import json
import os
import numpy as np
from collections import Counter
from tqdm import tqdm

# 正解判定ライブラリ
from latex2sympy2 import latex2sympy
from sympy import simplify

# ==========================================
# 設定
# ==========================================
INPUT_FILE = "data/math500_results_full_scores_30k_v1.0.json" # N=64の結果ファイル

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
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        results = json.load(f)
    
    total_problems = len(results)
    print(f"Loaded {total_problems} problems (N={len(results[0]['paths'])}).")

    # 探索するKの値 (上位K個で多数決)
    k_values = [1, 2, 4, 8, 16, 32, 64]
    
    print("\n--- Running Top-K Filtering Analysis ---")
    
    # 比較用ベースライン (全データの単純多数決)
    maj_correct = 0
    for item in results:
        gold = item["gold"]
        extracted_answers = [extract_answer_content(p) for p in item["paths"]]
        valid = [a for a in extracted_answers if a]
        if valid:
            vote = Counter(valid).most_common(1)[0][0]
            if check_correctness(vote, gold): maj_correct += 1
    print(f"Baseline (Majority Vote @ All): {maj_correct/total_problems:.2%}")
    print("-" * 40)

    # Top-K ループ
    for use_metric in ["min", "mean"]: # スコア基準
        print(f"\nMetric: {use_metric.upper()}")
        
        for k in k_values:
            if k > len(results[0]["paths"]): continue
            
            correct_count = 0
            
            for item in results:
                gold = item["gold"]
                paths = item["paths"]
                step_scores_list = item["step_scores"]
                
                extracted_answers = [extract_answer_content(p) for p in paths]
                
                # スコア計算
                path_scores = []
                for steps in step_scores_list:
                    if not steps: val = -99.0
                    elif use_metric == "min": val = min(steps)
                    else: val = np.mean(steps)
                    path_scores.append(val)
                
                # 有効なインデックス
                valid_indices = [i for i, a in enumerate(extracted_answers) if a]
                if not valid_indices: continue
                
                # スコア順にソート
                scored_indices = [(i, path_scores[i]) for i in valid_indices]
                scored_indices.sort(key=lambda x: x[1], reverse=True)
                
                # Top-K 抽出
                top_indices = [x[0] for x in scored_indices[:k]]
                top_answers = [extracted_answers[i] for i in top_indices]
                
                # 多数決
                if top_answers:
                    vote = Counter(top_answers).most_common(1)[0][0]
                    if check_correctness(vote, gold):
                        correct_count += 1
            
            acc = correct_count / total_problems
            print(f"  Top-{k:<2}: {acc:.2%}")

if __name__ == "__main__":
    main()