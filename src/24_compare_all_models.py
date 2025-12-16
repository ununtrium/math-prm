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
# src/22 で保存したフォルダ (全モデルのスコアが入っているはず)
INPUT_DIR = "data/experiments/evaluation_v3.0_new_prm"

# Weighted Voteの設定
WV_SCALE = 0.5 

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
# 3. 集計戦略 (全種類)
# ==========================================
def agg_min(scores): return min(scores) if scores else -99.0
def agg_mean(scores): return np.mean(scores) if scores else -99.0
def agg_last(scores): return scores[-1] if scores else -99.0
def agg_max(scores): return max(scores) if scores else -99.0 # スパイク検知
def agg_sum(scores): return sum(scores) if scores else -99.0 # 累積報酬

# ==========================================
# 4. メイン集計処理
# ==========================================
def calculate_metrics_for_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        results = json.load(f)
    
    # モデル定義 (キー名と表示名のマッピング)
    models = [
        {"key": "step_scores_delta", "name": "Old Delta (v1)"},
        {"key": "step_scores_orm",   "name": "ORM (Baseline)"},
        {"key": "step_scores_new",   "name": "New Delta (v2)"}
    ]
    
    # カウンター初期化
    # metrics[model_name][metric_name] = count
    metrics = {m["name"]: {
        "min": 0, "mean": 0, "last": 0, "max": 0, "sum": 0, "weighted": 0
    } for m in models}
    
    metrics["pass1"] = 0
    metrics["maj"] = 0
    
    total_paths = 0
    total_problems = len(results)
    
    for item in results:
        gold = item["gold"]
        paths = item["paths"]
        
        extracted = [extract_answer_content(p) for p in paths]
        valid_idx = [i for i, a in enumerate(extracted) if a]
        
        # --- Common Metrics ---
        for i, ans in enumerate(extracted):
            if check_correctness(ans, gold): metrics["pass1"] += 1
        total_paths += len(paths)
        
        if not valid_idx: continue

        valid_ans = [extracted[i] for i in valid_idx]
        maj = Counter(valid_ans).most_common(1)[0][0]
        if check_correctness(maj, gold): metrics["maj"] += 1
        
        # --- Model Specific Metrics ---
        for model in models:
            key = model["key"]
            name = model["name"]
            
            # データが存在しない場合はスキップ (念の為)
            if key not in item: continue
            
            step_scores_list = item[key]
            
            # 集計スコア
            s_min = [agg_min(s) for s in step_scores_list]
            s_mean = [agg_mean(s) for s in step_scores_list]
            s_last = [agg_last(s) for s in step_scores_list]
            s_max = [agg_max(s) for s in step_scores_list]
            s_sum = [agg_sum(s) for s in step_scores_list]
            
            # Best-of-N
            if check_correctness(extracted[np.argmax(s_min)], gold): metrics[name]["min"] += 1
            if check_correctness(extracted[np.argmax(s_mean)], gold): metrics[name]["mean"] += 1
            if check_correctness(extracted[np.argmax(s_last)], gold): metrics[name]["last"] += 1
            if check_correctness(extracted[np.argmax(s_max)], gold): metrics[name]["max"] += 1
            if check_correctness(extracted[np.argmax(s_sum)], gold): metrics[name]["sum"] += 1
            
            # Weighted Vote (Mean base)
            votes = {}
            for i in valid_idx:
                ans = extracted[i]
                score = s_mean[i]
                weight = math.exp(score * WV_SCALE)
                votes[ans] = votes.get(ans, 0) + weight
            
            if check_correctness(max(votes, key=votes.get), gold): 
                metrics[name]["weighted"] += 1

    # 精度計算
    final_res = {}
    final_res["pass1"] = metrics["pass1"] / total_paths
    final_res["maj"] = metrics["maj"] / total_problems
    
    for model in models:
        name = model["name"]
        final_res[name] = {k: v / total_problems for k, v in metrics[name].items()}
        
    return final_res

def main():
    if not os.path.exists(INPUT_DIR):
        print("Directory not found.")
        return

    files = glob.glob(os.path.join(INPUT_DIR, "trial_*.json"))
    print(f"Found {len(files)} trials. Aggregating...")
    
    # 履歴コンテナ
    # history[model][metric] = [val1, val2, val3]
    history = {
        "Old Delta (v1)": {"min": [], "mean": [], "last": [], "max": [], "sum": [], "weighted": []},
        "ORM (Baseline)": {"min": [], "mean": [], "last": [], "max": [], "sum": [], "weighted": []},
        "New Delta (v2)": {"min": [], "mean": [], "last": [], "max": [], "sum": [], "weighted": []},
        "Common": {"pass1": [], "maj": []}
    }
    
    for fpath in tqdm(files):
        res = calculate_metrics_for_file(fpath)
        
        history["Common"]["pass1"].append(res["pass1"])
        history["Common"]["maj"].append(res["maj"])
        
        for name in ["Old Delta (v1)", "ORM (Baseline)", "New Delta (v2)"]:
            if name in res:
                for metric in history[name].keys():
                    history[name][metric].append(res[name][metric])

    # --- 最終表示 ---
    print("\n" + "="*100)
    print(f"{'METRIC':<15} | {'ORM (Baseline)':<22} | {'Old Delta (v1)':<22} | {'New Delta (v3)':<22}")
    print("="*100)
    
    # Common
    p1_m = np.mean(history["Common"]["pass1"])
    maj_m = np.mean(history["Common"]["maj"])
    print(f"{'Pass@1':<15} | {p1_m:.2%} {'(Avg)':<53}")
    print(f"{'Majority Vote':<15} | {maj_m:.2%} {'(Base)':<53}")
    print("-" * 100)
    
    # 各モデル比較
    metrics_order = ["min", "mean", "last", "sum", "max", "weighted"]
    
    for met in metrics_order:
        row_str = f"BoN ({met.title()})".ljust(15) + " | "
        
        # ORM
        vals = history["ORM (Baseline)"][met]
        row_str += f"{np.mean(vals):.2%} ± {np.std(vals):.2%}   | "
        
        # Old
        vals = history["Old Delta (v1)"][met]
        row_str += f"{np.mean(vals):.2%} ± {np.std(vals):.2%}   | "
        
        # New
        vals = history["New Delta (v2)"][met]
        row_str += f"**{np.mean(vals):.2%}** ± {np.std(vals):.2%}"
        
        print(row_str)
        
    print("="*100)

if __name__ == "__main__":
    main()