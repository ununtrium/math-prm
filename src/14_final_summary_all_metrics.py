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
# 保存されたトライアルデータの場所
INPUT_DIR = "data/experiments/final_comparison_1.5b_30k_v1.0"

# Weighted Voteの固定パラメータ (最適化結果に基づく)
WV_SCALE = 1.0
WV_METRIC = "mean" # mean or min

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
# 計算ロジック
# ==========================================
def calculate_trial_metrics(data):
    """1つのトライアルファイルから全指標を計算"""
    
    # カウンター
    metrics = {
        "pass1": 0, "maj": 0,
        "delta": {"min": 0, "mean": 0, "last": 0, "weighted": 0},
        "orm":   {"min": 0, "mean": 0, "last": 0, "weighted": 0}
    }
    
    total_problems = len(data)
    total_paths = 0
    
    for item in data:
        gold = item["gold"]
        paths = item["paths"]
        extracted = [extract_answer_content(p) for p in paths]
        valid_idx = [i for i, a in enumerate(extracted) if a]
        
        # --- 1. Pass@1 ---
        for i, ans in enumerate(extracted):
            if check_correctness(ans, gold): metrics["pass1"] += 1
        total_paths += len(paths)
        
        if not valid_idx: continue

        # --- 2. Majority Vote ---
        valid_ans = [extracted[i] for i in valid_idx]
        maj = Counter(valid_ans).most_common(1)[0][0]
        if check_correctness(maj, gold): metrics["maj"] += 1
        
        # --- 内部関数: 各モデルの指標計算 ---
        def calc_model_metrics(step_scores_key, output_key):
            step_scores_list = item[step_scores_key]
            
            # 各集計スコア
            s_min = [min(s) if s else -99 for s in step_scores_list]
            s_mean = [np.mean(s) if s else -99 for s in step_scores_list]
            s_last = [s[-1] if s else -99 for s in step_scores_list]
            
            # BoN
            if check_correctness(extracted[np.argmax(s_min)], gold): metrics[output_key]["min"] += 1
            if check_correctness(extracted[np.argmax(s_mean)], gold): metrics[output_key]["mean"] += 1
            if check_correctness(extracted[np.argmax(s_last)], gold): metrics[output_key]["last"] += 1
            
            # Weighted Vote
            votes = {}
            for i in valid_idx:
                ans = extracted[i]
                # 設定されたメトリクスを使用
                val = s_mean[i] if WV_METRIC == "mean" else s_min[i]
                weight = math.exp(val * WV_SCALE)
                votes[ans] = votes.get(ans, 0) + weight
            
            if check_correctness(max(votes, key=votes.get), gold): 
                metrics[output_key]["weighted"] += 1

        # Delta-PRMの計算
        calc_model_metrics("step_scores_delta", "delta")
        # ORMの計算
        calc_model_metrics("step_scores_orm", "orm")

    # 精度に変換
    return {
        "pass1": metrics["pass1"] / total_paths,
        "maj": metrics["maj"] / total_problems,
        "delta_min": metrics["delta"]["min"] / total_problems,
        "delta_mean": metrics["delta"]["mean"] / total_problems,
        "delta_last": metrics["delta"]["last"] / total_problems,
        "delta_weighted": metrics["delta"]["weighted"] / total_problems,
        "orm_min": metrics["orm"]["min"] / total_problems,
        "orm_mean": metrics["orm"]["mean"] / total_problems,
        "orm_last": metrics["orm"]["last"] / total_problems,
        "orm_weighted": metrics["orm"]["weighted"] / total_problems,
    }

# ==========================================
# メイン処理
# ==========================================
def main():
    files = glob.glob(os.path.join(INPUT_DIR, "trial_*.json"))
    if not files:
        print("No files found.")
        return
        
    print(f"Aggregating results from {len(files)} trials...")
    print(f"Weighted Vote Settings: Metric={WV_METRIC.upper()}, Scale={WV_SCALE}")
    
    # 全トライアルの結果をリストに格納
    history = {k: [] for k in ["pass1", "maj", 
                               "delta_min", "delta_mean", "delta_last", "delta_weighted",
                               "orm_min", "orm_mean", "orm_last", "orm_weighted"]}
    
    for fpath in tqdm(files):
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        res = calculate_trial_metrics(data)
        for k, v in res.items():
            history[k].append(v)

    # 集計と表示
    print("\n" + "="*80)
    print(f"{'METRIC':<20} | {'DELTA-PRM (Ours)':<22} | {'ORM (Baseline)':<22} | {'GAP':<8}")
    print("="*80)
    
    def fmt(key_delta, key_orm=None):
        # Delta
        d_mean = np.mean(history[key_delta])
        d_std = np.std(history[key_delta])
        d_str = f"{d_mean:.2%} ± {d_std:.2%}"
        
        if key_orm:
            # ORM
            o_mean = np.mean(history[key_orm])
            o_std = np.std(history[key_orm])
            o_str = f"{o_mean:.2%} ± {o_std:.2%}"
            
            # Gap
            gap = d_mean - o_mean
            gap_str = f"{gap:+.2%}"
            return d_str, o_str, gap_str
        else:
            return d_str, "-", "-"

    # Pass@1 & Maj
    p1, _, _ = fmt("pass1")
    print(f"{'Pass@1':<20} | {p1:<22} | {'-':<22} | -")
    
    maj, _, _ = fmt("maj")
    print(f"{'Majority Vote':<20} | {maj:<22} | {'-':<22} | -")
    print("-" * 80)
    
    # Best-of-N Comparison
    d, o, g = fmt("delta_min", "orm_min")
    print(f"{'BoN (Min)':<20} | {d:<22} | {o:<22} | {g:<8} <-- Process")
    
    d, o, g = fmt("delta_mean", "orm_mean")
    print(f"{'BoN (Mean)':<20} | {d:<22} | {o:<22} | {g:<8}")
    
    d, o, g = fmt("delta_last", "orm_last")
    print(f"{'BoN (Last)':<20} | {d:<22} | {o:<22} | {g:<8} <-- Outcome")
    
    print("-" * 80)
    
    # Weighted Vote
    d, o, g = fmt("delta_weighted", "orm_weighted")
    print(f"{'Weighted Vote':<20} | {d:<22} | {o:<22} | {g:<8} <-- Final")
    print("="*80)

if __name__ == "__main__":
    main()