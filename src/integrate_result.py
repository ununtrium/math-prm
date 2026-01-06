import os
import json
import glob
import pandas as pd
import numpy as np
import joblib
from collections import Counter

# --- 設定（環境に合わせて変更してください） ---
RESULTS_DIR = "results"
MODELS_DIR = "models/learned_aggregators"
OUTPUT_EXCEL = "integrated_results_no_trigger.xlsx"

# ==========================================
# 1. ヒューリスティック手法の解析
# ==========================================
def get_heuristic_results():
    AGG_METHODS = ["min", "mean", "last", "sum"]
    files = glob.glob(os.path.join(RESULTS_DIR, "*", "*", "*", "*", "seed_*.jsonl"))
    
    records = []
    for file_path in files:
        parts = file_path.split(os.sep)
        if len(parts) < 5: continue
        bench, gen, prm, s_dir = parts[-5], parts[-4], parts[-3], parts[-2]
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                items = [json.loads(line) for line in f]
            
            total = len(items)
            if total == 0: continue

            # このシードにおける各手法の正解数をカウント
            correct_counts = {m: 0 for m in AGG_METHODS + ["majority_vote"]}
            for item in items:
                resps = item.get("responses", [])
                
                # Majority Vote
                answers = [r["extracted"] for r in resps if r.get("extracted")]
                if answers:
                    vote = Counter(answers).most_common(1)[0][0]
                    if any(r.get("extracted") == vote and r.get("is_correct") for r in resps):
                        correct_counts["majority_vote"] += 1
                
                # 各ヒューリスティック手法
                for m in AGG_METHODS:
                    best_is_correct = False
                    best_score = -float('inf')
                    for r in resps:
                        scores = [s for s in r.get("step_scores", []) if s is not None]
                        if not scores: continue
                        
                        if m == "min": s_val = min(scores)
                        elif m == "mean": s_val = np.mean(scores)
                        elif m == "last": s_val = scores[-1]
                        elif m == "sum": s_val = sum(scores)
                        
                        if s_val > best_score:
                            best_score = s_val
                            best_is_correct = r.get("is_correct", False)
                    if best_is_correct: correct_counts[m] += 1
            
            # 各手法のスコア（%）を辞書に格納
            metrics = {k: (v/total)*100 for k, v in correct_counts.items()}
            
            # 1レコードに全手法の結果を入れる
            row = {
                "Benchmark": bench, "Generator": gen, "Samples": s_dir, "PRM": prm,
                "MajVote": metrics["majority_vote"]
            }
            for m in AGG_METHODS:
                row[m] = metrics[m] # min, mean, last, sum のスコアを個別に保持
                
            records.append(row)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    df = pd.DataFrame(records)
    if df.empty: return df

    # 1. 設定ごとにグループ化して、全手法のスコアを平均する
    df_grouped = df.groupby(["Benchmark", "Generator", "Samples", "PRM"]).mean(numeric_only=True).reset_index()
    
    # 2. 平均したスコアの中でどれが最大か判定する
    # AGG_METHODSの中で最大の列名とその値を取得
    df_grouped["Best_Heur_Score"] = df_grouped[AGG_METHODS].max(axis=1)
    df_grouped["Best_Heur_Method"] = df_grouped[AGG_METHODS].idxmax(axis=1).apply(lambda x: x.upper())
    
    # 3. 不要になった個別手法の列を消して MajVote と Best 関連だけに絞る（必要に応じて）
    final_cols = ["Benchmark", "Generator", "Samples", "PRM", "MajVote", "Best_Heur_Method", "Best_Heur_Score"]
    return df_grouped[final_cols]
# ==========================================
# 2. 学習ベース手法の解析
# ==========================================
def extract_features(step_scores):
    scores = np.array([float(s) for s in step_scores if s is not None])
    if len(scores) == 0: return [0.0]*9
    feats = [np.min(scores), np.mean(scores), np.max(scores), np.std(scores), scores[-1], scores[0], 
             np.min(scores[-3:]) if len(scores)>=3 else np.min(scores), float(len(scores)), np.sum(scores)]
    return np.nan_to_num(feats).tolist()

def get_learned_results():
    model_files = glob.glob(os.path.join(MODELS_DIR, "*.pkl"))
    results = []
    
    for m_file in model_files:
        model = joblib.load(m_file)
        train_gen, prm_name = os.path.basename(m_file).replace(".pkl", "").split("__", 1)
        
        # テストデータの探索
        test_dirs = glob.glob(os.path.join(RESULTS_DIR, "*", "*", prm_name, "samples_*"))
        for t_dir in test_dirs:
            parts = t_dir.split(os.sep)
            bench, test_gen, s_dir = parts[-4], parts[-3], parts[-1]
            
            correct_learned, total = 0, 0
            for f_path in glob.glob(os.path.join(t_dir, "seed_*.jsonl")):
                with open(f_path, "r") as f:
                    for line in f:
                        item = json.loads(line)
                        resps = item.get("responses", [])
                        if not resps: continue
                        
                        feats = [extract_features(r.get("step_scores", [])) for r in resps if r.get("step_scores")]
                        labels = [1 if r.get("is_correct") else 0 for r in resps if r.get("step_scores")]
                        
                        if feats:
                            probs = model.predict_proba(feats)[:, 1]
                            if labels[np.argmax(probs)] == 1: correct_learned += 1
                            total += 1
            
            if total > 0:
                results.append({
                    "Benchmark": bench, "Generator": test_gen, "Samples": s_dir, "PRM": prm_name,
                    "Learned_Acc": (correct_learned/total)*100,
                    "Aggregator_Train_Gen": train_gen
                })
    return pd.DataFrame(results)

# ==========================================
# 3. 統合と保存
# ==========================================
if __name__ == "__main__":
    print("Processing Heuristics...")
    df_h = get_heuristic_results()
    
    print("Processing Learned...")
    df_l = get_learned_results()
    
    # 結合 (Benchmark, Generator, Samples, PRMをキーにする)
    df_final = pd.merge(df_h, df_l, on=["Benchmark", "Generator", "Samples", "PRM"], how="outer")
    
    # 見栄えのためのソート
    df_final = df_final.sort_values(["Benchmark", "Generator", "Samples", "PRM"])
    
    # エクセル保存
    df_final.to_excel(OUTPUT_EXCEL, index=False)
    print(f"Done! Saved to {OUTPUT_EXCEL}")