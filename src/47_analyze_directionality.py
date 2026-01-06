import os
import json
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# 特徴量名 (順序を合わせる)
FEATURE_NAMES = [
    "Min (Logic Break)", "Mean (Coherence)", "Max (Peak)", 
    "Std (Instability)", "Last (Conclusion)", "First (Start)", 
    "Min Last 3 (Late Break)", "Length (Complexity)", "Sum Logits (Joint Prob)"
]

def get_correlations(train_path):
    files = glob.glob(os.path.join(train_path, "**", "seed_*.jsonl"), recursive=True)
    all_data = []

    for file_path in tqdm(files, desc="Data Loading"):
        # ノイズ（no_trigger, orm）をパスで除外
        if "no_trigger" in file_path.lower() or "orm" in file_path.lower():
            continue
            
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                for r in item.get("responses", []):
                    scores = [float(s) for s in r.get("step_scores", []) if s is not None]
                    if not scores: continue
                    
                    # 特徴量抽出 (以前の extract_features と同じ論理)
                    logits = np.array(scores)
                    feat = [
                        np.min(logits), np.mean(logits), np.max(logits),
                        np.std(logits), logits[-1], logits[0],
                        np.min(logits[-3:]) if len(logits)>=3 else np.min(logits),
                        float(len(logits)), np.sum(logits)
                    ]
                    label = 1 if r.get("is_correct", False) else 0
                    all_data.append(feat + [label])

    df = pd.DataFrame(all_data, columns=FEATURE_NAMES + ["is_correct"])
    
    # is_correct との相関を計算
    correlations = df.corr()["is_correct"].drop("is_correct").sort_values(ascending=False)
    
    print("\n=== FEATURE CORRELATION WITH CORRECTNESS (Direction) ===")
    print(correlations)
    
    # 可視化
    plt.figure(figsize=(10, 6))
    colors = ["#4C72B0" if x > 0 else "#C44E52" for x in correlations]
    sns.barplot(x=correlations.values, y=correlations.index, palette=colors)
    plt.axvline(0, color='black', linewidth=1)
    plt.title("Correlation: Feature Value vs. Correctness")
    plt.xlabel("Pearson Correlation Coefficient")
    plt.tight_layout()
    plt.savefig("prm_feature_correlations.png")
    return correlations

# 実行（パスは適宜調整してください）
get_correlations("results/numina_train/Qwen2.5-Math-1.5B-Instruct")