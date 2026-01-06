import json
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

INPUT_FILE = "data/annotated_train_data_30k_v2.0.jsonl"
OUTPUT_IMG = "data/final_v2_distribution_linear.png" # ファイル名変更

def main():
    if not os.path.exists(INPUT_FILE):
        print("Waiting for file...")
        return

    labels_correct = []
    labels_incorrect = []
    
    print(f"Reading {INPUT_FILE}...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            try:
                rec = json.loads(line)
                lbl = rec["label"]
                if rec["is_outcome_correct"]:
                    labels_correct.append(lbl)
                else:
                    labels_incorrect.append(lbl)
            except: pass

    # 統計
    cor = np.array(labels_correct)
    inc = np.array(labels_incorrect)
    
    print("\n--- Statistics ---")
    print(f"Correct (N={len(cor)}): Mean={cor.mean():.4f}, Max={cor.max():.4f}, Min={cor.min():.4f}")
    print(f"Incorrect (N={len(inc)}): Mean={inc.mean():.4f}, Max={inc.max():.4f}, Min={inc.min():.4f}")
    
    # プロット
    plt.figure(figsize=(12, 6))
    
    # ビン設定: -3.0 から 3.0 まで
    bins = np.linspace(-3.0, 3.0, 100)
    
    # ★変更点: log=False にする (デフォルトなので指定しなくてもOK)
    # density=True (正規化) は、データ数が違う緑と赤を比較しやすくするために残します
    plt.hist(inc, bins=bins, alpha=0.5, label='Incorrect', color='red', density=True, log=False)
    plt.hist(cor, bins=bins, alpha=0.5, label='Correct', color='green', density=True, log=False)
    
    plt.title("Distribution of Final Labels (Linear Scale)")
    plt.xlabel("Reward Score")
    plt.ylabel("Density")
    plt.axvline(0, color='black', linestyle='--')
    plt.legend()
    
    # グリッド線を見やすくする
    plt.grid(axis='y', alpha=0.5)
    
    plt.savefig(OUTPUT_IMG)
    print(f"Saved plot to {OUTPUT_IMG}")

if __name__ == "__main__":
    main()