import json
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# ==========================================
# 設定
# ==========================================
INPUT_FILE = "data/p_scaled_value_train_30k.jsonl"
OUTPUT_IMAGE = "data/label_distribution_v3.0.png"

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    print(f"Loading data from {INPUT_FILE}...")

    correct_labels = []
    incorrect_labels = []
    
    # 統計用
    total_count = 0
    
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        # 行数カウント（tqdm用）
        lines = f.readlines()
        
        for line in tqdm(lines, desc="Processing"):
            try:
                record = json.loads(line)
                label = record["label"]
                is_correct = record["is_outcome_correct"]
                
                if is_correct:
                    correct_labels.append(label)
                else:
                    incorrect_labels.append(label)
                
                total_count += 1
            except:
                continue

    # ==========================================
    # 統計情報の表示
    # ==========================================
    print("\n" + "="*30)
    print("Label Distribution Stats")
    print("="*30)
    
    print(f"Total Samples: {total_count}")
    
    # 正解データの統計
    if correct_labels:
        c_mean = np.mean(correct_labels)
        c_std = np.std(correct_labels)
        print(f"\n[Correct Outcome Paths]")
        print(f"  Count: {len(correct_labels)} ({len(correct_labels)/total_count:.1%})")
        print(f"  Mean Label: {c_mean:.4f}")
        print(f"  Std Dev:    {c_std:.4f}")
    else:
        print("\n[Correct Outcome Paths]: None")

    # 不正解データの統計
    if incorrect_labels:
        i_mean = np.mean(incorrect_labels)
        i_std = np.std(incorrect_labels)
        print(f"\n[Incorrect Outcome Paths]")
        print(f"  Count: {len(incorrect_labels)} ({len(incorrect_labels)/total_count:.1%})")
        print(f"  Mean Label: {i_mean:.4f}")
        print(f"  Std Dev:    {i_std:.4f}")
    else:
        print("\n[Incorrect Outcome Paths]: None")

    # ==========================================
    # ヒストグラムの描画
    # ==========================================
    plt.figure(figsize=(10, 6))
    
    # ビンの設定 (-2.0 から 2.0 まで 50分割)
    bins = np.linspace(-2.1, 2.1, 50)
    
    # 不正解データのプロット (赤)
    plt.hist(incorrect_labels, bins=bins, alpha=0.5, label='Incorrect Paths', 
             color='red', edgecolor='black', density=True)
    
    # 正解データのプロット (緑)
    plt.hist(correct_labels, bins=bins, alpha=0.5, label='Correct Paths', 
             color='green', edgecolor='black', density=True)

    # 装飾
    plt.title('Distribution of PRM Labels (Correct vs Incorrect)', fontsize=15)
    plt.xlabel('Label Score (Reward)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # 保存
    plt.savefig(OUTPUT_IMAGE)
    print(f"\nGraph saved to {OUTPUT_IMAGE}")
    # plt.show() # GUI環境ならコメントアウトを外す

if __name__ == "__main__":
    main()