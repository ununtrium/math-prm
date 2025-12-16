import json
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# ==========================================
# 設定
# ==========================================
INPUT_FILE = "data/annotated_train_data_30k.jsonl"
OUTPUT_IMG = "data/pure_delta_distribution.png"

# 学習時のパラメータ（逆算に必須）
GAMMA = 0.9
OUTCOME_REWARD = 1.0 

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    print(f"Calculating Pure Deltas from {INPUT_FILE}...")

    correct_deltas = []
    incorrect_deltas = []
    
    # データを全て読み込む
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # パスごとに分解するロジック
    # JSONLはパスごとに連続して記録されている前提
    
    current_path_rows = []
    
    for i, line in enumerate(tqdm(lines)):
        try:
            row = json.loads(line)
        except:
            continue

        # 新しいパスの開始判定:
        # 1. リストが空なら開始
        # 2. source_id が変わったら新しい問題
        # 3. テキストの長さが短くなったら新しいパス (前のパスが終わった)
        is_new_path =False
        if not current_path_rows:
            is_new_path = True
        else:
            last_row = current_path_rows[-1]
            if row["source_id"] != last_row["source_id"]:
                is_new_path = True
            elif len(row["full_text"]) < len(last_row["full_text"]):
                is_new_path = True
        
        if is_new_path:
            # 直前のパスのDeltaを計算して保存
            if current_path_rows:
                process_path(current_path_rows, correct_deltas, incorrect_deltas)
            # リセット
            current_path_rows = [row]
        else:
            # 同じパスの続き
            current_path_rows.append(row)

    # 最後のパスを処理
    if current_path_rows:
        process_path(current_path_rows, correct_deltas, incorrect_deltas)

    # ==========================================
    # 可視化
    # ==========================================
    plot_deltas(correct_deltas, incorrect_deltas)

def process_path(rows, correct_list, incorrect_list):
    """
    1つのパスに含まれる行(ステップ)リストからDeltaを逆算する
    """
    is_correct_path = rows[0]["is_outcome_correct"]
    
    for i in range(len(rows)):
        current_val = rows[i]["raw_score"] # クリップ前の値を使う
        
        # 次のステップの価値 (未来)
        if i < len(rows) - 1:
            next_val = rows[i+1]["raw_score"]
            future_value = next_val
        else:
            # 最後のステップの場合、未来は Outcome Reward
            if is_correct_path:
                future_value = OUTCOME_REWARD
            else:
                future_value = 0.0 # 不正解の未来はゼロ
        
        # ★逆算: Delta = V_t - gamma * V_{t+1}
        delta = current_val - (GAMMA * future_value)
        
        if is_correct_path:
            correct_list.append(delta)
        else:
            incorrect_list.append(delta)

def plot_deltas(correct, incorrect):
    cor = np.array(correct)
    inc = np.array(incorrect)
    
    print("\n" + "="*40)
    print("PURE DELTA STATISTICS (Outcome Removed)")
    print("="*40)
    print(f"Correct Steps (N={len(cor)}):")
    print(f"  Mean: {cor.mean():.4f}")
    print(f"  Max:  {cor.max():.4f}")
    print(f"  Min:  {cor.min():.4f}")
    
    print(f"Incorrect Steps (N={len(inc)}):")
    print(f"  Mean: {inc.mean():.4f}")
    print(f"  Max:  {inc.max():.4f}")
    print(f"  Min:  {inc.min():.4f}")
    
    # グラフ描画
    plt.figure(figsize=(12, 6))
    
    # Deltaは -1.0 ~ 1.0 の範囲に収まるはず
    # まれに計算誤差やスケールで飛び出すことがあるので範囲制限
    bins = np.linspace(-1.5, 1.5, 100)
    
    plt.hist(inc, bins=bins, alpha=0.5, label='Incorrect Path Deltas', 
             color='red', density=True, histtype='stepfilled')
    plt.hist(cor, bins=bins, alpha=0.5, label='Correct Path Deltas', 
             color='green', density=True, histtype='stepfilled')

    plt.title(f'Distribution of PURE Deltas (Outcome Reward Removed)')
    plt.xlabel('Pure Delta (Probability Increment)')
    plt.ylabel('Density')
    plt.axvline(0, color='black', linestyle='--', linewidth=1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(OUTPUT_IMG)
    print(f"\nGraph saved to {OUTPUT_IMG}")

if __name__ == "__main__":
    main()