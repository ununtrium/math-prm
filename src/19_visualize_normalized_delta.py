import json
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from scipy.interpolate import interp1d

# ==========================================
# 設定
# ==========================================
INPUT_FILE = "data/annotated_train_data_30k.jsonl"
OUTPUT_IMG = "data/normalized_delta_trajectory.png"

# 逆算用パラメータ
GAMMA = 0.9
OUTCOME_REWARD = 1.0 

# 正規化の解像度 (0% ~ 100% を何分割するか)
NORMALIZED_LEN = 100 

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    print(f"Loading and analyzing trajectories from {INPUT_FILE}...")

    # データ読み込み & パス復元
    # (メモリ節約のため、必要なデータだけ保持)
    raw_data_buffer = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                raw_data_buffer.append(json.loads(line))
            except:
                pass

    # パスごとにグループ化 (source_id と full_text の包含関係を利用)
    # 簡易的に source_id でソートしてから処理
    # 注意: numinamath_gen_2k.jsonl はID順とは限らないので、辞書でまとめる
    grouped_by_id = {}
    for item in tqdm(raw_data_buffer, desc="Grouping"):
        sid = item.get("source_id", "unknown")
        if sid not in grouped_by_id: grouped_by_id[sid] = []
        grouped_by_id[sid].append(item)

    # 補間されたDelta配列を格納するリスト
    correct_trajectories = []
    incorrect_trajectories = []

    print("Reconstructing and Normalizing Deltas...")
    
    for sid, items in tqdm(grouped_by_id.items()):
        # テキスト長でソート (Step 1 -> Step 2 ...)
        items.sort(key=lambda x: len(x["full_text"]))
        
        # パスを分離する
        # 同じ問題でも8つのパスがあるため、包含関係でツリーを復元
        paths = []
        for item in items:
            text = item["full_text"]
            found_parent = False
            for path in paths:
                last_step = path[-1]
                # 前のステップが今のステップの前半部分と一致するか
                if text.startswith(last_step["full_text"]) and len(text) > len(last_step["full_text"]):
                    path.append(item)
                    found_parent = True
                    break
            if not found_parent:
                paths.append([item])

        # 各パスについてDelta計算 & 正規化
        for path in paths:
            # パスが短すぎる場合はスキップ (補間できない)
            if len(path) < 1: continue
            
            is_correct = path[0]["is_outcome_correct"]
            deltas = []
            
            # 1. 生のDeltaを逆算
            for i in range(len(path)):
                current_val = path[i]["raw_score"]
                
                if i < len(path) - 1:
                    next_val = path[i+1]["raw_score"]
                    delta = current_val - GAMMA * next_val
                else:
                    # 最終ステップ
                    future = OUTCOME_REWARD if is_correct else 0.0
                    delta = current_val - GAMMA * future
                
                deltas.append(delta)
            
            # 2. 長さを 100 に正規化 (Interpolation)
            if len(deltas) > 1:
                x_old = np.linspace(0, 1, len(deltas))
                f = interp1d(x_old, deltas, kind='linear') # 線形補間
                x_new = np.linspace(0, 1, NORMALIZED_LEN)
                deltas_norm = f(x_new)
            else:
                # 1ステップしかない場合は、その値を100個並べる
                deltas_norm = np.full(NORMALIZED_LEN, deltas[0])
            
            if is_correct:
                correct_trajectories.append(deltas_norm)
            else:
                incorrect_trajectories.append(deltas_norm)

    # ==========================================
    # 集計とプロット
    # ==========================================
    cor_matrix = np.array(correct_trajectories) # Shape: (N, 100)
    inc_matrix = np.array(incorrect_trajectories)
    
    print(f"\nStats:")
    print(f"Correct Paths: {cor_matrix.shape[0]}")
    print(f"Incorrect Paths: {inc_matrix.shape[0]}")
    
    # 平均と標準誤差(または標準偏差)を計算
    x_axis = np.linspace(0, 100, NORMALIZED_LEN)
    
    cor_mean = np.mean(cor_matrix, axis=0)
    cor_std = np.std(cor_matrix, axis=0)
    # 信頼区間 (Mean +/- 0.5 * Std で見やすくする)
    cor_upper = cor_mean + 0.5 * cor_std
    cor_lower = cor_mean - 0.5 * cor_std

    inc_mean = np.mean(inc_matrix, axis=0)
    inc_std = np.std(inc_matrix, axis=0)
    inc_upper = inc_mean + 0.5 * inc_std
    inc_lower = inc_mean - 0.5 * inc_std

    # プロット
    plt.figure(figsize=(12, 6))
    
    # Correct (Green)
    plt.plot(x_axis, cor_mean, color='green', label='Correct Paths (Mean Delta)', linewidth=2)
    plt.fill_between(x_axis, cor_lower, cor_upper, color='green', alpha=0.1)
    
    # Incorrect (Red)
    plt.plot(x_axis, inc_mean, color='red', label='Incorrect Paths (Mean Delta)', linewidth=2)
    plt.fill_between(x_axis, inc_lower, inc_upper, color='red', alpha=0.1)

    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    
    plt.title('Normalized Trajectory of Pure Deltas (0% = Start, 100% = End)')
    plt.xlabel('Progress (%)')
    plt.ylabel('Average Raw Delta')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(OUTPUT_IMG)
    print(f"\nGraph saved to {OUTPUT_IMG}")

if __name__ == "__main__":
    main()