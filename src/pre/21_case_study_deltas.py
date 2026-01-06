import json
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import random

# ==========================================
# 設定
# ==========================================
INPUT_FILE = "data/annotated_train_data_30k.jsonl"
OUTPUT_DIR = "data/case_studies"
NUM_CASES = 10  # 何問チェックするか

# 逆算用パラメータ
GAMMA = 0.9
OUTCOME_REWARD = 1.0 

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading data from {INPUT_FILE}...")
    
    # データを問題IDごとにまとめる
    data_by_id = defaultdict(list)
    
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                data_by_id[rec["source_id"]].append(rec)
            except:
                continue

    # 正解パスと不正解パスの両方を持つ問題だけを抽出
    valid_ids = []
    for sid, items in data_by_id.items():
        has_correct = any(x["is_outcome_correct"] for x in items)
        has_incorrect = any(not x["is_outcome_correct"] for x in items)
        if has_correct and has_incorrect:
            valid_ids.append(sid)
            
    print(f"Found {len(valid_ids)} problems with both correct and incorrect paths.")
    
    # ランダムに選択
    target_ids = random.sample(valid_ids, min(NUM_CASES, len(valid_ids)))
    
    for i, sid in enumerate(target_ids):
        analyze_case(i, sid, data_by_id[sid])

def reconstruct_path_deltas(items, is_correct_path):
    """パス内のアイテムリストから生Deltaを復元する"""
    # テキスト長でソートして順序を整える
    items.sort(key=lambda x: len(x["full_text"]))
    
    deltas = []
    steps = []
    
    for i in range(len(items)):
        current_val = items[i]["raw_score"]
        step_text = items[i].get("step_text", "")
        steps.append(step_text)
        
        if i < len(items) - 1:
            next_val = items[i+1]["raw_score"]
            delta = current_val - GAMMA * next_val
        else:
            # 最終ステップ
            future = OUTCOME_REWARD if is_correct_path else 0.0
            delta = current_val - GAMMA * future
            
        deltas.append(delta)
        
    return deltas, steps

def analyze_case(idx, sid, items):
    """1つの問題について可視化"""
    print(f"\n=== Case {idx+1} (ID: {sid}) ===")
    
    # パスごとに分離 (簡易ロジック: テキスト包含関係)
    # 正解パス1つ、不正解パス1つを抽出
    correct_path_items = []
    incorrect_path_items = []
    
    # テキスト長でソート
    items.sort(key=lambda x: len(x["full_text"]))
    
    # 正解/不正解を分ける
    correct_candidates = [x for x in items if x["is_outcome_correct"]]
    incorrect_candidates = [x for x in items if not x["is_outcome_correct"]]
    
    if not correct_candidates or not incorrect_candidates: return

    # 1つのパスを復元する関数
    def extract_single_path(candidates):
        # 最も長いテキストを持つものを終点とし、その親を探していく
        leaf = candidates[-1] # 一番長いもの
        path = [leaf]
        current_text = leaf["full_text"]
        
        # 逆順に親を探す (簡易的)
        for cand in reversed(candidates[:-1]):
            if cand["full_text"] in current_text:
                path.insert(0, cand)
                current_text = cand["full_text"]
        return path

    path_c = extract_single_path(correct_candidates)
    path_i = extract_single_path(incorrect_candidates)
    
    # Delta計算
    deltas_c, steps_c = reconstruct_path_deltas(path_c, True)
    deltas_i, steps_i = reconstruct_path_deltas(path_i, False)
    
    # --- グラフ描画 ---
    plt.figure(figsize=(12, 6))
    
    # Correct Path
    plt.plot(range(len(deltas_c)), deltas_c, marker='o', label='Correct Path', color='green', linewidth=2)
    
    # Incorrect Path
    plt.plot(range(len(deltas_i)), deltas_i, marker='x', label='Incorrect Path', color='red', linewidth=2, linestyle='--')
    
    plt.axhline(0, color='black', linewidth=0.5)
    plt.title(f"Raw Delta Trajectory (Case {idx+1})")
    plt.xlabel("Step Number")
    plt.ylabel("Raw Delta (Probability Gain)")
    plt.legend()
    plt.grid(True)
    
    filename = os.path.join(OUTPUT_DIR, f"case_{idx+1}_id_{sid}.png")
    plt.savefig(filename)
    print(f"Graph saved to {filename}")
    
    # --- テキスト表示 (不正解の原因分析) ---
    # 不正解パスの中で、Deltaが最も低かったステップを表示
    min_delta_idx = np.argmin(deltas_i)
    print(f"[Incorrect Path Analysis]")
    print(f"Lowest Delta Step ({min_delta_idx}): {deltas_i[min_delta_idx]:.4f}")
    print(f"Text: \"{steps_i[min_delta_idx]}\"")
    
    # その前後の文脈
    if min_delta_idx > 0:
        print(f"Prev : \"{steps_i[min_delta_idx-1]}\"")

if __name__ == "__main__":
    main()