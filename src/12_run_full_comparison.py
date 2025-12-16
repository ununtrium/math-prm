import numpy as np
import os
import json
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from evaluate_math500 import Evaluator

# ==========================================
# 設定
# ==========================================
MODEL_A_PATH = "models/delta_prm_1.5b_30k_v1.0/checkpoint-13000"
MODEL_A_NAME = "Delta-PRM"

MODEL_B_PATH = "models/orm_1.5b_30k_v1.0/checkpoint-13000"
MODEL_B_NAME = "ORM"

NUM_TRIALS = 3
SEEDS = [42, 100, 999]

# ★変更点: 保存先を整理されたフォルダに変更
OUTPUT_DIR = "data/experiments/final_comparison_1.5b_30k_v1.0_checkpoint-13000"

def main():
    # フォルダがなければ作る（再帰的に作成）
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Starting Comparison: {MODEL_A_NAME} vs {MODEL_B_NAME}")
    print(f"Output Directory: {OUTPUT_DIR}")
    
    # 結果格納用
    history = {
        "pass1": [], "maj": [],
        "delta": {"weighted": [], "min": [], "mean": [], "last": []},
        "orm":   {"weighted": [], "min": [], "mean": [], "last": []}
    }
    
    evaluator = Evaluator()
    
    for i, seed in enumerate(SEEDS):
        print(f"\n" + "="*50)
        print(f"  TRIAL {i+1}/{NUM_TRIALS} (Seed={seed})")
        print("="*50)
        
        # 1. 生成
        print(">>> Generating Paths...")
        gen_results = evaluator.run_generation(seed=seed)
        
        # 2. Delta-PRM 採点
        print(f">>> Scoring with {MODEL_A_NAME}...")
        res_a = evaluator.run_scoring(gen_results, MODEL_A_PATH)
        met_a = evaluator.calculate_metrics(res_a, scale=0.5)
        
        # 3. ORM 採点
        print(f">>> Scoring with {MODEL_B_NAME}...")
        res_b = evaluator.run_scoring(gen_results, MODEL_B_PATH)
        met_b = evaluator.calculate_metrics(res_b, scale=0.5)
        
        # --- データ保存処理 ---
        save_data = []
        for idx, item in enumerate(res_a):
            # ベースはDelta-PRMの結果 (step_texts等も含む)
            combined_item = item.copy()
            
            # わかりやすくリネームして保存
            combined_item["scores_delta"] = item["scores"]
            combined_item["step_scores_delta"] = item["step_scores"]
            del combined_item["scores"]      # 重複削除
            del combined_item["step_scores"] # 重複削除
            
            # ORMのスコアを結合
            combined_item["scores_orm"] = res_b[idx]["scores"]
            combined_item["step_scores_orm"] = res_b[idx]["step_scores"]
            
            # 注: step_texts は共通なので res_a のものをそのまま使う
            
            save_data.append(combined_item)
            
        filename = os.path.join(OUTPUT_DIR, f"trial_{i+1}_seed_{seed}.json")
        print(f">>> Saving detailed data to {filename}...")
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        # --------------------

        # 履歴記録
        history["pass1"].append(met_a["pass1"])
        history["maj"].append(met_a["maj_vote"])
        
        history["delta"]["weighted"].append(met_a["weighted_vote"])
        history["delta"]["min"].append(met_a["bon_min"])
        history["delta"]["last"].append(met_a["bon_last"])
        
        history["orm"]["weighted"].append(met_b["weighted_vote"])
        history["orm"]["min"].append(met_b["bon_min"])
        history["orm"]["last"].append(met_b["bon_last"])
        
        print(f"  [Result Trial {i+1}] Pass@1: {met_a['pass1']:.2%} | Maj: {met_a['maj_vote']:.2%}")
        print(f"  Delta(W): {met_a['weighted_vote']:.2%} | Min: {met_a['bon_min']:.2%}")
        print(f"  ORM(W)  : {met_b['weighted_vote']:.2%} | Last: {met_b['bon_last']:.2%}")

    # 最終集計表示
    def print_stat(label, values):
        mean = np.mean(values)
        std = np.std(values)
        print(f"{label:<25}: {mean:.2%} ± {std:.2%}")

    print("\n" + "="*50)
    print("  FINAL COMPARISON REPORT (Mean of 3 Trials)")
    print("="*50)
    print_stat("Pass@1 (Avg)", history["pass1"])
    print_stat("Majority Vote", history["maj"])
    print("-" * 30)
    print_stat(f"{MODEL_A_NAME} (Weighted)", history["delta"]["weighted"])
    print_stat(f"{MODEL_A_NAME} (BoN Min)",  history["delta"]["min"])
    print("-" * 30)
    print_stat(f"{MODEL_B_NAME} (Weighted)", history["orm"]["weighted"])
    print_stat(f"{MODEL_B_NAME} (BoN Last)", history["orm"]["last"])
    print("="*50)

if __name__ == "__main__":
    main()