import json
import os
import torch
import glob
import re
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ==========================================
# 設定
# ==========================================
# ★今回学習した新しいモデルのパス
NEW_MODEL_PATH = "models/prm_7b_30k_v3.0/checkpoint-16174" 
NEW_MODEL_NAME = "PRM_7b_30k_v3.0"

# ★過去の実験結果が入っているフォルダ (src/12の出力先)
INPUT_DIR = "data/experiments/final_comparison_1.5b_30k_v1.0"

# 新しい結果の保存先
OUTPUT_DIR = "data/experiments/evaluation_v3.0_new_prm_7b"

# 推論設定
BATCH_SIZE = 8
MAX_LENGTH = 3072
STEP_MERGE_CHARS = 50
TARGET_MAX_STEPS = 15

# GPU設定
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

# ==========================================
# ユーティリティ
# ==========================================
def reduce_step_count(steps, target_max=15, min_chars=50):
    if len(steps) <= target_max: return steps
    merged_steps = []
    buffer = ""
    for i, step in enumerate(steps):
        if i == 0: buffer = step; continue
        if len(step) < min_chars or len(buffer) < min_chars: buffer += "\n" + step
        else: merged_steps.append(buffer); buffer = step
    if buffer: merged_steps.append(buffer)
    while len(merged_steps) > target_max:
        new_merged = []
        for i in range(0, len(merged_steps), 2):
            if i + 1 < len(merged_steps): new_merged.append(merged_steps[i] + "\n" + merged_steps[i+1])
            else: new_merged.append(merged_steps[i])
        merged_steps = new_merged
        if len(merged_steps) <= 1: break
    return merged_steps

# ==========================================
# メイン処理
# ==========================================
def main():
    if not os.path.exists(INPUT_DIR):
        print(f"Error: Input directory {INPUT_DIR} not found.")
        return

    # 保存先作成
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ファイルリスト取得
    files = glob.glob(os.path.join(INPUT_DIR, "trial_*.json"))
    if not files:
        print("No trial files found.")
        return
    
    print(f"Found {len(files)} trials. Loading New Model: {NEW_MODEL_PATH}...")
    
    # モデルロード
    tokenizer = AutoTokenizer.from_pretrained(NEW_MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(
        NEW_MODEL_PATH, 
        num_labels=1, 
        torch_dtype=DTYPE, 
        device_map="auto",
        attn_implementation="flash_attention_2" # Flash Attnが使えるなら指定
    )
    model.eval()

    for fpath in files:
        fname = os.path.basename(fpath)
        print(f"\nProcessing {fname}...")
        
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        rescored_data = []
        
        for item in tqdm(data, desc="Rescoring"):
            problem = item["problem"]
            paths = item["paths"]
            
            # 新しいスコア格納用
            new_path_scores = []
            new_step_scores = []
            
            for path in paths:
                # ステップ分割 (学習時と同じロジック)
                raw_steps = [s.strip() for s in re.split(r'\n\s*\n', path) if s.strip()]
                if not raw_steps: raw_steps = [s.strip() for s in path.split('\n') if s.strip()]
                steps = reduce_step_count(raw_steps, target_max=TARGET_MAX_STEPS, min_chars=STEP_MERGE_CHARS)
                
                if not steps:
                    new_path_scores.append(-99.0)
                    new_step_scores.append([-99.0])
                    continue
                
                # 入力作成
                step_inputs = []
                curr_text = problem
                for step in steps:
                    curr_text += "\n" + step
                    step_inputs.append(curr_text)
                
                # 推論
                current_rewards = []
                with torch.no_grad():
                    for i in range(0, len(step_inputs), BATCH_SIZE):
                        batch = step_inputs[i : i+BATCH_SIZE]
                        inputs = tokenizer(
                            batch, 
                            return_tensors="pt", 
                            padding=True, 
                            truncation=True, 
                            max_length=MAX_LENGTH
                        ).to(model.device)
                        
                        out = model(**inputs)
                        current_rewards.extend(out.logits.squeeze(-1).tolist())
                
                # 集計 (Min)
                final_score = min(current_rewards) if current_rewards else -99.0
                
                new_path_scores.append(final_score)
                new_step_scores.append(current_rewards)
            
            # 既存データに新しいスコアを追加
            new_item = item.copy()
            # "scores_new" というキーで保存
            new_item["scores_new"] = new_path_scores
            new_item["step_scores_new"] = new_step_scores
            
            rescored_data.append(new_item)
            
        # 保存
        out_path = os.path.join(OUTPUT_DIR, fname)
        print(f"Saving to {out_path}...")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(rescored_data, f, ensure_ascii=False, indent=2)

    print("\nDone! All trials rescored.")

if __name__ == "__main__":
    main()