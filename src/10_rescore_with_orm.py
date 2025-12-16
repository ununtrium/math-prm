import json
import os
import torch
import re
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ==========================================
# 1. 設定パラメータ
# ==========================================
# ★重要★ Delta-PRMで生成したときの結果ファイルを指定
INPUT_JSON = "data/math500_results_full_scores_30k_v1.0.json"

# ORMモデルのパス
ORM_MODEL_PATH = "models/orm_1.5b_30k_v1.0"  # 実際のパスに合わせて変更してください

# 出力ファイル名
OUTPUT_JSON = "data/math500_results_orm_fixed_set_30k_v1.0.json"

# 推論設定
BATCH_SIZE = 8
MAX_LENGTH = 3072
STEP_MERGE_CHARS = 50
TARGET_MAX_STEPS = 15

# GPU設定
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

# ==========================================
# 2. ユーティリティ (前処理)
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
# 3. メイン処理
# ==========================================
def main():
    if not os.path.exists(INPUT_JSON):
        print(f"Error: {INPUT_JSON} not found.")
        return

    print(f"Loading data from {INPUT_JSON}...")
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print(f"Loading ORM Model from {ORM_MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(ORM_MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(
        ORM_MODEL_PATH, 
        num_labels=1, 
        torch_dtype=DTYPE,
        device_map="auto"
    )
    model.eval()

    print(f"Re-scoring {len(data)} problems...")
    
    rescored_data = []
    
    for item in tqdm(data, desc="Rescoring"):
        problem = item["problem"]
        paths = item["paths"]
        
        # 既存のスコアは上書きするので、新しいリストを用意
        new_path_scores = []
        new_step_scores = []
        
        for path in paths:
            # 1. ステップ分割 (学習時と同じ)
            raw_steps = [s.strip() for s in re.split(r'\n\s*\n', path) if s.strip()]
            if not raw_steps: 
                raw_steps = [s.strip() for s in path.split('\n') if s.strip()]
            
            steps = reduce_step_count(raw_steps, target_max=TARGET_MAX_STEPS, min_chars=STEP_MERGE_CHARS)
            
            if not steps:
                new_path_scores.append(-99.0)
                new_step_scores.append([-99.0])
                continue
            
            # 2. 入力作成
            step_inputs = []
            curr_text = problem
            for step in steps:
                curr_text += "\n" + step
                step_inputs.append(curr_text)
            
            # 3. 推論
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
            
            # 4. 集計 (Min)
            final_score = min(current_rewards) if current_rewards else -99.0
            
            new_path_scores.append(final_score)
            new_step_scores.append(current_rewards)
        
        # データを更新（元の情報は保持しつつ、スコアだけ書き換え）
        new_item = item.copy()
        new_item["scores"] = new_path_scores
        new_item["step_scores"] = new_step_scores
        new_item["model_name"] = "ORM" # 識別用タグ
        
        rescored_data.append(new_item)

    # 保存
    print(f"Saving rescored results to {OUTPUT_JSON}...")
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(rescored_data, f, ensure_ascii=False, indent=2)
        
    print("Done! Now run src/05_recalculate.py with this new file.")

if __name__ == "__main__":
    main()