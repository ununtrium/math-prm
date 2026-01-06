import json
import os
from tqdm import tqdm

# 設定
INPUT_FILE = "data/annotated_train_data_30k.jsonl"
OUTPUT_FILE = "data/orm_train_data_30k.jsonl"

# ORMのラベル設定
POSITIVE_LABEL = 1.0
NEGATIVE_LABEL = -1.0  # Delta-PRMの下限に合わせて調整 (-1.0 または 0.0)

def main():
    print(f"Converting {INPUT_FILE} to ORM format...")
    
    with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
        
        for line in tqdm(fin):
            record = json.loads(line)
            
            # Outcomeに基づいてラベルを一律に上書き
            if record["is_outcome_correct"]:
                record["label"] = POSITIVE_LABEL
            else:
                record["label"] = NEGATIVE_LABEL
            
            # メタデータとして記録（分析用）
            record["is_orm"] = True
            
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            
    print(f"Done! Saved to {OUTPUT_FILE}")
    print("Next step: Run src/03_train.py with TRAIN_FILE='data/orm_train_data.jsonl'")

if __name__ == "__main__":
    main()