import json
import argparse
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="data/prm_train_ensemble_2M.jsonl")
    parser.add_argument("--output_file", type=str, default="data/orm_train_stepwise.jsonl")
    args = parser.parse_args()

    print(f"Creating Step-wise ORM data from {args.input_file}...")
    
    count = 0
    with open(args.input_file, "r", encoding="utf-8") as fin, \
         open(args.output_file, "w", encoding="utf-8") as fout:
        
        for line in tqdm(fin):
            try:
                rec = json.loads(line)
                
                # 最終結果が合っているかどうか
                is_correct = rec["is_outcome_correct"]
                
                # ORMとしての教師ラベル (Hard Label)
                if is_correct:
                    target_raw_prob = 1.0
                    target_log_prob = 0.0
                else:
                    target_raw_prob = 0.0
                    target_log_prob = -100.0 # 実質ゼロ
                
                # 全ての関連フィールドを書き換え
                rec["raw_prob"] = target_raw_prob
                rec["log_prob"] = target_log_prob
                
                # Deltaは概念として存在しないため完全にゼロにする
                rec["raw_prob_delta"] = 0.0
                rec["log_prob_delta"] = 0.0
                
                # 識別用にID変更
                rec["model_id"] = "ground_truth_outcome"
                
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                count += 1
            except:
                continue
                
    print(f"Done! Created {count} ORM samples.")

if __name__ == "__main__":
    main()