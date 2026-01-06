import json
import argparse
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference_file", type=str, required=True, 
                        help="Path to the already halved file (e.g., prm_train_half.jsonl)")
    parser.add_argument("--target_file", type=str, required=True, 
                        help="Path to the full file you want to filter (e.g., prm_train_qwen_2M.jsonl)")
    parser.add_argument("--output_file", type=str, required=True, 
                        help="Output path for the filtered file")
    args = parser.parse_args()

    print(f"Step 1: collecting source_ids from {args.reference_file}...")
    valid_ids = set()
    
    with open(args.reference_file, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            try:
                rec = json.loads(line)
                if "source_id" in rec:
                    valid_ids.add(rec["source_id"])
            except:
                continue
    
    print(f"Found {len(valid_ids)} unique problems in reference file.")

    print(f"Step 2: Filtering {args.target_file}...")
    count = 0
    skipped = 0
    
    with open(args.target_file, "r", encoding="utf-8") as fin, \
         open(args.output_file, "w", encoding="utf-8") as fout:
        
        for line in tqdm(fin):
            try:
                rec = json.loads(line)
                sid = rec.get("source_id")
                
                # 参照ファイルにある問題IDなら書き出す
                if sid in valid_ids:
                    fout.write(line)
                    count += 1
                else:
                    skipped += 1
            except:
                continue

    print(f"Done.")
    print(f" - Kept lines: {count}")
    print(f" - Skipped:    {skipped}")
    print(f"Output saved to {args.output_file}")

if __name__ == "__main__":
    main()