import json
import argparse
import random
from collections import defaultdict
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Original large jsonl")
    parser.add_argument("--output_file", type=str, required=True, help="Output subset jsonl")
    parser.add_argument("--ratio", type=float, default=0.5, help="Keep ratio (e.g. 0.5)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    print(f"Reading {args.input_file}...")
    
    # source_id ごとにデータをグループ化
    data_by_problem = defaultdict(list)
    
    with open(args.input_file, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            try:
                rec = json.loads(line)
                # 問題ID (source_id) をキーにする
                # もし source_id がなければ一意なキー扱いでそのまま追加
                key = rec.get("source_id", random.random())
                data_by_problem[key].append(line)
            except:
                continue

    # 問題IDのリストを取得してシャッフル
    problem_ids = list(data_by_problem.keys())
    random.shuffle(problem_ids)
    
    # 指定割合だけ抽出
    n_keep = int(len(problem_ids) * args.ratio)
    keep_ids = set(problem_ids[:n_keep])
    
    print(f"Total Problems: {len(problem_ids)}")
    print(f"Selected Problems: {len(keep_ids)} (Ratio: {args.ratio})")
    
    # 保存
    print(f"Writing to {args.output_file}...")
    count = 0
    with open(args.output_file, "w", encoding="utf-8") as f:
        for pid in keep_ids:
            lines = data_by_problem[pid]
            for line in lines:
                f.write(line)
                count += 1
                
    print(f"Done. Total lines: {count}")

if __name__ == "__main__":
    main()