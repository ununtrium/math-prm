import os
import json
import glob
import argparse
import pandas as pd

def check_files(target_dir):
    # ファイルを探す
    files = glob.glob(os.path.join(target_dir, "**", "seed_*.jsonl"), recursive=True)
    
    if not files:
        print(f"No files found in {target_dir}")
        return

    report = []
    
    print(f"Checking {len(files)} files in: {target_dir}\n")

    for file_path in sorted(files):
        # パスを見やすく整形
        rel_path = os.path.relpath(file_path, target_dir)
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            
            total_lines = len(lines)
            valid_entries = 0
            scored_entries = 0
            
            for line in lines:
                try:
                    item = json.loads(line)
                    responses = item.get("responses", [])
                    
                    if responses:
                        valid_entries += 1
                        
                        # スコアリング済みかチェック (step_scoresがあるか)
                        # 最初のレスポンスだけ見れば十分
                        if "step_scores" in responses[0] and responses[0]["step_scores"]:
                            scored_entries += 1
                            
                except json.JSONDecodeError:
                    pass # JSON破損など

            # ステータス判定
            # AIME25は30問、Math500は500問などベンチマークによるが、
            # 明らかに少ないものを検知する
            status = "OK"
            if total_lines == 0:
                status = "EMPTY"
            elif valid_entries < total_lines:
                status = "BROKEN JSON"
            elif scored_entries < total_lines:
                status = "UNSCORED" # 生成はあるがPRMスコアがない
            
            # ディレクトリ名からベンチマーク名などを推測
            parts = rel_path.split(os.sep)
            # 例: aime25/Qwen/prm/samples_64/seed_0.jsonl
            bench_name = parts[0] if len(parts) > 0 else "?"
            seed_name = os.path.basename(file_path)

            report.append({
                "Benchmark": bench_name,
                "File": seed_name,
                "Lines": total_lines,
                "Valid": valid_entries,
                "Scored": scored_entries,
                "Status": status,
                "Path": rel_path
            })

        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    # 結果表示
    if report:
        df = pd.DataFrame(report)
        # 見やすいようにカラム選択
        print(df[["Benchmark", "File", "Lines", "Scored", "Status", "Path"]].to_string(index=False))
        
        print("\n--- Summary ---")
        print(f"Total Lines found: {df['Lines'].sum()}")
        print(f"Total Scored found: {df['Scored'].sum()}")
        print("(If you expect 150 for AIME25 (5 seeds), compare with Total Scored)")

def main():
    parser = argparse.ArgumentParser()
    # resultsフォルダを指定
    parser.add_argument("--target_dir", type=str, default="results", help="Directory to scan")
    args = parser.parse_args()
    
    check_files(args.target_dir)

if __name__ == "__main__":
    main()