import os
import json
import glob
import pandas as pd
import argparse
import re

def parse_filename_config(filename):
    """ファイル名から設定値を抽出 (beam{W}_cand{C}_seed{S}.json)"""
    match = re.search(r"beam(\d+)_cand(\d+)_seed(\d+)", filename)
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3))
    return None, None, None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results/tree_search", help="Directory containing tree search results")
    parser.add_argument("--output_file", type=str, default="tree_search_report", help="Output filename base (without extension)")
    args = parser.parse_args()

    # ファイル探索
    search_pattern = os.path.join(args.results_dir, "*", "*", "*", "beam*.json")
    files = glob.glob(search_pattern)
    
    if not files:
        # 再帰検索（階層構造が少し違う場合用）
        files = glob.glob(os.path.join(args.results_dir, "**", "beam*.json"), recursive=True)

    print(f"Found {len(files)} result files. Processing...")

    raw_data = []

    for file_path in files:
        try:
            path_parts = os.path.normpath(file_path).split(os.sep)
            filename = path_parts[-1]
            
            beam_width, num_cand, seed = parse_filename_config(filename)
            if beam_width is None: continue

            # 階層からモデル名などを取得
            # 期待構成: .../bench/gen_model/prm_model/filename.json
            if len(path_parts) >= 4:
                prm_model = path_parts[-2]
                gen_model = path_parts[-3]
                bench_name = path_parts[-4]
            else:
                prm_model = "unknown"
                gen_model = "unknown"
                bench_name = "unknown"

            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            accuracy = data.get("accuracy", 0.0)
            
            # コスト計算（生成ノード総数）
            total_nodes = 0
            details = data.get("details", [])
            problem_count = len(details)
            
            for item in details:
                history = item.get("tree_history", [])
                if history:
                    for step_nodes in history:
                        total_nodes += len(step_nodes)
            
            avg_nodes = total_nodes / problem_count if problem_count > 0 else 0

            raw_data.append({
                "Benchmark": bench_name,
                "Generator": gen_model,
                "PRM": prm_model,
                "Beam (W)": beam_width,
                "Cand (C)": num_cand,
                "Config": f"W={beam_width}, C={num_cand}",
                "Seed": seed,
                "Accuracy": accuracy,
                "Avg Nodes": avg_nodes
            })

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    if not raw_data:
        print("No valid data found.")
        return

    # DataFrame作成
    df_raw = pd.DataFrame(raw_data)

    # ---------------------------------------------------------
    # 集計 (Summary)
    # ---------------------------------------------------------
    group_cols = ["Benchmark", "Generator", "PRM", "Config", "Beam (W)", "Cand (C)"]
    
    summary = df_raw.groupby(group_cols).agg(
        Seeds=("Seed", "count"),
        Acc_Mean=("Accuracy", "mean"),
        Acc_Max=("Accuracy", "max"),
        Acc_Std=("Accuracy", "std"),
        Acc_Min=("Accuracy", "min"),
        Nodes_Mean=("Avg Nodes", "mean")
    ).reset_index()

    # ソート
    summary = summary.sort_values(by=["Benchmark", "Generator", "PRM", "Beam (W)", "Cand (C)"])

    # 表示用カラム作成
    summary["Accuracy (Mean ± Std)"] = summary.apply(
        lambda x: f"{x['Acc_Mean']:.2f} ± {x['Acc_Std']:.2f}", axis=1
    )
    summary["Accuracy (Max)"] = summary["Acc_Max"].map('{:.2f}'.format)
    summary["Avg Cost"] = summary["Nodes_Mean"].map('{:.1f}'.format)

    # ---------------------------------------------------------
    # 出力処理
    # ---------------------------------------------------------
    
    # 1. コンソール & Markdown
    disp_cols = ["Benchmark", "Generator", "PRM", "Config", "Seeds", "Accuracy (Mean ± Std)", "Accuracy (Max)", "Avg Cost"]
    print("\n=== Tree Search Evaluation Report ===")
    print(summary[disp_cols].to_markdown(index=False))
    
    md_file = args.output_file + ".md"
    with open(md_file, "w") as f:
        f.write("# Tree Search Evaluation Report\n\n")
        f.write(summary[disp_cols].to_markdown(index=False))
    print(f"\nMarkdown saved to: {md_file}")

    # 2. Excel保存 (シート分け機能付き)
    xlsx_file = args.output_file + ".xlsx"
    try:
        with pd.ExcelWriter(xlsx_file, engine='openpyxl') as writer:
            # シート1: 集計結果 (数値のまま保存)
            summary_save_cols = group_cols + ["Seeds", "Acc_Mean", "Acc_Max", "Acc_Min", "Acc_Std", "Nodes_Mean"]
            summary[summary_save_cols].to_excel(writer, sheet_name='Summary', index=False)
            
            # シート2: 生データ (全シードの結果)
            # シードごとのバラつきを確認するのに便利
            df_raw.sort_values(by=group_cols + ["Seed"]).to_excel(writer, sheet_name='Raw Data', index=False)
            
        print(f"Excel report saved to: {xlsx_file}")
        print("  - Sheet 'Summary': Aggregated results")
        print("  - Sheet 'Raw Data': Individual seed results")
        
    except ImportError:
        print("Error: 'openpyxl' module not found. Run `pip install openpyxl`.")
    except Exception as e:
        print(f"Error saving Excel: {e}")

if __name__ == "__main__":
    main()