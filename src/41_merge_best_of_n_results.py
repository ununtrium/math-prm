import os
import json
import glob
import pandas as pd
import argparse
import numpy as np
from collections import Counter

# ==========================================
# スコア計算ロジック
# ==========================================
def calculate_path_score(step_scores, method="min"):
    """ステップごとのスコアリストからパス全体のスコアを計算する"""
    if not step_scores: return -float('inf')
    
    # Noneを除外（念のため）
    valid_scores = [s for s in step_scores if s is not None]
    if not valid_scores: return -float('inf')

    if method == "min":
        return min(valid_scores)
    elif method == "mean":
        return np.mean(valid_scores)
    elif method == "last":
        return valid_scores[-1]
    elif method == "sum": 
        return sum(valid_scores)
    elif method == "product":
        # 確率空間での積（対数スコアの和）を想定している場合はsumと同じだが、
        # 生の確率(0-1)ならnp.prod。PRMは通常logit出力なのでsumが適切。
        # ここでは念のためsumと同じ扱いにする
        return sum(valid_scores)
    else:
        return min(valid_scores)

def get_majority_vote(responses):
    """Majority Voteによる予測結果の正誤を返す"""
    # 抽出された答えのリストを作成 (空でないもの)
    answers = [r["extracted"] for r in responses if r.get("extracted")]
    if not answers:
        return False
    
    # 最頻値を取得
    vote = Counter(answers).most_common(1)[0][0]
    
    # その答えを持つレスポンスのうち、どれか1つでも正解なら正解とする
    # (厳密にはgoldと突き合わせるべきだが、is_correctフラグを信頼する)
    for r in responses:
        if r.get("extracted") == vote:
            return r.get("is_correct", False)
    return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results", help="Base directory of scored results")
    parser.add_argument("--output_file", type=str, default="best_of_n_report.md", help="Output markdown file name")
    args = parser.parse_args()

    # 探索対象の集計メソッド
    AGG_METHODS = ["min", "mean", "last", "sum"]
    
    # ファイル探索
    # Path pattern: results/{bench}/{gen_model}/{prm_model}/samples_{N}/seed_{seed}.jsonl
    files = glob.glob(os.path.join(args.results_dir, "*", "*", "*", "*", "seed_*.jsonl"))
    
    # データ格納用: stats[bench][gen_model][samples][prm_model] = [ {metrics_seed0}, {metrics_seed1}... ]
    stats = {}

    print(f"Found {len(files)} result files. Processing...")

    for file_path in files:
        parts = file_path.split(os.sep)
        # 階層構造チェック
        # parts[-5]: bench, parts[-4]: gen_model, parts[-3]: prm_model, parts[-2]: samples
        if len(parts) < 5: continue
        
        bench_name = parts[-5]
        gen_model = parts[-4]
        prm_model = parts[-3]
        samples_dir = parts[-2]
        
        # 辞書初期化
        if bench_name not in stats: stats[bench_name] = {}
        if gen_model not in stats[bench_name]: stats[bench_name][gen_model] = {}
        if samples_dir not in stats[bench_name][gen_model]: stats[bench_name][gen_model][samples_dir] = {}
        if prm_model not in stats[bench_name][gen_model][samples_dir]: stats[bench_name][gen_model][samples_dir][prm_model] = []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                items = [json.loads(line) for line in f]
                
            total = len(items)
            if total == 0: continue

            # シードごとの集計結果
            seed_metrics = {m: 0 for m in AGG_METHODS}
            seed_metrics["majority_vote"] = 0
            seed_metrics["avg_k"] = 0 # 全サンプルの平均正解率 (Pass@1の期待値)
            seed_metrics["oracle"] = 0 # N個の中に正解が1つでもある確率

            for item in items:
                responses = item["responses"]
                if not responses: continue

                # 1. Majority Vote
                if get_majority_vote(responses):
                    seed_metrics["majority_vote"] += 1
                
                # 2. Avg@k & Oracle
                correct_count = sum(1 for r in responses if r.get("is_correct", False))
                seed_metrics["avg_k"] += correct_count / len(responses)
                
                if correct_count > 0:
                    seed_metrics["oracle"] += 1

                # 3. Best-of-N (各メソッド)
                for method in AGG_METHODS:
                    best_score = -float('inf')
                    best_is_correct = False
                    
                    # 全レスポンスをスキャンして最高スコアを探す
                    for resp in responses:
                        # step_scoresがない場合や空の場合はスキップ
                        if not resp.get("step_scores"): continue
                        
                        score = calculate_path_score(resp["step_scores"], method)
                        
                        # 同点の場合は最初に見つけたものを採用（あるいはランダムなど定義次第だがシンプルに）
                        if score > best_score:
                            best_score = score
                            best_is_correct = resp.get("is_correct", False)
                    
                    if best_is_correct:
                        seed_metrics[method] += 1

            # 精度(%)に変換してリストに追加
            final_metrics = {k: (v / total * 100) for k, v in seed_metrics.items()}
            stats[bench_name][gen_model][samples_dir][prm_model].append(final_metrics)

        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    # ==========================================
    # レポート作成 (DataFrame化)
    # ==========================================
    display_records = []
    excel_records = []
    
    for bench in stats:
        for gen in stats[bench]:
            for s_dir in stats[bench][gen]:
                for prm in stats[bench][gen][s_dir]:
                    results_list = stats[bench][gen][s_dir][prm] # list of dicts
                    num_seeds = len(results_list)
                    if num_seeds == 0: continue
                    
                    # 指標ごとにリスト化
                    metrics_data = {k: [] for k in results_list[0].keys()}
                    for res in results_list:
                        for k, v in res.items():
                            metrics_data[k].append(v)
                    
                    # ベース情報
                    base_info = {
                        "Benchmark": bench,
                        "Generator": gen,
                        "Samples": s_dir,
                        "PRM": prm,
                        "Seeds": num_seeds,
                    }
                    
                    disp_row = base_info.copy()
                    excel_row = base_info.copy()
                    
                    # 指標ごとに集計 (Mean, Max, Std)
                    for k, v_list in metrics_data.items():
                        mean_val = np.mean(v_list)
                        max_val = np.max(v_list)
                        std_val = np.std(v_list)
                        
                        # カラム名整形
                        if k in AGG_METHODS:
                            col_base = f"BoN ({k})"
                        elif k == "majority_vote":
                            col_base = "MajVote"
                        elif k == "avg_k":
                            col_base = "Avg@k"
                        elif k == "oracle":
                            col_base = "Oracle"
                        else:
                            col_base = k

                        # Markdown/Console用: "Mean (Max)"
                        disp_row[col_base] = f"{mean_val:.2f} (Max:{max_val:.2f})"
                        
                        # ソート用 (Meanでソート, min基準)
                        if k == "min": disp_row["_sort_key"] = mean_val
                        
                        # Excel用: 分割保存
                        excel_row[f"{col_base}_Mean"] = mean_val
                        excel_row[f"{col_base}_Max"] = max_val
                        excel_row[f"{col_base}_Std"] = std_val

                    display_records.append(disp_row)
                    excel_records.append(excel_row)

    df_disp = pd.DataFrame(display_records)
    df_excel = pd.DataFrame(excel_records)
    
    if df_disp.empty:
        print("No results found.")
        return

    # カラム順序の整理
    metrics_cols_order = ["Avg@k", "MajVote", "Oracle"] + [f"BoN ({m})" for m in AGG_METHODS]
    meta_cols = ["Benchmark", "Generator", "Samples", "PRM", "Seeds"]
    
    # 実際に存在するカラムのみ抽出
    final_disp_cols = meta_cols + [c for c in metrics_cols_order if c in df_disp.columns]
    
    # ソート (Benchmark -> Generator -> Samples -> Score降順)
    sort_key = "_sort_key" if "_sort_key" in df_disp.columns else "Benchmark"
    df_disp = df_disp.sort_values(by=["Benchmark", "Generator", "Samples", sort_key], ascending=[True, True, True, False])

    # Excel用のカラム整理（メタデータ + アルファベット順）
    excel_data_cols = sorted([c for c in df_excel.columns if c not in meta_cols])
    final_excel_cols = meta_cols + excel_data_cols

    # 1. コンソール & Markdown出力
    print("\n=== Best-of-N Evaluation Report ===")
    print(df_disp[final_disp_cols].to_markdown(index=False))
    
    with open(args.output_file, "w", encoding="utf-8") as f:
        f.write(f"# Best-of-N Evaluation Report\n\n")
        f.write(df_disp[final_disp_cols].to_markdown(index=False))
    print(f"\nMarkdown Report saved to {args.output_file}")

    # 2. Excel保存
    xlsx_file = args.output_file.replace(".md", ".xlsx")
    try:
        import openpyxl # check existence
        df_excel[final_excel_cols].to_excel(xlsx_file, index=False)
        print(f"Excel Report saved to {xlsx_file}")
    except ImportError:
        print("Error: 'openpyxl' module not found. Install it with `pip install openpyxl` to save Excel files.")
    except Exception as e:
        print(f"Error saving Excel file: {e}")

if __name__ == "__main__":
    main()