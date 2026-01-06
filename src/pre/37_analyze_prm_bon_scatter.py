import argparse
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ==========================================
# スコア集計関数の定義
# ==========================================
def agg_min(scores):
    return np.min(scores) if scores else 0.0

def agg_mean(scores):
    return np.mean(scores) if scores else 0.0

def agg_max(scores):
    return np.max(scores) if scores else 0.0

def agg_last(scores):
    return scores[-1] if scores else 0.0

def agg_sum(scores):
    return np.sum(scores) if scores else 0.0

def agg_prod(scores):
    # 確率として扱う場合の積 (アンダーフロー注意だが簡易的に)
    # スコアがlogitの場合は exp してから計算などの調整が必要だが、
    # ここでは単純な積、または log sum とする
    return np.prod(scores) if scores else 0.0

AGG_FUNCS = {
    "min": agg_min,
    "mean": agg_mean,
    "max": agg_max,
    "last": agg_last,
    "sum": agg_sum,
    "prod": agg_prod
}

# ==========================================
# 設定と引数
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="Plot PRM Scatter Analysis")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the rescored JSON file.")
    parser.add_argument("--output_file", type=str, default="prm_scatter.png", help="Path to save the plot image.")
    
    # 軸の設定
    parser.add_argument("--x_metric", type=str, choices=AGG_FUNCS.keys(), default="mean", help="Metric for X-axis")
    parser.add_argument("--y_metric", type=str, choices=AGG_FUNCS.keys(), default="min", help="Metric for Y-axis")
    
    # データフィルタリング（重すぎる場合用）
    parser.add_argument("--max_points", type=int, default=10000, help="Max number of points to plot (random sample).")
    
    return parser.parse_args()

# ==========================================
# メイン処理
# ==========================================
def main():
    args = parse_args()
    
    print(f"Loading data from {args.input_file}...")
    with open(args.input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    # JSON構造の正規化 (detailsキーがある場合とない場合に対応)
    items = data.get("details", data) if isinstance(data, dict) else data
    
    plot_data = []
    
    print("Processing items...")
    for item in tqdm(items):
        # generated_samples, paths, または top-level の構造に対応
        # 前回のコードの出力形式 (rescored_items) に準拠
        
        samples = []
        
        # パターン1: generated_samples 内に情報がある場合
        if "generated_samples" in item:
            samples = item["generated_samples"]
        # パターン2: ルートに step_scores_new がある場合 (1つだけの解など)
        elif "step_scores_new" in item:
            # 複数のパスがフラットに入っている場合を想定
            # 前回のコードでは generated_samples に書き戻しているのでパターン1がメイン
            pass
            
        for sample in samples:
            # スコアの取得 (newがあればnewを優先)
            if "step_scores" in sample:
                scores = sample["step_scores"]
            else:
                continue # スコアがないものはスキップ
                
            # 正誤判定情報の取得
            # 事前に正誤判定がついている前提 (is_correct)
            if "is_correct" not in sample:
                continue
            
            is_correct = sample["is_correct"]
            
            # 集計計算
            x_val = AGG_FUNCS[args.x_metric](scores)
            y_val = AGG_FUNCS[args.y_metric](scores)
            
            plot_data.append({
                "X": x_val,
                "Y": y_val,
                "Correctness": "Correct" if is_correct else "Incorrect"
            })

    # DataFrame化
    df = pd.DataFrame(plot_data)
    print(f"Total paths collected: {len(df)}")
    
    if len(df) == 0:
        print("Error: No valid data points found. Check if 'step_scores' and 'is_correct' exist in JSON.")
        return

    # データが多すぎる場合のサンプリング
    if len(df) > args.max_points:
        print(f"Sampling {args.max_points} points for visualization...")
        df = df.sample(n=args.max_points, random_state=42)

    # ==========================================
    # プロット作成
    # ==========================================
    plt.figure(figsize=(10, 8))
    sns.set_theme(style="whitegrid")
    
    # カラーパレット: 正解=青, 不正解=赤
    palette = {"Correct": "blue", "Incorrect": "red"}
    
    # 散布図
    # alpha: 重なりを見るために透明度を下げるのが重要
    sns.scatterplot(
        data=df, 
        x="X", 
        y="Y", 
        hue="Correctness", 
        palette=palette,
        alpha=0.3, 
        edgecolor=None,
        s=20 # 点のサイズ
    )
    
    # ガイドライン (y=x) を引く（軸が同じスケールの場合に有用）
    min_val = min(df["X"].min(), df["Y"].min())
    max_val = max(df["X"].max(), df["Y"].max())
    # 軸の種類が違うと比較できないので、Min/Meanなどの0-1スコアの場合のみ線を引く
    if args.x_metric in ["min", "mean", "max", "last"] and args.y_metric in ["min", "mean", "max", "last"]:
        plt.plot([min_val, max_val], [min_val, max_val], color='gray', linestyle='--', alpha=0.5, label='y=x')

    plt.title(f"PRM Analysis: {args.x_metric.capitalize()} vs {args.y_metric.capitalize()}\n(N={len(df)} paths)", fontsize=14)
    plt.xlabel(f"{args.x_metric.capitalize()} Score", fontsize=12)
    plt.ylabel(f"{args.y_metric.capitalize()} Score", fontsize=12)
    plt.legend(title="Answer Outcome")
    
    # 保存
    print(f"Saving plot to {args.output_file}...")
    plt.savefig(args.output_file, dpi=300, bbox_inches='tight')
    print("Done.")

if __name__ == "__main__":
    main()