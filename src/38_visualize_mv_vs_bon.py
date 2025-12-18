import argparse
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize MV vs BoN outcomes using pre-calculated correctness")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the JSON file with scores and correctness labels.")
    parser.add_argument("--output_image", type=str, default="mv_vs_bon_scatter.png", help="Path to save the output scatter plot.")
    return parser.parse_args()

def main():
    args = parse_args()

    # 1. データ読み込み
    print(f"Loading data from {args.input_file}...")
    if not os.path.exists(args.input_file):
        print(f"Error: File {args.input_file} not found.")
        return

    with open(args.input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # "details" キーの中にリストがある構造に対応
    items = data.get("details", [])
    if not items:
        # detailsがない場合、ルートがリストの可能性も考慮
        if isinstance(data, list):
            items = data
        else:
            print("Error: Could not find list of problems in JSON.")
            return

    print(f"Analyzing {len(items)} problems...")

    results = []
    
    # 2. データ処理
    for item in items:
        # A. 多数決の判定 (JSONにすでにあるフラグを使用)
        mv_is_correct = item.get("majority_is_correct", False)
        
        samples = item.get("generated_samples", [])
        if not samples:
            continue

        # B. BoN (Best-of-N) の判定
        # 基準: final_score (Min Score) が最大のパスを選ぶ
        # (同点の場合はリストの最初に出現した方を選択)
        try:
            best_path = max(samples, key=lambda x: x.get("final_score", -float('inf')))
        except ValueError:
            continue # サンプルが空の場合など

        # BoNで選ばれたパスの情報を取得
        bon_is_correct = best_path.get("is_correct", False)
        min_score = best_path.get("final_score", 0.0)
        
        # Last Scoreの取得 (step_scoresの最後の要素)
        step_scores = best_path.get("step_scores", [])
        if step_scores:
            last_score = step_scores[-1]
        else:
            last_score = min_score # フォールバック

        # C. カテゴリ分類 (4色)
        if mv_is_correct and bon_is_correct:
            category = 'Both Correct'
        elif not mv_is_correct and not bon_is_correct:
            category = 'Both Incorrect'
        elif mv_is_correct and not bon_is_correct:
            category = 'MV Only' # 多数決のみ正解 (BoNは間違ったパスを選んだ)
        else:
            category = 'BoN Only' # BoNのみ正解 (PRMのファインプレー)

        results.append({
            "Last Score": last_score,
            "Min Score": min_score,
            "Category": category
        })

    # DataFrame作成
    df = pd.DataFrame(results)
    
    if df.empty:
        print("No valid data found to plot.")
        return

    # カテゴリごとの件数を表示
    print("\nOutcome Distribution:")
    print(df["Category"].value_counts())

    # 3. 散布図の描画
    plt.figure(figsize=(11, 9))
    sns.set_theme(style="whitegrid")

    # カラーパレット定義
    palette = {
        'Both Correct': '#2ca02c',  # Green
        'Both Incorrect': '#d62728',# Red
        'MV Only': '#1f77b4',       # Blue (多数決の強み)
        'BoN Only': '#ff7f0e'       # Orange (PRMの強み)
    }
    
    # 描画順序 (目立たせたい点を手前に)
    hue_order = ['Both Incorrect', 'Both Correct', 'MV Only', 'BoN Only']

    # Scatter Plot
    sns.scatterplot(
        data=df,
        x='Last Score',
        y='Min Score',
        hue='Category',
        palette=palette,
        hue_order=hue_order,
        style='Category', # 形でも区別
        s=80,             # 点のサイズ
        alpha=0.7,        # 透明度
        edgecolor='w',    # 点の縁取り
        linewidth=0.5
    )

    # 対角線 (y=x) の描画
    # データの範囲に基づいて線を引く
    all_scores = pd.concat([df['Last Score'], df['Min Score']])
    low_lim, high_lim = all_scores.min(), all_scores.max()
    # 少しマージンを持たせる
    buffer = (high_lim - low_lim) * 0.05
    plt.plot([low_lim - buffer, high_lim + buffer], [low_lim - buffer, high_lim + buffer], 
             linestyle='--', color='gray', alpha=0.5, label='y=x (Min <= Last)')

    # タイトルとラベル
    plt.title('Performance Analysis: Majority Voting vs PRM BoN (Min Score)\n(Points represent the single path chosen by PRM)', fontsize=15)
    plt.xlabel('Last Score of Selected Path', fontsize=12)
    plt.ylabel('Min Score of Selected Path', fontsize=12)
    
    # 凡例の位置調整
    plt.legend(title='Outcome Category', title_fontsize='11', loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    
    # 保存
    print(f"\nSaving plot to {args.output_image}...")
    plt.savefig(args.output_image, dpi=300, bbox_inches='tight')
    print("Done.")

if __name__ == "__main__":
    main()