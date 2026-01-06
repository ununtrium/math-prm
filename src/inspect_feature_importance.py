import os
import glob
import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# 特徴量名は extract_features の順番と完全に一致させる必要があります
FEATURE_NAMES = [
    "Min (Logic Break)",       # 0: 最小値 (論理の破綻点)
    "Mean (Coherence)",        # 1: 平均値 (全体の一貫性)
    "Max (Peak)",              # 2: 最大値
    "Std (Instability)",       # 3: 標準偏差 (不安定さ)
    "Last (Conclusion)",       # 4: 最後のステップ (結論の確度)
    "First (Start)",           # 5: 最初のステップ
    "Min Last 3 (Late Break)", # 6: 終盤の最小値 (結論直前のミス)
    "Length (Complexity)",     # 7: ステップ数
    "Sum Logits (Joint Prob)"  # 8: ロジット和 (確率の積相当)
]

def analyze_model(model_path):
    try:
        model = joblib.load(model_path)
    except Exception as e:
        print(f"Error loading {model_path}: {e}")
        return None

    model_name = os.path.basename(model_path).replace(".pkl", "")
    
    # モデルの種類によって重要度の取り方が違う
    if hasattr(model, "feature_importances_"):
        # GBDT / XGBoost / Random Forest
        # 値は 0~1 の相対重要度 (合計すると1になる)
        importances = model.feature_importances_
        model_type = "GBDT"
    elif hasattr(model, "coef_"):
        # Logistic Regression
        # 値は係数 (絶対値が大きいほど影響大)
        importances = model.coef_[0]
        model_type = "Logistic"
    else:
        print(f"Unknown model type: {model_path}")
        return None

    # データフレーム化
    df = pd.DataFrame({
        "Feature": FEATURE_NAMES,
        "Importance": importances,
        "Model": model_name,
        "Type": model_type
    })
    
    # 重要度の絶対値でソート（ロジスティック回帰の負の係数対策）
    df["AbsImportance"] = df["Importance"].abs()
    df = df.sort_values("AbsImportance", ascending=False)
    
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_dir", type=str, default="models/learned_aggregators", help="Directory containing .pkl models")
    parser.add_argument("--output_csv", type=str, default="feature_importance_summary.csv")
    args = parser.parse_args()

    model_files = glob.glob(os.path.join(args.models_dir, "*.pkl"))
    print(f"Found {len(model_files)} models in {args.models_dir}")

    all_results = []

    for path in model_files:
        df = analyze_model(path)
        if df is not None:
            all_results.append(df)
            
            # コンソールにも簡易表示 (Top 3)
            print(f"\nModel: {df.iloc[0]['Model']}")
            print(df[["Feature", "Importance"]].head(3).to_string(index=False))

    if not all_results:
        print("No models analyzed.")
        return

    # 全結合して保存
    final_df = pd.concat(all_results, ignore_index=True)
    final_df.to_csv(args.output_csv, index=False)
    print(f"\nSaved full details to {args.output_csv}")

    # --- 平均重要度の集計 (論文用の図表作成に便利) ---
    # GBDT系のみで集計（ロジスティックと混ぜると数値の意味が変わるため）
    gbdt_df = final_df[final_df["Type"] == "GBDT"]
    
    if not gbdt_df.empty:
        avg_importance = gbdt_df.groupby("Feature")["Importance"].mean().reset_index()
        avg_importance = avg_importance.sort_values("Importance", ascending=False)
        
        print("\n" + "="*40)
        print("AVERAGE IMPORTANCE (All GBDT Models)")
        print("="*40)
        print(avg_importance.to_string(index=False))
        
        # 簡易プロット保存
        plt.figure(figsize=(10, 6))
        sns.barplot(data=avg_importance, x="Importance", y="Feature", palette="viridis")
        plt.title("Average Feature Importance across all PRMs")
        plt.tight_layout()
        plt.savefig("feature_importance_plot.png")
        print("Saved plot to feature_importance_plot.png")

if __name__ == "__main__":
    main()