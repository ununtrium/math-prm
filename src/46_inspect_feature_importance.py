import os
import glob
import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 特徴量定義 (以前のコードと完全一致させる)
FEATURE_NAMES = [
    "Min (Logic Break)",       # 0
    "Mean (Coherence)",        # 1
    "Max (Peak)",              # 2
    "Std (Instability)",       # 3
    "Last (Conclusion)",       # 4
    "First (Start)",           # 5
    "Min Last 3 (Late Break)", # 6
    "Length (Complexity)",     # 7
    "Sum Logits (Joint Prob)"  # 8
]

def analyze_model(model_path):
    try:
        model = joblib.load(model_path)
    except Exception as e:
        print(f"Error loading {model_path}: {e}")
        return None

    model_name = os.path.basename(model_path).replace(".pkl", "")
    
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        model_type = "GBDT"
    elif hasattr(model, "coef_"):
        importances = model.coef_[0]
        model_type = "Logistic"
    else:
        return None

    return pd.DataFrame({
        "Feature": FEATURE_NAMES,
        "Importance": importances,
        "Model": model_name,
        "Type": model_type
    })

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_dir", type=str, default="models/learned_aggregators", help="Directory containing .pkl models")
    args = parser.parse_args()

    model_files = glob.glob(os.path.join(args.models_dir, "*.pkl"))
    all_results = []

    print(f"Searching in {args.models_dir}...")
    for path in model_files:
        model_name = os.path.basename(path).lower()
        
        # --- フィルタリング: no_trigger と orm を除外 ---
        if "no_trigger" in model_name or "orm" in model_name:
            continue
        
        df = analyze_model(path)
        if df is not None:
            all_results.append(df)

    if not all_results:
        print("No valid PRM models found after filtering.")
        return

    final_df = pd.concat(all_results, ignore_index=True)

    # ==========================================
    # 1. GBDTによる重要度の可視化 (寄与度)
    # ==========================================
    gbdt_df = final_df[final_df["Type"] == "GBDT"]
    if not gbdt_df.empty:
        avg_imp = gbdt_df.groupby("Feature")["Importance"].mean().sort_values(ascending=False).reset_index()
        
        print("\n=== AVERAGE IMPORTANCE (GBDT - Magnitude) ===")
        print(avg_imp.to_string(index=False))

        plt.figure(figsize=(10, 6))
        sns.barplot(data=avg_imp, x="Importance", y="Feature", hue="Feature", palette="viridis", legend=False)
        plt.title("Pure PRM Feature Importance (Magnitude)")
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig("prm_importance_magnitude.png")

    # ==========================================
    # 2. Logistic Regressionによる係数の可視化 (方向性)
    # ==========================================
    logic_df = final_df[final_df["Type"] == "Logistic"]
    if not logic_df.empty:
        avg_coef = logic_df.groupby("Feature")["Importance"].mean().sort_values(ascending=False).reset_index()
        
        print("\n=== AVERAGE COEFFICIENTS (Logistic - Direction) ===")
        print(avg_coef.to_string(index=False))

        plt.figure(figsize=(10, 6))
        # 正負で色分け (Positive: Blue, Negative: Red)
        colors = ["#4C72B0" if x > 0 else "#C44E52" for x in avg_coef["Importance"]]
        sns.barplot(data=avg_coef, x="Importance", y="Feature", palette=colors)
        plt.axvline(0, color='black', linewidth=1)
        plt.title("Pure PRM Feature Directionality (Positive=Correct, Negative=Incorrect)")
        plt.xlabel("Coefficient Value")
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig("prm_directionality.png")

    print("\nAnalysis complete. Plots saved: 'prm_importance_magnitude.png' and 'prm_directionality.png'")

if __name__ == "__main__":
    main()