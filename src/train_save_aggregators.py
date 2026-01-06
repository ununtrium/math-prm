import os
import json
import glob
import argparse
import joblib  # モデル保存用
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from scipy.special import logit  # 確率 -> ロジット変換 (逆シグモイド)

# ==========================================
# 1. 特徴量エンジニアリング (ロジット統一版)
# ==========================================
def extract_features(step_scores):
    """
    ステップスコアを全て「ロジット(-inf ~ +inf)」空間に統一して特徴量を抽出する。
    """
    if not step_scores:
        return [0.0] * 9

    # None除去 & float化
    scores = np.array([float(s) for s in step_scores if s is not None])
    
    if len(scores) == 0:
        return [0.0] * 9

    # --- ロジット空間への統一 ---
    # もしスコアが全て [0, 1] の範囲内なら「確率」とみなしてロジットに変換する
    # そうでなければ「既にロジット」とみなしてそのまま使う
    logits = scores

    # --- 統計量の計算 (全てロジットベース) ---
    _min = np.min(logits)
    _mean = np.mean(logits)
    _max = np.max(logits)
    _std = np.std(logits)
    _last = logits[-1]
    _first = logits[0]
    
    # 後半の崩れ (ラスト3ステップの最小値)
    _min_last_3 = np.min(logits[-3:]) if len(logits) >= 3 else _min
    
    # ステップ数
    _len = float(len(logits))
    
    # ロジットの和 (確率の積に相当するが、ロジット空間での加算として扱う)
    # これが「合計スコア」として機能する
    _sum_logits = np.sum(logits)

    features = [
        _min, _mean, _max, _std, _last, _first, _min_last_3, _len, _sum_logits
    ]
    
    # NaN/Inf 対策
    features = np.nan_to_num(features, nan=0.0, posinf=1e5, neginf=-1e5).tolist()

    return features

# ==========================================
# 2. データロード & 学習
# ==========================================
def process_and_save_single_prm(train_path, output_dir, model_type="gbdt"):
    # PRM名をパスから推定 (例: .../prm_name/samples_16/...)
    # ディレクトリ構造に依存するため、適宜調整
    parts = train_path.split(os.sep)
    # 例: results/numina_train/Qwen.../prm_name/samples_16/...
    # generator名とprm名を連結してIDにする
    try:
        prm_name = parts[-2] # samples_16の2つ上
        gen_name = parts[-3] # そのさらに上
        model_id = f"{gen_name}__{prm_name}"
    except:
        model_id = "unknown_prm"

    print(f"Processing: {model_id}")
    print(f"  Path: {train_path}")

    # --- データ読み込み ---
    files = glob.glob(os.path.join(train_path, "**", "seed_*.jsonl"), recursive=True)
    X_all, y_all = [], []

    for file_path in tqdm(files, desc="Loading", leave=False):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    item = json.loads(line)
                    for r in item.get("responses", []):
                        step_scores = r.get("step_scores", [])
                        if not step_scores: continue
                        
                        feat = extract_features(step_scores)
                        label = 1 if r.get("is_correct", False) else 0
                        
                        X_all.append(feat)
                        y_all.append(label)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    if not X_all:
        print("  No data found. Skipping.")
        return

    X_train = np.array(X_all)
    y_train = np.array(y_all)

    # --- モデル学習 ---
    if model_type == "gbdt":
        model = GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
        )
    else:
        model = LogisticRegression(random_state=42, max_iter=1000)

    model.fit(X_train, y_train)

    # --- 保存 ---
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{model_id}.pkl")
    joblib.dump(model, save_path)
    
    print(f"  Saved model to: {save_path}")

# ==========================================
# 3. メイン処理
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_train_dir", type=str, required=True, help="Root dir of training data (e.g., results/numina_train)")
    parser.add_argument("--save_dir", type=str, default="models/learned_aggregators", help="Where to save .pkl models")
    parser.add_argument("--model_type", type=str, default="gbdt", choices=["gbdt", "logistic"])
    args = parser.parse_args()

    # ディレクトリ探索: base_train_dir/*/ *(Generator) / *(PRM) / samples_*
    # 構造に合わせてglobパターンを調整してください
    # 例: results/numina_train/Qwen-1.5B/prm_model_name/samples_16
    search_pattern = os.path.join(args.base_train_dir, "*", "*", "samples_*")
    prm_dirs = glob.glob(search_pattern)

    print(f"Found {len(prm_dirs)} potential PRM directories.")

    for prm_dir in prm_dirs:
        if "no_trigger" not in prm_dir:
            print(f"Skipping (no 'no_trigger' in path): {prm_dir}")
            continue
        process_and_save_single_prm(prm_dir, args.save_dir, args.model_type)

if __name__ == "__main__":
    main()