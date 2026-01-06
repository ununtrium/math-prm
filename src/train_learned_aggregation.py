import os
import json
import glob
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score

# ==========================================
# 1. 特徴量エンジニアリング (修正版)
# ==========================================
def extract_features(step_scores):
    """
    可変長のステップスコアリストを、固定長の「特徴量ベクトル」に変換する。
    NaN対策とLogits対応済み。
    """
    # 空リスト対策
    if not step_scores:
        return [0.0] * 9

    # Noneを除外し、float型に変換
    scores = np.array([float(s) for s in step_scores if s is not None])
    
    if len(scores) == 0:
        return [0.0] * 9

    # --- 基本統計量 ---
    _min = np.min(scores)
    _mean = np.mean(scores)
    _max = np.max(scores)
    _std = np.std(scores)
    _last = scores[-1]
    _first = scores[0]
    
    # --- 応用特徴量 ---
    # 1. 後半の崩れを検知: ラスト3ステップの最小値
    _min_last_3 = np.min(scores[-3:]) if len(scores) >= 3 else _min
    
    # 2. 長さ（ステップ数）
    _len = float(len(scores))
    
    # 3. 確率的解釈 (Logitsの和 / 対数和)
    # スコアが負(logits)の場合はそのまま和をとり、正(確率)の場合はlogをとる
    if np.min(scores) < 0:
        # Logitsの場合、そのまま足せば「確率の積の対数」相当になる
        # 長さによるバイアスを防ぐため平均化する
        _prod_proxy = np.sum(scores) / _len
    else:
        # 確率(0~1)の場合、logをとって足す (0除算回避のため +1e-9)
        _prod_proxy = np.sum(np.log(scores + 1e-9)) / _len

    features = [
        _min, _mean, _max, _std, _last, _first, _min_last_3, _len, _prod_proxy
    ]
    
    # ★重要: NaN / Infinity を 0 や 有限値に置換してエラーを防ぐ
    # (nan=0, posinf=大きな値, neginf=小さな値)
    features = np.nan_to_num(features, nan=0.0, posinf=1e5, neginf=-1e5).tolist()

    return features

FEATURE_NAMES = [
    "Min Score", "Mean Score", "Max Score", "Std Dev", 
    "Last Step", "First Step", "Min(Last 3)", "Step Count", "Sum Logits/LogProb"
]

# ==========================================
# 2. データロード
# ==========================================
def load_data_from_jsonl(file_pattern, desc="Loading"):
    """
    指定されたパスパターンからJSONLを読み込み、
    (問題ごとの候補リスト, 全候補の特徴量X, 全候補のラベルy) を返す
    """
    files = glob.glob(file_pattern)
    if not files:
        # 再帰検索
        files = glob.glob(os.path.join(os.path.dirname(file_pattern), "**", os.path.basename(file_pattern)), recursive=True)
    
    print(f"[{desc}] Found {len(files)} files.")
    
    problems = [] # BoN計算用: [ { "candidates": [features...], "labels": [bool...] }, ... ]
    X_all = []    # 学習用: 全候補の特徴量フラットリスト
    y_all = []    # 学習用: 全候補の正解ラベルフラットリスト
    
    total_samples = 0
    
    for file_path in tqdm(files, desc=desc):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    item = json.loads(line)
                    responses = item.get("responses", [])
                    if not responses: continue

                    cand_features = []
                    cand_labels = []
                    
                    for r in responses:
                        # step_scoresが無い、または空の場合はスキップ
                        step_scores = r.get("step_scores", [])
                        if not step_scores: continue
                        
                        feat = extract_features(step_scores)
                        label = 1 if r.get("is_correct", False) else 0
                        
                        cand_features.append(feat)
                        cand_labels.append(label)
                        
                        # 学習用リストに追加
                        X_all.append(feat)
                        y_all.append(label)
                    
                    if cand_features:
                        problems.append({
                            "features": cand_features, # この問題に対するN個の候補の特徴量
                            "labels": cand_labels      # この問題に対するN個の候補の正誤
                        })
                        total_samples += len(cand_features)

        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            
    return problems, np.array(X_all), np.array(y_all)

# ==========================================
# 3. メイン処理
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, required=True, help="Directory for TRAINING data (e.g., math500 results)")
    parser.add_argument("--test_dir", type=str, required=True, help="Directory for TESTING data (e.g., aime24 results)")
    parser.add_argument("--model_type", type=str, default="gbdt", choices=["gbdt", "logistic"], help="Model type")
    args = parser.parse_args()

    # --- 1. 学習データのロード ---
    print("=== 1. Loading Training Data (Source) ===")
    train_pattern = os.path.join(args.train_dir, "**", "seed_*.jsonl")
    _, X_train, y_train = load_data_from_jsonl(train_pattern, desc="Train Data")
    
    if len(X_train) == 0:
        print("Error: No training data found.")
        return

    print(f"   Training Samples: {len(X_train)} (Positive: {sum(y_train)}, Negative: {len(y_train)-sum(y_train)})")

    # --- 2. モデル学習 ---
    print("\n=== 2. Training Aggregator Model ===")
    if args.model_type == "gbdt":
        # 勾配ブースティング (NaNにも強いが、念のため前処理済みデータを使う)
        model = GradientBoostingClassifier(
            n_estimators=100, 
            learning_rate=0.1, 
            max_depth=3, 
            random_state=42
        )
    else:
        # ロジスティック回帰
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        model = LogisticRegression(random_state=42)

    model.fit(X_train, y_train)
    print("   Training finished.")

    # 特徴量重要度の表示
    if hasattr(model, "feature_importances_"):
        print("\n   [Feature Importance]")
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        for i in range(len(FEATURE_NAMES)):
            idx = indices[i]
            print(f"   {i+1}. {FEATURE_NAMES[idx]:<16}: {importances[idx]:.4f}")

    # --- 3. テストデータのロード ---
    print("\n=== 3. Loading Test Data (Target) ===")
    test_pattern = os.path.join(args.test_dir, "**", "seed_*.jsonl")
    test_problems, _, _ = load_data_from_jsonl(test_pattern, desc="Test Data")
    
    if len(test_problems) == 0:
        print("No test data found.")
        return

    # --- 4. 評価 ---
    print("\n=== 4. Evaluating on Test Data ===")
    
    # 比較用: 従来の単純集計での精度を計算するための関数
    def calc_baseline_acc(problems, method_idx):
        # method_idx: 0=min, 1=mean, 4=last (extract_featuresの順序依存)
        correct = 0
        total = 0
        for p in problems:
            feats = np.array(p["features"])
            labels = np.array(p["labels"])
            
            # 該当する特徴量列を取り出す
            scores = feats[:, method_idx]
            
            # 最大値を持つインデックスを取得
            best_idx = np.argmax(scores)
            
            if labels[best_idx] == 1:
                correct += 1
            total += 1
        return (correct / total) * 100

    # 従来の精度 (Min, Mean, Last)
    # extract_features の戻り値順序: 
    # [0:min, 1:mean, 2:max, 3:std, 4:last, 5:first, 6:min_last_3, 7:len, 8:sum_logits]
    acc_min = calc_baseline_acc(test_problems, 0)
    acc_mean = calc_baseline_acc(test_problems, 1)
    acc_last = calc_baseline_acc(test_problems, 4)

    # 提案手法: Learned Aggregation
    learned_correct = 0
    total = 0
    
    for p in test_problems:
        feats = np.array(p["features"])
        labels = np.array(p["labels"])
        
        if args.model_type == "logistic":
            feats = scaler.transform(feats)
        
        # モデルで「正解確率」を予測 (クラス1の確率)
        probs = model.predict_proba(feats)[:, 1]
        
        # 最も確率が高い候補を選ぶ
        best_idx = np.argmax(probs)
        
        if labels[best_idx] == 1:
            learned_correct += 1
        total += 1
        
    acc_learned = (learned_correct / total) * 100

    print("-" * 40)
    print(f"Benchmark      : {os.path.basename(os.path.normpath(args.test_dir))}")
    print(f"Total Problems : {total}")
    print("-" * 40)
    print(f"Baseline (Min) : {acc_min:.2f}%")
    print(f"Baseline (Mean): {acc_mean:.2f}%")
    print(f"Baseline (Last): {acc_last:.2f}%")
    print("-" * 40)
    print(f"Learned Agg.   : {acc_learned:.2f}%")
    print("-" * 40)

    heuristics_max = max(acc_min, acc_mean, acc_last)
    if acc_learned > heuristics_max:
        print(f"SUCCESS: Learned Aggregation improved accuracy by +{acc_learned - heuristics_max:.2f}% points!")
    else:
        print(f"Result: Learned Aggregation is comparable (Gap: {acc_learned - heuristics_max:.2f}%).")

if __name__ == "__main__":
    main()