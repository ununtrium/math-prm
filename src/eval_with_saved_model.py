import os
import json
import glob
import argparse
import joblib
import numpy as np
import pandas as pd
from scipy.special import logit

# ★特徴量抽出関数 (ユーザー提示のものを維持)
def extract_features(step_scores):
    if not step_scores: return [0.0] * 9
    scores = np.array([float(s) for s in step_scores if s is not None])
    if len(scores) == 0: return [0.0] * 9

    # ロジット変換 (学習時と同じロジック: ここではそのまま代入)
    logits = scores

    _min = np.min(logits)
    _mean = np.mean(logits)
    _max = np.max(logits)
    _std = np.std(logits)
    _last = logits[-1]
    _first = logits[0]
    _min_last_3 = np.min(logits[-3:]) if len(logits) >= 3 else _min
    _len = float(len(logits))
    _sum_logits = np.sum(logits)

    features = [_min, _mean, _max, _std, _last, _first, _min_last_3, _len, _sum_logits]
    return np.nan_to_num(features, nan=0.0, posinf=1e5, neginf=-1e5).tolist()

def evaluate_prm(test_dir, model_path):
    print(f"Evaluating: {test_dir}")
    print(f"Using Aggregator: {model_path}")

    # モデルロード
    try:
        model = joblib.load(model_path)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None

    # テストデータロード
    files = glob.glob(os.path.join(test_dir, "**", "seed_*.jsonl"), recursive=True)
    problems = []

    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                responses = item.get("responses", [])
                if not responses: continue
                
                cand_features = []
                cand_labels = []
                
                for r in responses:
                    step_scores = r.get("step_scores", [])
                    if not step_scores: continue
                    
                    feat = extract_features(step_scores)
                    label = 1 if r.get("is_correct", False) else 0
                    
                    cand_features.append(feat)
                    cand_labels.append(label)
                
                if cand_features:
                    problems.append({"features": cand_features, "labels": cand_labels})

    if not problems:
        print("No test problems found.")
        return None

    # 評価
    correct_base_mean = 0
    correct_learned = 0
    total = 0

    for p in problems:
        feats = np.array(p["features"])
        labels = np.array(p["labels"])

        # Baseline: Mean (index 1) ※ロジットのMean
        scores_mean = feats[:, 1]
        if labels[np.argmax(scores_mean)] == 1:
            correct_base_mean += 1
            
        # Learned Aggregation
        # predict_probaの2列目(クラス1の確率)を使用
        scores_learned = model.predict_proba(feats)[:, 1]
        if labels[np.argmax(scores_learned)] == 1:
            correct_learned += 1
            
        total += 1

    acc_mean = (correct_base_mean / total) * 100
    acc_learned = (correct_learned / total) * 100

    return {"Baseline(Mean)": acc_mean, "Learned": acc_learned, "Total": total}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_base_dir", type=str, required=True, help="e.g. results/aime24")
    parser.add_argument("--models_dir", type=str, default="models/learned_aggregators")
    args = parser.parse_args()

    # 1. 保存済みモデル (.pkl) を全検索
    model_files = glob.glob(os.path.join(args.models_dir, "*.pkl"))
    if not model_files:
        print(f"No models found in {args.models_dir}")
        return

    # 2. テスト対象のGeneratorディレクトリを全検索
    # 構造: results/aime24/{GeneratorName}/...
    # test_base_dir 直下のサブディレクトリを全て取得
    try:
        test_generators = [
            d for d in os.listdir(args.test_base_dir) 
            if os.path.isdir(os.path.join(args.test_base_dir, d))
        ]
    except FileNotFoundError:
        print(f"Test directory not found: {args.test_base_dir}")
        return

    results = []

    print(f"Found {len(model_files)} aggregators and {len(test_generators)} generators in test dir.")

    for model_file in model_files:
        model_filename = os.path.basename(model_file).replace(".pkl", "")
        
        # モデル名から PRM名 を抽出
        # 想定形式: TrainGenName__PrmName.pkl
        try:
            # splitの回数を1回に制限して、PRM名にアンダースコアが含まれていても対応
            train_gen_name, prm_name = model_filename.split("__", 1)
        except ValueError:
            print(f"Skipping unknown filename format: {model_filename}")
            continue

        # ★変更点: 全てのテスト用Generatorに対して、このPRMフォルダがあるか確認して回る
        for test_gen_name in test_generators:
            
            # 探しに行くパス: results/aime24/{TestGenName}/{PrmName}
            target_test_dir = os.path.join(args.test_base_dir, test_gen_name, prm_name)
            
            # PRMフォルダが存在するか確認
            if not os.path.exists(target_test_dir):
                # このGeneratorには、このPRMで採点したデータが無いのでスキップ
                continue
            
            # samples_* ディレクトリを全て取得
            potential_dirs = glob.glob(os.path.join(target_test_dir, "samples_*"))
            
            for test_dir in potential_dirs:
                samples_dir_name = os.path.basename(test_dir)
                
                # 評価実行
                res = evaluate_prm(test_dir, model_file)
                
                if res:
                    # 結果にメタデータを追加
                    res["Test_Generator"] = test_gen_name     # 実際にテストしたモデル (例: Qwen-Base)
                    res["Aggregator_Train_Gen"] = train_gen_name # 集計器を学習したモデル (例: Qwen-Instruct)
                    res["PRM"] = prm_name                     # 使用したPRM
                    res["Samples"] = samples_dir_name         # samples_16 etc.
                    results.append(res)

    if results:
        df = pd.DataFrame(results)
        
        # カラムの並び順を整理
        cols = ["Test_Generator", "Samples", "PRM", "Baseline(Mean)", "Learned", "Aggregator_Train_Gen", "Total"]
        # 上記以外のカラムがあれば後ろに追加
        cols += [c for c in df.columns if c not in cols]
        df = df[cols]
        
        # ソート: テスト対象 -> サンプル数 -> PRM
        df = df.sort_values(by=["Test_Generator", "Samples", "PRM"])

        print("\n=== Evaluation Results ===")
        print(df.to_string(index=False))
        df.to_csv("final_evaluation_results_cross.csv", index=False)
    else:
        print("No matching test data found.")

if __name__ == "__main__":
    main()