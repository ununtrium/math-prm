import json
import argparse
import math
import numpy as np
import os
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import roc_auc_score

def load_jsonl(filepath):
    data = {}
    print(f"Loading {filepath}...")
    with open(filepath, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            try:
                rec = json.loads(line)
                # キー: (source_id, path_id, step_index)
                key = (rec["source_id"], rec["path_id"], rec["step_index"])
                data[key] = rec
            except:
                continue
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen_file", type=str, required=True, help="Path to Qwen-7B annotation")
    parser.add_argument("--deepseek_file", type=str, required=True, help="Path to DeepSeek-7B annotation")
    parser.add_argument("--llama_file", type=str, required=True, help="Path to Llama-8B annotation")
    parser.add_argument("--output_file", type=str, required=True, help="Output merged jsonl file")
    args = parser.parse_args()

    # 1. データ読み込み
    data_qwen = load_jsonl(args.qwen_file)
    data_ds = load_jsonl(args.deepseek_file)
    data_llama = load_jsonl(args.llama_file)

    # 共通のキーを取得
    common_keys = set(data_qwen.keys()) & set(data_ds.keys()) & set(data_llama.keys())
    print(f"\nTotal common steps: {len(common_keys)}")

    # 2. 分析用リスト
    qwen_probs = []
    ds_probs = []
    llama_probs = []
    labels = [] # 正解パスなら1, 不正解パスなら0

    merged_data = []

    print("Merging and Analyzing...")
    for key in tqdm(common_keys):
        rec_q = data_qwen[key]
        rec_d = data_ds[key]
        rec_l = data_llama[key]

        # --- A. 分析用データ収集 ---
        p_q = rec_q.get("raw_prob", 0.0)
        p_d = rec_d.get("raw_prob", 0.0)
        p_l = rec_l.get("raw_prob", 0.0)
        
        qwen_probs.append(p_q)
        ds_probs.append(p_d)
        llama_probs.append(p_l)
        
        # is_outcome_correct はパス単位で同じはずなのでQwenのものを使用
        labels.append(1 if rec_q["is_outcome_correct"] else 0)

        # --- B. アンサンブル計算 ---
        # Raw Prob の平均をとる
        avg_raw_prob = (p_q + p_d + p_l) / 3.0
        
        # Log Prob 逆算
        avg_log_prob = math.log(avg_raw_prob) if avg_raw_prob > 1e-9 else -20.0
        
        # レコード作成
        new_rec = rec_q.copy() # ベースはQwen
        new_rec["model_id"] = "ensemble_3models"
        new_rec["raw_prob"] = avg_raw_prob
        new_rec["log_prob"] = avg_log_prob
        
        # Deltaの計算について: 
        # 本来は「前のステップのアンサンブル値」との差分をとるべきですが、
        # 簡易的に「各モデルのDeltaの平均」でも十分機能します。
        d_q = rec_q.get("raw_prob_delta", 0.0)
        d_d = rec_d.get("raw_prob_delta", 0.0)
        d_l = rec_l.get("raw_prob_delta", 0.0)
        
        new_rec["raw_prob_delta"] = (d_q + d_d + d_l) / 3.0
        
        # Log Delta も同様に平均化（もしくは再計算）
        ld_q = rec_q.get("log_prob_delta", 0.0)
        ld_d = rec_d.get("log_prob_delta", 0.0)
        ld_l = rec_l.get("log_prob_delta", 0.0)
        new_rec["log_prob_delta"] = (ld_q + ld_d + ld_l) / 3.0

        merged_data.append(new_rec)

    # 3. 分析結果の出力 (論文にそのまま使える数値)
    print("\n" + "="*50)
    print("ANALYSIS RESULT")
    print("="*50)

    # 相関係数 (Correlation)
    corr_qd = np.corrcoef(qwen_probs, ds_probs)[0, 1]
    corr_ql = np.corrcoef(qwen_probs, llama_probs)[0, 1]
    corr_dl = np.corrcoef(ds_probs, llama_probs)[0, 1]
    
    print(f"Correlation (Qwen vs DeepSeek): {corr_qd:.4f}")
    print(f"Correlation (Qwen vs Llama):    {corr_ql:.4f}")
    print(f"Correlation (DeepSeek vs Llama): {corr_dl:.4f}")

    # AUCスコア (正解/不正解の識別能力)
    # 値が高いほど、そのモデルが「正しいステップ」を高く評価できている
    try:
        auc_qwen = roc_auc_score(labels, qwen_probs)
        auc_ds = roc_auc_score(labels, ds_probs)
        auc_llama = roc_auc_score(labels, llama_probs)
        
        # アンサンブル後のスコア
        ensemble_probs = [(q+d+l)/3 for q,d,l in zip(qwen_probs, ds_probs, llama_probs)]
        auc_ensemble = roc_auc_score(labels, ensemble_probs)

        print("-" * 30)
        print(f"AUC (Qwen):      {auc_qwen:.4f}")
        print(f"AUC (DeepSeek):  {auc_ds:.4f}")
        print(f"AUC (Llama):     {auc_llama:.4f}")
        print(f"AUC (Ensemble):  {auc_ensemble:.4f}  <-- Check this improvement!")
    except Exception as e:
        print(f"AUC Calculation Failed: {e}")

    print("="*50)

    # 4. 保存
    print(f"Saving merged data to {args.output_file}...")
    # ソートして保存 (学習の効率化のため)
    merged_data.sort(key=lambda x: (x["source_id"], x["path_id"], x["step_index"]))
    
    with open(args.output_file, "w", encoding="utf-8") as f:
        for item in merged_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print("Done.")

if __name__ == "__main__":
    main()