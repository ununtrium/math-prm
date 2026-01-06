import json
import argparse
import math
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

def ascii_histogram(values, bins=10, title="Histogram"):
    """簡易的なテキストヒストグラムを表示"""
    if not values: return
    counts, bin_edges = np.histogram(values, bins=bins, range=(min(values), max(values)))
    max_count = max(counts) if max(counts) > 0 else 1
    print(f"\n[{title}]")
    print(f"Range: {min(values):.4f} ~ {max(values):.4f}")
    for i, count in enumerate(counts):
        bar = "#" * int(20 * count / max_count)
        range_str = f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}"
        print(f"{range_str:>12}: {bar} ({count})")

def main():
    parser = argparse.ArgumentParser()
    # マージ済みのファイルを指定してください
    parser.add_argument("--input_file", type=str, default="data/prm_train_ensemble_2M.jsonl")
    args = parser.parse_args()

    print(f"Loading {args.input_file}...")
    
    raw_probs = []
    deltas = []
    labels = []
    
    # データをロードしてメモリに展開
    with open(args.input_file, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            try:
                rec = json.loads(line)
                
                # アンサンブル済みの Raw Prob
                p = rec.get("raw_prob", 0.0)
                
                # アンサンブル済みの Delta (なければ計算、あるいは0)
                # 注: マージ時に計算していない場合はここで簡易計算できません(前ステップが必要)
                # マージスクリプトで raw_prob_delta を保存している前提です
                d = rec.get("raw_prob_delta", 0.0) # 生確率の差分
                # もし対数デルタを使うなら: d = math.tanh(rec.get("log_prob_delta", 0.0))
                
                # 対数デルタのTanhを使う方針の場合
                # log_delta があればそれを使う。なければ raw_prob から擬似的に計算は...難しい
                # ここでは「マージデータに log_prob_delta がある」と仮定して tanh します
                ld = rec.get("log_prob_delta", 0.0)
                tanh_d = math.tanh(ld)
                
                raw_probs.append(p)
                deltas.append(tanh_d)
                labels.append(1 if rec["is_outcome_correct"] else 0)
            except:
                continue

    # 1. 分布の確認
    print("\n" + "="*60)
    print("1. DISTRIBUTION CHECK")
    print("="*60)
    ascii_histogram(raw_probs, title="Raw Probability (Main Term)")
    ascii_histogram(deltas, title="Tanh(Log Delta) (Aux Term)")
    
    # 統計量
    print(f"\nStats:")
    print(f"Raw Prob: Mean={np.mean(raw_probs):.4f}, Std={np.std(raw_probs):.4f}")
    print(f"Tanh Delta: Mean={np.mean(deltas):.4f}, Std={np.std(deltas):.4f}")
    
    # 2. スコアリングの比較 (AUC)
    print("\n" + "="*60)
    print("2. SCORING SIMULATION (AUC Comparison)")
    print("="*60)
    print(f"{'Formula':<40} | {'AUC':<8} | {'Note'}")
    print("-" * 65)
    
    # 比較する係数の候補
    candidates = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
    
    best_auc = -1
    best_w = 0.0
    
    for w in candidates:
        # Score = RawProb + w * Tanh(LogDelta)
        # ※ 0.0 ~ 1.0 の範囲外に出ることもありますが、順序(AUC)には影響しません
        scores = [p + w * d for p, d in zip(raw_probs, deltas)]
        
        try:
            auc = roc_auc_score(labels, scores)
            print(f"Prob + {w:.2f} * Tanh(Delta) {' ':<16} | {auc:.4f}   |")
            
            if auc > best_auc:
                best_auc = auc
                best_w = w
        except:
            print(f"Prob + {w:.2f} * ... Error")

    print("-" * 65)
    print(f"Best Weight seems to be: {best_w} (AUC: {best_auc:.4f})")
    print("\n[Advice]")
    if best_w == 0.0:
        print("-> Delta doesn't help. Use 'aux_weight = 0.0' (Raw Prob only).")
    elif best_auc - roc_auc_score(labels, raw_probs) < 0.001:
        print("-> Delta impact is minimal. Stick to 0.0 or 0.1 for stability.")
    else:
        print(f"-> Delta improves AUC! Use 'aux_weight = {best_w}'.")

if __name__ == "__main__":
    main()