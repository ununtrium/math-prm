import json
import os
import glob
import numpy as np
import math
from collections import Counter
from tqdm import tqdm

# ==========================================
# 1. 設定パラメータ
# ==========================================
# 入力ディレクトリ (前のステップの出力先を指定)
INPUT_DIR = "data/experiments/bon_stepwise_scored_orm_v3.0_1.5b_chat" 

# ==========================================
# 2. ユーティリティ (正誤判定ロジック)
# ==========================================
# 基本的にはJSON内のフラグを使うが、念のためフォールバック用に残す
def extract_answer_content(text):
    if not text: return None
    import re
    matches = re.findall(r"\\boxed\{(.*?)\}", text)
    if matches: return matches[-1].strip()
    return None

# ==========================================
# 3. 集計戦略
# ==========================================
def agg_min(scores): return min(scores) if scores else -999.0
def agg_mean(scores): return np.mean(scores) if scores else -999.0
def agg_last(scores): return scores[-1] if scores else -999.0
def agg_max(scores): return max(scores) if scores else -999.0
def agg_sum(scores): return sum(scores) if scores else -999.0

# ==========================================
# 4. メイン集計処理
# ==========================================
def calculate_metrics_for_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # JSON構造の正規化
    if "details" in data:
        results = data["details"]
    else:
        results = data

    # カウンター
    metrics = {
        "maj_vote": 0,
        "bon_min": 0, "bon_mean": 0, "bon_last": 0, "bon_max": 0, "bon_sum": 0,
        "weighted_vote": 0, "filter_vote": 0,
    }
    
    total_problems = 0
    
    for item in results:
        # サンプルリストの取得
        if "generated_samples" in item:
            samples = item["generated_samples"]
        elif "paths" in item: 
            # 古い形式への対応 (念のため)
            # この場合 is_correct が無い可能性があるので注意が必要だが今回は割愛
            continue
        else:
            continue

        if not samples:
            total_problems += 1
            continue

        # 必要なデータの抽出
        # 前のステップで計算済みの is_correct を使う (高速化)
        is_correct_list = [s.get("is_correct", False) for s in samples]
        
        # 答えの文字列（Majority Vote用）
        # pred_answer があればそれを使う。なければ text から抽出
        answers_str = []
        for s in samples:
            if "pred_answer" in s and s["pred_answer"]:
                answers_str.append(s["pred_answer"])
            else:
                answers_str.append(extract_answer_content(s.get("text", "")))
        
        # スコアリストの抽出
        step_scores_list = [s.get("step_scores", []) for s in samples]
        
        # 有効なサンプル（答えが抽出できたもの）のインデックス
        valid_idx = [i for i, a in enumerate(answers_str) if a is not None]
        
        if not valid_idx:
            total_problems += 1
            continue

        # --- 1. Majority Vote ---
        valid_ans_str = [answers_str[i] for i in valid_idx]
        if valid_ans_str:
            maj_ans = Counter(valid_ans_str).most_common(1)[0][0]
            # 多数決で選ばれた答えを持つサンプルの正誤を確認
            # (同じ答えを持つサンプルが複数ある場合、どれか1つの正誤フラグを見れば良い)
            # ここでは厳密に判定するため、その答えを持つ最初のサンプルの正誤を採用
            for i in valid_idx:
                if answers_str[i] == maj_ans:
                    if is_correct_list[i]:
                        metrics["maj_vote"] += 1
                    break

        # --- 2. Best-of-N (Aggregation) ---
        # 各戦略でスコア計算
        s_min = [agg_min(s) for s in step_scores_list]
        s_mean = [agg_mean(s) for s in step_scores_list]
        s_last = [agg_last(s) for s in step_scores_list]
        s_max = [agg_max(s) for s in step_scores_list]
        s_sum = [agg_sum(s) for s in step_scores_list]

        # 各戦略でベストなインデックスを選び、その正誤を加算
        if is_correct_list[np.argmax(s_min)]: metrics["bon_min"] += 1
        if is_correct_list[np.argmax(s_mean)]: metrics["bon_mean"] += 1
        if is_correct_list[np.argmax(s_last)]: metrics["bon_last"] += 1
        if is_correct_list[np.argmax(s_max)]: metrics["bon_max"] += 1
        if is_correct_list[np.argmax(s_sum)]: metrics["bon_sum"] += 1
        
        # --- 改良版 Weighted Vote ---
        # パラメータ調整用
        temperature = 1.0  # 推奨: 0.5 〜 2.0 の間でグリッドサーチ
        use_score = "mean"  # "min" または "mean"。結果から見るに "min" 推奨

        votes = {}
        scores_for_weight = s_min if use_score == "min" else s_mean

        # --- 前処理: スコアの正規化（任意だが推奨） ---
        # スコアのレンジがバラバラだとexpが暴れるため、Maxスコアを引いて安定させるテクニック
        # (softmaxの計算時によくやる手法です)
        valid_scores = [scores_for_weight[i] for i in valid_idx]
        max_score_in_batch = max(valid_scores) if valid_scores else 0

        for i in valid_idx:
            ans = answers_str[i]
            raw_score = scores_for_weight[i]
            
            # 【重要】改良点
            # 1. Temperature: 値を小さくすると高スコアの重みが極端になり(BoNに近づく)、
            #    大きくすると平等になる(Majority Voteに近づく)。
            # 2. Shift: オーバーフロー防止のため最大値を引く（相対関係は変わらない）
            scaled_score = (raw_score - max_score_in_batch) / temperature
            
            weight = math.exp(scaled_score)
            
            votes[ans] = votes.get(ans, 0) + weight

        if votes:
            best_weighted_ans = max(votes, key=votes.get)
            # その答えが正解かどうか
            for i in valid_idx:
                if answers_str[i] == best_weighted_ans:
                    if is_correct_list[i]:
                        metrics["weighted_vote"] += 1
                    break
        
        # --- Filter-then-Vote ---
        # 下位N%を捨ててから多数決する手法

        # 1. (回答, スコア) のペアを作成
        candidates = []
        for i in valid_idx:
            candidates.append({
                "ans": answers_str[i],
                "score": s_min[i], # ここでもMinスコア推奨
                "is_correct": is_correct_list[i]
            })

        # 2. スコア順にソートして、上位K個だけ残す
        # 例: 生成数N=10の場合、上位5個(Top-50%)に絞るなど
        candidates.sort(key=lambda x: x["score"], reverse=True)
        top_k = len(candidates) // 1  # 上位半分
        filtered_candidates = candidates[:top_k]

        # 3. 残った候補だけで多数決 (Weighted Voteと組み合わせてもOK)
        votes = {}
        for cand in filtered_candidates:
            votes[cand["ans"]] = votes.get(cand["ans"], 0) + 1 # 単純多数決

        if votes:
            best_weighted_ans = max(votes, key=votes.get)
            # その答えが正解かどうか
            for i in valid_idx:
                if answers_str[i] == best_weighted_ans:
                    if is_correct_list[i]:
                        metrics["filter_vote"] += 1
                    break
                        

        total_problems += 1

    if total_problems == 0: return {}
    
    return {k: v / total_problems for k, v in metrics.items()}

def main():
    if not os.path.exists(INPUT_DIR):
        print(f"Directory not found: {INPUT_DIR}")
        return

    # ファイル探索
    if os.path.isfile(INPUT_DIR):
        files = [INPUT_DIR]
    else:
        files = glob.glob(os.path.join(INPUT_DIR, "*.json"))
        
    print(f"Found {len(files)} trial files. Aggregating...")
    
    history = {k: [] for k in ["maj_vote", "bon_min", "bon_mean", "bon_last", "bon_max", "bon_sum", "weighted_vote", "filter_vote"]}
    
    for fpath in tqdm(files):
        try:
            res = calculate_metrics_for_file(fpath)
            for k, v in res.items():
                if k in history: history[k].append(v)
        except Exception as e:
            print(f"Error processing {fpath}: {e}")
            
    print("\n" + "="*60)
    print(f"  NEW PRM EVALUATION (Avg of {len(files)} Trials)")
    print("="*60)
    
    if not history["maj_vote"]:
        print("No valid results found.")
        return

    def p(label, key):
        vals = history[key]
        if not vals: return
        print(f"{label:<20} | {np.mean(vals):.2%} ± {np.std(vals):.2%}")

    p("Majority Vote", "maj_vote")
    print("-" * 40)
    p("BoN (Min)", "bon_min")
    p("BoN (Mean)", "bon_mean")
    p("BoN (Last)", "bon_last")
    p("BoN (Max)", "bon_max")
    p("BoN (Sum)", "bon_sum")
    print("-" * 40)
    p("Weighted Vote", "weighted_vote")
    p("filter Vote", "filter_vote")
    print("="*60)

if __name__ == "__main__":
    main()