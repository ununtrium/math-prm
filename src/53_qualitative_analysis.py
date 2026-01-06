import os
import json
import glob

def extract_qualitative_samples_full_path(target_dir, threshold_drop=0.5, initial_high=0.1, top_n=20):
    """
    PRMの生スコアを用いて、スコアが急落した不正解パスの全ステップを抽出・強調表示する。
    """
    files = glob.glob(os.path.join(target_dir, "**", "seed_*.jsonl"), recursive=True)
    if not files:
        print(f"No files found in {target_dir}")
        return

    candidates = []

    for file_path in files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    item = json.loads(line)
                    problem = item.get("problem", "")
                    
                    for r in item.get("responses", []):
                        if r.get("is_correct", False):
                            continue
                        
                        steps = r.get("steps", [])
                        scores = r.get("step_scores", [])
                        
                        valid_scores = [float(s) for s in scores if s is not None]
                        if len(valid_scores) < 2:
                            continue
                        
                        # 初期スコアのしきい値チェック
                        if valid_scores[0] < initial_high:
                            continue
                        
                        # 最大の急落箇所を探す
                        max_drop = -1
                        error_idx = -1
                        
                        for i in range(1, len(valid_scores)):
                            drop = valid_scores[i-1] - valid_scores[i]
                            if drop >= threshold_drop and drop > max_drop:
                                max_drop = drop
                                error_idx = i
                        
                        if error_idx != -1:
                            candidates.append({
                                "problem": problem,
                                "steps": steps,
                                "scores": valid_scores,
                                "error_idx": error_idx,
                                "drop_val": max_drop,
                                "file": file_path
                            })
                                
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    # 下落幅が大きい順にソート
    candidates = sorted(candidates, key=lambda x: x['drop_val'], reverse=True)

    output_file = "qualitative_analysis_full_trajectories.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=== 定性分析：全ステップのスコア推移とエラー箇所の強調 ===\n\n")
        f.write(f"抽出条件: 下落幅(Logit) >= {threshold_drop}, 初期スコア >= {initial_high}\n\n")
        
        for i, c in enumerate(candidates[:top_n]):
            f.write(f"【No. {i+1}】 最大下落幅: {c['drop_val']:.3f} (Step {c['error_idx']} -> {c['error_idx']+1})\n")
            f.write(f"ファイル: {c['file']}\n")
            f.write("\n--- 問題 (Problem) ---\n")
            f.write(c['problem'] + "\n")
            f.write("\n--- 推論プロセスの全ステップ (Full Trajectory) ---\n")
            
            # 各ステップをスコアと共に表示
            for j, (step_text, score) in enumerate(zip(c['steps'], c['scores'])):
                # ミスが検知されたステップを強調
                if j == c['error_idx']:
                    f.write("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
                    f.write(f">>> [ERROR DETECTED] Step {j+1} | Score: {score:.3f} (Drop: -{c['drop_val']:.3f})\n")
                    f.write(f"TEXT: {step_text.strip()}\n")
                    f.write("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n")
                else:
                    f.write(f"Step {j+1} | Score: {score:.3f}\n")
                    f.write(f"TEXT: {step_text.strip()}\n\n")
            
            f.write("\n" + "="*80 + "\n\n")

    print(f"完了: '{output_file}' に全ステップを含むレポートを保存しました。")


def extract_logical_errors_middle_only(target_dir, threshold_drop=0.3, initial_high=0.1, top_n=20):
    files = glob.glob(os.path.join(target_dir, "**", "seed_*.jsonl"), recursive=True)
    candidates = []

    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                for r in item.get("responses", []):
                    if r.get("is_correct", False): continue
                    steps, scores = r.get("steps", []), r.get("step_scores", [])
                    valid_scores = [float(s) for s in scores if s is not None]
                    
                    if len(valid_scores) < 3: continue # 短すぎるパスは除外
                    if valid_scores[0] < initial_high: continue

                    max_drop = -1
                    error_idx = -1
                    
                    # ★修正：最後のステップ（i = len-1）でのドロップは無視する
                    # これにより、最後のループ崩壊によるノイズを除去
                    for i in range(1, len(valid_scores) - 1):
                        drop = valid_scores[i-1] - valid_scores[i]
                        if drop >= threshold_drop and drop > max_drop:
                            max_drop = drop
                            error_idx = i
                    
                    if error_idx != -1:
                        candidates.append({
                            "problem": item.get("problem", ""),
                            "steps": steps, "scores": valid_scores,
                            "error_idx": error_idx, "drop_val": max_drop,
                            "file": file_path
                        })

    # 下落幅順にソート
    candidates = sorted(candidates, key=lambda x: x['drop_val'], reverse=True)

    # ...（以下、ファイル出力処理は前回と同じ）...
    output_file = "qualitative_analysis_full_trajectories.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=== 定性分析：全ステップのスコア推移とエラー箇所の強調 ===\n\n")
        f.write(f"抽出条件: 下落幅(Logit) >= {threshold_drop}, 初期スコア >= {initial_high}\n\n")
        
        for i, c in enumerate(candidates[:top_n]):
            f.write(f"【No. {i+1}】 最大下落幅: {c['drop_val']:.3f} (Step {c['error_idx']} -> {c['error_idx']+1})\n")
            f.write(f"ファイル: {c['file']}\n")
            f.write("\n--- 問題 (Problem) ---\n")
            f.write(c['problem'] + "\n")
            f.write("\n--- 推論プロセスの全ステップ (Full Trajectory) ---\n")
            
            # 各ステップをスコアと共に表示
            for j, (step_text, score) in enumerate(zip(c['steps'], c['scores'])):
                # ミスが検知されたステップを強調
                if j == c['error_idx']:
                    f.write("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
                    f.write(f">>> [ERROR DETECTED] Step {j+1} | Score: {score:.3f} (Drop: -{c['drop_val']:.3f})\n")
                    f.write(f"TEXT: {step_text.strip()}\n")
                    f.write("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n")
                else:
                    f.write(f"Step {j+1} | Score: {score:.3f}\n")
                    f.write(f"TEXT: {step_text.strip()}\n\n")
            
            f.write("\n" + "="*80 + "\n\n")

    print(f"完了: '{output_file}' に全ステップを含むレポートを保存しました。")

if __name__ == "__main__":
    # ターゲットパスを指定
    target_path = "results/math500/Qwen2.5-Math-1.5B-Instruct/prm_1.5b_ensemble_raw_only_ckpt4054"
    extract_logical_errors_middle_only(target_path)