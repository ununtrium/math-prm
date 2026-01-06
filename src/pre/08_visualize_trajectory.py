import json
import matplotlib.pyplot as plt
import numpy as np
import os

INPUT_FILE = "data/math500_results_full_scores.json"
OUTPUT_IMG = "data/score_trajectory.png"

def main():
    if not os.path.exists(INPUT_FILE):
        print("File not found.")
        return

    with open(INPUT_FILE, "r") as f:
        data = json.load(f)

    # 不正解パスのスコア推移を集める
    trajectories = []
    
    for item in data:
        # 正解を抽出
        import re
        def ext(t):
            m = re.findall(r"\\boxed\{(.*?)\}", t)
            return m[-1].strip() if m else ""
            
        gold = item["gold"]
        paths = item["paths"]
        scores_list = item["step_scores"] # List of List
        
        for path, scores in zip(paths, scores_list):
            if not scores or scores[0] == -99.0: continue
            
            # 答えが間違っているパスだけを抽出
            pred = ext(path)
            if pred != gold:
                # スコアを正規化せずにそのまま保存
                trajectories.append(scores)
                
        if len(trajectories) > 100: break # 100本あれば十分

    # プロット
    plt.figure(figsize=(12, 6))
    
    for scores in trajectories[:20]: # 最初の20本だけ描画して見やすく
        x = range(len(scores))
        plt.plot(x, scores, marker='o', alpha=0.6)

    plt.title("Score Trajectory of INCORRECT Paths")
    plt.xlabel("Step Number")
    plt.ylabel("PRM Score")
    plt.grid(True)
    plt.axhline(0, color='black', linestyle='--')
    
    plt.savefig(OUTPUT_IMG)
    print(f"Saved trajectory plot to {OUTPUT_IMG}")

if __name__ == "__main__":
    main()