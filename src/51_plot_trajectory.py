import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt

# 論文用にフォントサイズを調整
plt.rcParams.update({'font.size': 14})

def load_and_normalize_path_centric(target_dir, n_bins=10):
    """
    改善案①：パス（response）ごとに正規化されたベクトルを作成し、パス単位で集計する。
    これにより、ステップ数の多いパスが平均を支配するバイアスを排除する（マクロ平均）。
    """
    files = glob.glob(os.path.join(target_dir, "**", "seed_*.jsonl"), recursive=True)
    if not files:
        return None, None

    correct_trajectories = []
    incorrect_trajectories = []
    
    print(f"Processing: {target_dir}")

    for file_path in files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    item = json.loads(line)
                    responses = item.get("responses", [])
                    for r in responses:
                        step_scores = r.get("step_scores", [])
                        if not step_scores: continue
                        
                        # 数値化とNone除外
                        valid_scores = [float(s) for s in step_scores if s is not None]
                        L = len(valid_scores)
                        if L == 0: continue

                        # --- 手順1: このレスポンス内でのビン割り当て ---
                        local_bins = [[] for _ in range(n_bins)]
                        for i, score in enumerate(valid_scores):
                            progress = i / (L - 1) if L > 1 else 1.0
                            bin_idx = int(progress * (n_bins - 1))
                            bin_idx = max(0, min(bin_idx, n_bins - 1))
                            local_bins[bin_idx].append(score)
                        
                        # --- 手順2: レスポンス内の各ビンを平均し、パス・ベクトルを作成 ---
                        # そのビンにデータがない場合は NaN とし、後の集計で無視する
                        path_vector = [np.mean(b) if b else np.nan for b in local_bins]
                        
                        is_correct = r.get("is_correct", False)
                        if is_correct:
                            correct_trajectories.append(path_vector)
                        else:
                            incorrect_trajectories.append(path_vector)
                            
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    # (N_responses, n_bins) の形状の numpy 配列として返す
    return np.array(correct_trajectories), np.array(incorrect_trajectories)

def aggregate_trajectories(trajectories):
    """
    各ビン（列）ごとにパス間の平均、標準偏差、標準誤差を計算。
    """
    if trajectories is None or len(trajectories) == 0:
        return None, None, None, None

    # nanmean/nanstd を使い、データが存在するパスのみで集計
    means = np.nanmean(trajectories, axis=0)
    stds = np.nanstd(trajectories, axis=0)
    
    # 各ビンにおける有効なパス数（サンプル数）をカウント
    counts = np.sum(~np.isnan(trajectories), axis=0)
    # サンプル数が 10 未満の地点は信頼性が低いため NaN にする
    means[counts < 10] = np.nan
    
    # 標準誤差 (SEM)
    sems = np.where(counts >= 10, stds / np.sqrt(counts), np.nan)
    
    indices = np.linspace(0, 1, len(means))
    return indices, means, sems, stds

def plot_normalized_trajectories(c_trajs, i_trajs, output_file):
    """
    集計結果をプロット。SD/SEMの帯、右上凡例、NaNマスク処理を含む。
    """
    c_x, c_mean, c_sem, c_std = aggregate_trajectories(c_trajs)
    i_x, i_mean, i_sem, i_std = aggregate_trajectories(i_trajs)

    plt.figure(figsize=(10, 6))
    
    def draw_group(x, mean, sem, std, base_label, color, linestyle='-'):
        if mean is None: return
        mask = ~np.isnan(mean)
        if not np.any(mask): return
        mx, mm, ms, mt = x[mask], mean[mask], sem[mask], std[mask]
        
        # --- 修正ポイント：SDの帯にlabelを明示的に付与 ---
        plt.fill_between(mx, mm - mt, mm + mt, color=color, alpha=0.07, label=f'{base_label} (SD)')
        
        # SEMの帯（こちらは凡例が混雑するためラベルなしのまま）
        plt.fill_between(mx, mm - ms, mm + ms, color=color, alpha=0.18)
        
        # 平均線
        plt.plot(mx, mm, label=f'{base_label} (Mean)', color=color, linewidth=2.5, linestyle=linestyle)

    # 描画
    draw_group(c_x, c_mean, c_sem, c_std, 'Correct', 'blue', '-')
    draw_group(i_x, i_mean, i_sem, i_std, 'Incorrect', 'red', '--')

    # --- グラフの体裁 ---
    plt.xlabel('Reasoning Progress (%)')
    plt.ylabel('Average PRM Score')
    plt.xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], ['0', '20', '40', '60', '80', '100'])
    plt.ylim(0.0, 1.05)
    
    # --- 修正ポイント：凡例の位置を「右上」に固定し、全てのラベルを表示 ---
    # loc='upper right' で右上に配置します
    plt.legend(loc='upper right', fontsize=11, frameon=True, shadow=False)
    
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Plot saved with Legend: {output_file}")

def main():
    base_dir = "results"
    output_base = "plots"
    os.makedirs(output_base, exist_ok=True)

    if not os.path.exists(base_dir):
        print(f"Error: '{base_dir}' directory not found.")
        return

    # results/{benchmark}/{gen_model}/{teacher_model}/
    for benchmark in os.listdir(base_dir):
        bench_path = os.path.join(base_dir, benchmark)
        if not os.path.isdir(bench_path): continue
        
        for gen_model in os.listdir(bench_path):
            if "Instruct" not in gen_model and "instruct" not in gen_model:
                continue
            gen_model_path = os.path.join(bench_path, gen_model)
            
            for teacher_model in os.listdir(gen_model_path):
                target_dir = os.path.join(gen_model_path, teacher_model)
                if not os.path.isdir(target_dir): continue

                # ファイル名の生成
                safe_teacher_name = teacher_model.replace("/", "_").replace("\\", "_")
                output_filename = f"traj_{benchmark}_{safe_teacher_name}.pdf"
                output_file_path = os.path.join(output_base, output_filename)

                # パス単位での正規化・ロード
                c_trajs, i_trajs = load_and_normalize_path_centric(target_dir, n_bins=10)
                
                # プロット実行
                if (c_trajs is not None and len(c_trajs) > 0) or \
                   (i_trajs is not None and len(i_trajs) > 0):
                    plot_normalized_trajectories(c_trajs, i_trajs, output_file_path)

if __name__ == "__main__":
    main()