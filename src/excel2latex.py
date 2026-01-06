import pandas as pd
import numpy as np
import os

# --- 設定 ---
INPUT_FILE = 'integrated_results_no_trigger.xlsx' # 必要に応じて 'integrated_results.xlsx' に変更してください

# ファイル読み込み
if INPUT_FILE.endswith('.csv'):
    df = pd.read_csv(INPUT_FILE)
else:
    df = pd.read_excel(INPUT_FILE)

df = df[df['PRM'].str.contains('no_trigger', case=False, na=False)]

# --- 1. 名前の簡略化と前処理 ---
def simplify_prm(name):
    name_lower = str(name).lower()
    if 'orm' in name_lower: return 'ORM'
    elif 'ensemble' in name_lower: return 'PRM (Ensemble)'
    elif 'qwen' in name_lower: return 'PRM (Qwen)'
    elif 'deepseek' in name_lower: return 'PRM (DeepSeek)'
    elif 'llama' in name_lower: return 'PRM (Llama)'
    return name

df['PRM_simple'] = df['PRM'].apply(simplify_prm)
df['N'] = df['Samples'].str.extract('(\d+)').astype(int)

# --- 2. ソート順の定義 ---
# ベンチマークの表示順
bench_order = {'math500': 0, 'aime24': 1, 'aime25': 2}
# PRMモデルの表示順
prm_order = {
    'ORM': 0, 
    'PRM (Qwen)': 1, 
    'PRM (Llama)': 2, 
    'PRM (DeepSeek)': 3, 
    'PRM (Ensemble)': 4
}

# ソート用の列を追加
df['bench_rank'] = df['Benchmark'].map(bench_order).fillna(99)
df['prm_rank'] = df['PRM_simple'].map(prm_order).fillna(99)

# --- 3. データのピボット処理 ---
# 確実に指定順で並ぶようにソートしてからピボット
df_sorted = df.sort_values(['bench_rank', 'prm_rank'])

metrics = ['Best_Heur_Score', 'Learned_Acc']
pivot_df = df_sorted.pivot_table(index=['bench_rank', 'Benchmark', 'prm_rank', 'PRM_simple'],
                                columns='N',
                                values=metrics + ['Best_Heur_Method'],
                                aggfunc='first')

# ベースライン項目。Avg@kがない場合はMajVoteのみ
baselines_to_check = ['MajVote', 'Avg@k']
found_baselines = [b for b in baselines_to_check if b in df.columns]

maj_pivot = df_sorted.pivot_table(index=['bench_rank', 'Benchmark'],
                                 columns='N',
                                 values=found_baselines,
                                 aggfunc='first')

# --- 4. LaTeXテーブル生成 ---
def generate_latex_table(pivot_df, maj_pivot, found_baselines):
    # bench_rank でループ
    bench_indices = pivot_df.index.get_level_values('Benchmark').unique()
    # 実際には bench_rank の順に取得される
    bench_list = pivot_df.index.droplevel(['prm_rank', 'PRM_simple']).unique()

    latex = []
    latex.append(r"\begin{table*}[t]")
    latex.append(r"\centering")
    latex.append(r"\caption{各ベンチマークにおける再順位付け性能（Pass@1, \%）の比較 ($N=16 / 64$)}")
    latex.append(r"\label{tab:results_final}")
    latex.append(r"\begin{tabular}{llcc}")
    
    latex.append(r"\toprule")
    latex.append(r"Benchmark & Method / PRM Model & Best Heuristic\textsuperscript{†} & Learned (Ours) \\")
    latex.append(r"\midrule")
    
    for b_rank, b_name in bench_list:
        bench_df = pivot_df.loc[(b_rank, b_name)]
        bench_maj = maj_pivot.loc[(b_rank, b_name)]
        
        # 太字判定（MajVoteを除外）
        max_vals = {}
        for n in [16, 64]:
            candidates = []
            if 'Avg@k' in found_baselines and ('Avg@k', n) in maj_pivot.columns:
                candidates.append(bench_maj[('Avg@k', n)])
            if n in bench_df.columns.get_level_values(1):
                candidates.extend(bench_df[('Best_Heur_Score', n)].dropna().tolist())
                candidates.extend(bench_df[('Learned_Acc', n)].dropna().tolist())
            max_vals[n] = max(candidates) if candidates else -1

        first_row_bench = True
        
        # A. Baseline rows
        for b_col in found_baselines:
            b_label = 'Majority Vote' if b_col == 'MajVote' else 'Avg@k'
            scores = []
            for n in [16, 64]:
                if (b_col, n) in maj_pivot.columns:
                    val = bench_maj[(b_col, n)]
                    is_bold = (b_col != 'MajVote') and (val == max_vals[n] and val > 0)
                    s_str = f"\\textbf{{{val:.2f}}}" if is_bold else f"{val:.2f}"
                    scores.append(s_str)
                else:
                    scores.append("-")
            
            combined = " / ".join(scores)
            b_display = b_name.upper() if first_row_bench else ""
            latex.append(f"{b_display} & {b_label} (Baseline) & \\multicolumn{{2}}{{c}}{{{combined}}} \\\\")
            first_row_bench = False
        
        # B. Model rows (prm_rankの順に表示される)
        for p_rank in sorted(bench_df.index.get_level_values('prm_rank')):
            prm_name = bench_df.loc[p_rank].index[0]
            row_data = bench_df.loc[(p_rank, prm_name)]
            
            formatted_cells = {}
            for m in metrics:
                scores = []
                for n in [16, 64]:
                    if (m, n) in bench_df.columns and not pd.isna(row_data[(m, n)]):
                        val = row_data[(m, n)]
                        s_str = f"\\textbf{{{val:.2f}}}" if val == max_vals[n] and val > 0 else f"{val:.2f}"
                        if m == 'Best_Heur_Score':
                            method = row_data[('Best_Heur_Method', n)]
                            s_str += f" {{\\scriptsize ({method})}}"
                        scores.append(s_str)
                    else:
                        scores.append("-")
                formatted_cells[m] = " / ".join(scores)
            
            latex.append(f" & {prm_name} & {formatted_cells['Best_Heur_Score']} & {formatted_cells['Learned_Acc']} \\\\")
        
        # 各ベンチマークの後にライン
        latex.append(r"\midrule")
        
    latex.pop() # 最後の midrule を削除
    latex.append(r"\bottomrule")
    latex.append(r"\multicolumn{4}{l}{\footnotesize † Min, Mean, Last, Sumの中で最高精度を記録した手法とその値を記載。} \\")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table*}")
    
    return "\n".join(latex)

print(generate_latex_table(pivot_df, maj_pivot, found_baselines))