import pandas as pd

# データの読み込み
file_name = 'integrated_results_no_trigger.xlsx'
df = pd.read_excel(file_name)

# モデル名の分類関数
def get_model_name(prm_str):
    if 'orm' in prm_str: return 'ORM'
    if 'deepseek' in prm_str: return 'PRM (DeepSeek)'
    if 'ensemble' in prm_str: return 'PRM (Ensemble)'
    if 'llama' in prm_str: return 'PRM (Llama)'
    if 'qwen' in prm_str: return 'PRM (Qwen)'
    return 'Other'

df['Model'] = df['PRM'].apply(get_model_name)
# 条件名の変更：読者が迷わないよう「Forced Prompt」に統一
df['Condition'] = df['PRM'].apply(
    lambda x: 'w/o Forced Prompt' if 'no_trigger' in x else 'w/ Forced Prompt'
)
df['Benchmark'] = df['Benchmark'].str.upper()

# PRM 4モデルのみを対象
prm_models = ['PRM (DeepSeek)', 'PRM (Ensemble)', 'PRM (Llama)', 'PRM (Qwen)']
df_prm = df[df['Model'].isin(prm_models)].copy()

# ベスト設定の抽出 (N=16 or 64)
best_n = df_prm.groupby(['Benchmark', 'Model', 'Condition'])['Learned_Acc'].max().unstack()

summary = []
target_benchmarks = ['MATH500', 'AIME24', 'AIME25']
for bench in target_benchmarks:
    row_data = best_n.loc[bench]
    summary.append({
        'Benchmark': bench,
        'Best w/ Forced': row_data['w/ Forced Prompt'].max(),
        'Best w/o Forced': row_data['w/o Forced Prompt'].max(),
        'Avg w/ Forced': row_data['w/ Forced Prompt'].mean(),
        'Avg w/o Forced': row_data['w/o Forced Prompt'].mean()
    })

results_df = pd.DataFrame(summary)

# LaTeX 表形式での出力
print("\\begin{table}[t]")
print("\\centering")
print("\\caption{解答強制（Prompted Forcing）の有無によるアブレーション研究（Success Rate, \\%）. $N \\in \\{16, 64\\}$ のうち高い方の値を採用し、PRM 4モデル（DeepSeek, Ensemble, Llama, Qwen）に基づき算出.}")
print("\\label{tab:ablation_forcing}")
print("\\begin{tabular}{llccc}")
print("\\hline")
print("Benchmark & Metric & w/o Forcing & w/ Forcing & $\\Delta$ \\\\ \\hline")

for idx, row in results_df.iterrows():
    bench_display = row['Benchmark'].replace('AIME24', 'AIME 2024').replace('AIME25', 'AIME 2025')
    
    # Best Model 行
    b_trig = row['Best w/ Forced']
    b_no = row['Best w/o Forced']
    b_diff = b_trig - b_no
    print(f"\\multirow{{2}}{{*}}{{{bench_display}}} & Best & {b_no:0.2f} & \\textbf{{{b_trig:0.2f}}} & +{b_diff:0.2f} \\\\")
    
    # Grand Avg 行
    a_trig = row['Avg w/ Forced']
    a_no = row['Avg w/o Forced']
    a_diff = a_trig - a_no
    diff_symbol = "+" if a_diff > 0 else ""
    print(f" & Avg & {a_no:0.2f} & \\textbf{{{a_trig:0.2f}}} & {diff_symbol}{a_diff:0.2f} \\\\ \\hline")

print("\\end{tabular}")
print("\\end{table}")