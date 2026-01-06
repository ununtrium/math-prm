import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

# 1. データの定義
data = {
    "Feature": [
        "Last (Conclusion)", "Max (Peak)", "Std (Instability)", "First (Start)",
        "Mean (Coherence)", "Min Last 3 (Late Break)", "Min (Logic Break)",
        "Length (Complexity)", "Sum Logits (Joint Prob)"
    ],
    "Importance": [
        0.576574, 0.126754, 0.099459, 0.052522,
        0.039552, 0.031591, 0.028298, 0.027138, 0.018113
    ],
    "Correlation": [
        0.357968, 0.379442, 0.232159, 0.254781,
        0.351498, 0.350483, 0.293125, -0.103081, 0.270928
    ]
}

df = pd.DataFrame(data).sort_values("Importance", ascending=False)

# 2. 色の設定 (正の相関: 青系 #4C72B0, 負の相関: 赤系 #C44E52)
df['Color'] = df['Correlation'].apply(lambda x: '#4C72B0' if x > 0 else '#C44E52')

# 3. グラフの描画
plt.figure(figsize=(12.5, 7))
ax = sns.barplot(
    data=df,
    x='Importance',
    y='Feature',
    palette=df['Color'].tolist()
)

# タイトルとラベルの設定
#plt.title('PRM Feature Analysis: Importance (Magnitude) and Correlation (Direction)', fontsize=14, pad=20)
plt.xlabel('GBDT Feature Importance (Contribution to Prediction)', fontsize=12)
plt.ylabel('Features extracted from Step Scores', fontsize=12)

# カスタム凡例の追加
legend_elements = [
    Patch(facecolor='#4C72B0', label='Positive Correlation (Higher value -> Correct)'),
    Patch(facecolor='#C44E52', label='Negative Correlation (Higher value -> Incorrect)')
]
plt.legend(handles=legend_elements, loc='lower right', frameon=True, fontsize=10)

# 棒の横に数値を表示
for i, p in enumerate(ax.patches):
    width = p.get_width()
    ax.text(width + 0.005, p.get_y() + p.get_height()/2, f'{width:.3f}', va='center', fontsize=10)

plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('prm_feature_analysis_integrated.pdf', dpi=300)