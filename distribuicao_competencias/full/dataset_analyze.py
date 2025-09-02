import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('../database/extended_essay-br.csv')

c1_counts = df['c1'].value_counts()
c2_counts = df['c2'].value_counts()
c3_counts = df['c3'].value_counts()
c4_counts = df['c4'].value_counts()
c5_counts = df['c5'].value_counts()
scores = [0, 40, 80, 120, 160, 200]
data_to_plot = pd.DataFrame({
  'Competência 1': c1_counts.reindex(scores).fillna(0),
  'Competência 2': c2_counts.reindex(scores).fillna(0),
  'Competência 3': c3_counts.reindex(scores).fillna(0),
  'Competência 4': c4_counts.reindex(scores).fillna(0),
  'Competência 5': c5_counts.reindex(scores).fillna(0)
})

x = np.arange(len(scores))
width = 0.15
num_competencias = 5

fig, ax = plt.subplots(figsize=(16, 8))

offsets = np.arange(-(num_competencias // 2), (num_competencias // 2) + 1) * width
colors = plt.get_cmap('viridis', num_competencias)

competencias = ['Competência 1', 'Competência 2', 'Competência 3', 'Competência 4', 'Competência 5']

for i, competencia in enumerate(competencias):
  ax.bar(x + offsets[i], data_to_plot[competencia], width, label=competencia, color=colors(i))

ax.set_title('Distribuição Comparativa das Notas para as 5 Competências do ENEM', fontsize=18)
ax.set_ylabel('Frequência (Quantidade de Redações)', fontsize=12)
ax.set_xlabel('Nota', fontsize=12)

ax.set_xticks(x)
ax.set_xticklabels(scores)

ax.legend(fontsize=10)

ax.yaxis.grid(True, linestyle='--', alpha=0.7)
ax.set_axisbelow(True)

fig.tight_layout()
plt.savefig('distribuicao_competencias.png')
# plt.show()