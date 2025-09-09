import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('../../database/extended_essay-br.csv')

c1_counts = df['c1'].value_counts()
scores = [0, 40, 80, 120, 160, 200]
data_to_plot = pd.DataFrame({
  'Competência 1': c1_counts.reindex(scores).fillna(0)
})

x = np.arange(len(scores))
width = 0.5
num_competencias = 1

fig, ax = plt.subplots(figsize=(16, 8))

offsets = np.arange(-(num_competencias // 2), (num_competencias // 2) + 1) * width
colors = plt.get_cmap('YlGnBu', num_competencias)

competencias = ['Competência 1']

for i, competencia in enumerate(competencias):
  bars = ax.bar(x + offsets[i], data_to_plot[competencia], width, label=competencia, color=colors(i))  
  for bar in bars:
    height = bar.get_height()
    ax.text(
      bar.get_x() + bar.get_width() / 2, height, f'{int(height)}',
      ha='center', va='bottom', fontsize=10
    )

for i, competencia in enumerate(competencias):
  ax.bar(x + offsets[i], data_to_plot[competencia], width, label=competencia, color=colors(i))

ax.set_title('Distribuição Comparativa das Notas para as Competência 1 do ENEM', fontsize=18)
ax.set_ylabel('Frequência (Quantidade de Redações)', fontsize=12)
ax.set_xlabel('Nota', fontsize=12)

ax.set_xticks(x)
ax.set_xticklabels(scores)

ax.legend(fontsize=10)

ax.yaxis.grid(True, linestyle='--', alpha=0.7)
ax.set_axisbelow(True)

fig.tight_layout()
plt.savefig('distribuicao_competencias_1.png')
# plt.show()