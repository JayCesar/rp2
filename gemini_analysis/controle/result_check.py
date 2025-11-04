import os
import pandas as pd
import matplotlib.pyplot as plt

from typing import Dict

from ..shared.json import json_dealer

CONTROLE_DIR = os.path.join("gemini_analysis", "controle")
AVALIACOES_PATH = os.path.join(CONTROLE_DIR, "avaliacoes.json")
C1_DATA_PATH = os.path.join(CONTROLE_DIR, "distribuicao_c1_data.json")


def plotar_histograma_com_valores(ax, data, bins, title, xlabel, color, xticks=None):

    n, bins_edges, patches = ax.hist(data, bins=bins, color=color, edgecolor='black')
    
    for i in range(len(n)):
        bin_center = (bins_edges[i] + bins_edges[i+1]) / 2
        
        if n[i] > 0:
            ax.text(
                bin_center,
                n[i] + (max(n) * 0.01),
                str(int(n[i])),
                ha='center',
                va='bottom',
                fontsize=7
            )

    ax.set_title(f"{title} (N={len(data)})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequência")

    if xticks is not None:
        ax.set_xticks(xticks)

def gerar_graficos_distribuicao():
    print("Iniciando carregamento e análise de dados...")

    dados_avaliacoes: Dict = json_dealer(AVALIACOES_PATH, 'read')

    if not dados_avaliacoes:
        print(f"ERRO: Não foi possível carregar dados do arquivo: {AVALIACOES_PATH}")
        return

    df = pd.DataFrame(list(dados_avaliacoes.values()))

    notas_originais_c1 = df['nota_c1']
    notas_originais_final = df['nota_final']

    notas_gemini_c1 = df['gemini_nota_c1'].dropna() 
    notas_gemini_final = df['gemini_nota_final'].dropna()

    c1_original_counts = notas_originais_c1.value_counts().sort_index()
    c1_gemini_counts = notas_gemini_c1.value_counts().sort_index()

    c1_data_export = {
        "original_c1": c1_original_counts.astype(int).to_dict(),
        "gemini_c1": c1_gemini_counts.astype(int).to_dict()
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Distribuição das Notas (Original vs. Gemini)', fontsize=16)

    plotar_histograma_com_valores(
        ax=axes[0, 0],
        data=notas_originais_c1,
        bins=[0, 40, 80, 120, 160, 200, 240],
        title="Distribuição Nota C1 Original",
        xlabel="Nota C1 (0, 40, 80, 120, 160, 200)",
        color='skyblue',
        xticks=[0, 40, 80, 120, 160, 200]
    )

    plotar_histograma_com_valores(
        ax=axes[0, 1],
        data=notas_originais_final,
        bins=26, 
        title="Distribuição Nota Final Original",
        xlabel="Nota Final (0 a 1000)",
        color='lightcoral'
    )

    plotar_histograma_com_valores(
        ax=axes[1, 0], 
        data=notas_gemini_c1, 
        bins=[0, 40, 80, 120, 160, 200, 240],
        title="Distribuição Gemini Nota C1",
        xlabel="Nota C1 Gemini (0, 40, 80, 120, 160, 200)",
        color='lightgreen',
        xticks=[0, 40, 80, 120, 160, 200]
    )

    plotar_histograma_com_valores(
        ax=axes[1, 1], 
        data=notas_gemini_final, 
        bins=26, 
        title="Distribuição Gemini Nota Final",
        xlabel="Nota Final Gemini (0 a 1000)",
        color='gold'
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    is_save = json_dealer(C1_DATA_PATH, 'write', c1_data_export)
    if is_save:
        print(f"Dados de distribuição C1 exportados com sucesso para: {C1_DATA_PATH}")
    else:
        print(f"ERRO: Falha ao exportar dados C1 para JSON.")

    output_path = os.path.join("gemini_analysis", "controle", "distribuicao_notas.png")
    plt.savefig(output_path, dpi=300)
    print(f"Geração de gráficos concluída. Salvo em: {output_path}")

if __name__ == "__main__":
    gerar_graficos_distribuicao()
