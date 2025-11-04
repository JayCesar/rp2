import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, cohen_kappa_score, confusion_matrix

from ...shared.json import json_dealer

RESULT_DIR_NAME = "teste-1"

avaliacoes = json_dealer(f"gemini_analysis/results/{RESULT_DIR_NAME}/avaliacoes.json", 'read')
df_avaliacoes = pd.DataFrame(list(avaliacoes.values()))

df_comparacao = df_avaliacoes.dropna(subset=['nota_c1', 'gemini_nota_c1'])

if df_comparacao.empty:
    print("Nenhum dado par a par foi encontrado para a análise.")

else:

    y_true = df_comparacao['nota_c1'].astype(int).tolist()
    y_pred = df_comparacao['gemini_nota_c1'].astype(int).tolist()

    # =======================================================
    # MÉTRICAS DE AVALIAÇÃO
    # =======================================================

    # 1. RMSE
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"RMSE: {rmse:.2f} pontos")

    # 2. QWK
    qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    print(f"QWK (Quadratic Weighted Kappa): {qwk:.4f}")

    payload = {
        "RSME": f'{rmse:.2f}',
        "QWK": f'{qwk:.4f}'
    }
    json_dealer(f"gemini_analysis/results/{RESULT_DIR_NAME}/statistics.json", 'write', payload)

    # =======================================================
    # MATRIZ DE CONFUSÃO E VISUALIZAÇÃO
    # =======================================================

    labels = [0, 40, 80, 120, 160, 200]
    matriz = confusion_matrix(y_true, y_pred, labels=labels)
    df_matriz = pd.DataFrame(matriz, 
                             index=[f"Humano: {l}" for l in labels], 
                             columns=[f"Gemini: {l}" for l in labels])

    plt.figure(figsize=(10, 8))
    sns.heatmap(df_matriz, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusão - Correção Gemini vs. Humano (C1)')
    plt.ylabel('Nota Real (Humano)')
    plt.xlabel('Nota Prevista (Gemini)')

    plt.savefig(f'gemini_analysis/results/{RESULT_DIR_NAME}/matriz_confusao.png')
    print("\nMatriz de confusão salva em 'matriz_confusao.png'")
