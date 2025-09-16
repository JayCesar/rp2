import numpy as np
import pandas as pd

def create_redacoes_dict(num_amostras=6557):

    extended_essay_br = pd.read_csv('../database/extended_essay-br.csv')
    info_redacoes = pd.read_csv('../database/prompts.csv')

    if num_amostras > len(extended_essay_br):
        num_amostras = len(extended_essay_br)
        print(f"Aviso: O número de amostras pedido é maior que o dataset. Usando o tamanho máximo: {num_amostras}")

    print(f"Selecionando {num_amostras} redações aleatórias do total de {len(extended_essay_br)}...")
    
    indices_disponiveis = extended_essay_br.index.to_numpy()
    indices_amostra = np.random.choice(indices_disponiveis, size=num_amostras, replace=False)
    df_amostra = extended_essay_br.loc[indices_amostra]

    dados_estruturados = []
    for index, row in df_amostra.iterrows():
        info = info_redacoes.loc[int(row['prompt'])]
        enunciado = "".join(info['description'])
        objeto_redacao = {
            'id': index,
            'titulo': row['title'],
            'texto': row['essay'],
            'nota_c1': row['c1'],
            'nota_final': row['score'],
            'enunciado_titulo': info['title'],
            'enunciado': enunciado,
            'categoria': info['category']
        }
        dados_estruturados.append(objeto_redacao)

    return dados_estruturados