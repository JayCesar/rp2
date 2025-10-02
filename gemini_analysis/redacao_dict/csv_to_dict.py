import pandas as pd

def create_redacoes_dict(num_amostras: int = None): 

    extended_essay_br = pd.read_csv('../database/extended_essay-br.csv')
    info_redacoes = pd.read_csv('../database/prompts.csv')
    
    df_a_processar = None

    if num_amostras is not None:
        if num_amostras > len(extended_essay_br):
            num_amostras = len(extended_essay_br)
        
        print(f"Selecionando {num_amostras} redações aleatórias...")
        df_a_processar = extended_essay_br.sample(n=num_amostras, random_state=42)
    
    else:
        print(f"Selecionando o dataset completo com {len(extended_essay_br)} redações...")
        df_a_processar = extended_essay_br

    dados_estruturados = []
    for index, row in df_a_processar.iterrows():
        info = info_redacoes.loc[int(row['prompt'])]
        
        enunciado = info['description'].to_string(index=False) if not info['description'].empty else ""

        objeto_redacao = {
            'id': index,
            'titulo': row['title'],
            'texto': row['essay'],
            'nota_c1': row['c1'],
            'nota_final': row['score'],
            'enunciado_titulo': info['title'].to_string(index=False) if not info['title'].empty else "",
            'enunciado': enunciado,
            'categoria': info['category'].to_string(index=False) if not info['category'].empty else ""
        }
        dados_estruturados.append(objeto_redacao)

    return dados_estruturados