import logging, os
import pandas as pd

from typing import Dict, Optional
from ..models.Redacao import Redacao

def create_redacoes_dict(num_amostras: Optional[int] = None) -> Dict[int, Redacao]:

    try:
        extended_essay_br = pd.read_csv(os.path.join('database', 'extended_essay-br.csv'))
        info_redacoes = pd.read_csv(os.path.join('database', 'prompts.csv'))
        
        if 'id' in info_redacoes.columns:
            info_redacoes = info_redacoes.set_index('id')
            
    except FileNotFoundError as e:
        raise Exception(f"ERRO: Não foi possível carregar os arquivos de dados. Verifique o caminho: {e}")
    
    df_a_processar: pd.DataFrame
    
    if num_amostras is not None:
        if num_amostras > len(extended_essay_br):
            num_amostras = len(extended_essay_br)
        
        logging.info(f"Selecionando {num_amostras} redações aleatórias...")
        df_a_processar = extended_essay_br.sample(n=num_amostras, random_state=42)
    
    else:
        logging.info(f"Selecionando o dataset completo com {len(extended_essay_br)} redações...")
        df_a_processar = extended_essay_br

    dados_estruturados: Dict[int, Redacao] = {}
    
    def safe_str(value) -> str:
        if pd.isna(value): return ""
        return str(value).strip()

    for index, row in df_a_processar.iterrows():
        
        prompt_id = int(row['prompt']) 
        
        try:
            info = info_redacoes.loc[prompt_id]
        except KeyError:
            logging.error(f"AVISO: ID de prompt {prompt_id} não encontrado no prompts.csv. Pulando.")
            continue 
        
        enunciado = safe_str(info.get('description'))
        enunciado_titulo = safe_str(info.get('title'))
        categoria = safe_str(info.get('category'))
        
        redacao = Redacao(
            id = index, 
            titulo = safe_str(row['title']),
            texto = safe_str(row['essay']),
            nota_c1 = int(row['c1']),
            nota_final = int(row['score']), 
            enunciado_titulo = enunciado_titulo,
            enunciado = enunciado,
            categoria = categoria
        )

        dados_estruturados[index] = redacao

    return dados_estruturados