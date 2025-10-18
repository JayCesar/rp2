import logging
from tqdm import tqdm
import concurrent.futures
import threading

from typing import Dict, List

from .models.prompt_input import PROMPT_INPUT_MODEL
from .models.Redacao import Redacao

from .shared import genai 
from .shared import json

from .shared.csv_to_class import create_redacoes_dict
from .shared.generate_redacao_input import generate_redacao_input
from .shared.validate_response import validate_and_extract_response
from .shared.json import json_dealer


# ========================================== #
#  CONFIGURAÇÕES DE PERFORMANCE E SEGURANÇA  #
# ========================================== #
MAX_WORKERS = 50 
FILE_LOCK = threading.Lock()
# ========================================== #

def run():

    dict_redacoes: Dict[int, Redacao] = create_redacoes_dict()

    avaliacoes_redacoes: Dict = json_dealer("gemini_analysis/controle/avaliacoes.json", 'read')
    ids_avaliados: Dict[str, List[int]] = json_dealer("gemini_analysis/controle/lista_avaliadas.json", 'read')
    
    from .keys import genai_key
    genai_client = genai.init(api_key=genai_key)

    logging.info(f"Iniciando processamento paralelo com {MAX_WORKERS} threads...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        
        future_to_id = {
            executor.submit(_avalia_redacao_thread, id_redacao, redacao_obj, genai_client, avaliacoes_redacoes, ids_avaliados): id_redacao
            for id_redacao, redacao_obj in dict_redacoes.items()
        }

        for future in tqdm(
            concurrent.futures.as_completed(future_to_id),
            total=len(dict_redacoes),
            desc="Avaliando redações (Paralelo)"
        ):
            try:
                future.result() 
            except Exception as exc:
                id_redacao = future_to_id[future]
                logging.error(f"Redação {id_redacao} - A thread falhou inesperadamente: {exc}", exc_info=True)
                
    logging.info("Processamento paralelo concluído. Tempo salvo!")



def _avalia_redacao_thread(id_redacao: int, redacao_obj: Redacao, genai_client, avaliacoes_redacoes: Dict, ids_avaliados: Dict):
    """
    Executa a avaliação, lógica de retry e salvamento seguro em uma thread.
    """
    
    if id_redacao in ids_avaliados['redacoes_avaliadas']: 
        logging.debug(f"Redação {id_redacao} - Redação já avaliada (Thread Skip).")
        return 
    
    current_ids_processed = [id_redacao]

    try:
        is_valid, redacao_obj = _avalia_redacao(redacao_obj, genai_client)
        
        is_retry = False
        if not is_valid:
            logging.warning(f"Redação {id_redacao} - Response inválido, tentando novamente na thread.")

            is_valid, redacao_obj = _avalia_redacao(redacao_obj, genai_client)

            if is_valid:
                logging.info(f"Redacao {id_redacao} - Response válido após nova tentativa.")
                is_retry = True
            else:
                logging.error(f"Redação {id_redacao} - Response inválido novamente. Pulando...")
                return
        
        with FILE_LOCK:
            avaliacoes_redacoes[id_redacao] = redacao_obj.to_dict()
            
            is_response_save = json_dealer("gemini_analysis/controle/avaliacoes.json", "write", avaliacoes_redacoes)
            
            if not is_response_save:
                logging.error(f"Redação {id_redacao} - Response não pode ser salvo. Perda de dados potencial.")
                return

            ids_avaliados["redacoes_avaliadas"].extend(current_ids_processed)
            json_dealer("gemini_analysis/controle/lista_avaliadas.json", "write", ids_avaliados)
             
        if not is_retry: logging.info(f"Redação {id_redacao} - Avaliada com sucesso!")

    except Exception as e:
        logging.error(f"Redação {id_redacao} - Falha geral ao processar na thread: {e}", exc_info=True)


def _avalia_redacao(redacao: Redacao, genai_client):

    redacao_input = generate_redacao_input(redacao, PROMPT_INPUT_MODEL)

    response = genai.generate(
        client=genai_client,
        input=redacao_input
    ) 
    is_valid, redacao = validate_and_extract_response(response, redacao)

    return is_valid, redacao
