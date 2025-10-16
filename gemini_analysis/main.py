# Configura o logger
import logging

import tqdm

from .models.prompt_input import PROMPT_INPUT_MODEL

from .shared import genai 
from .shared import util

from .shared.csv_to_dict import create_redacoes_dict
from .shared.generate_redacao_input import generate_redacao_input
from .shared.validate_response import validate_and_extract_response
from .shared.save_id import save_id, save_response_data

def run():

    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(message)s'
    )

    # Converte o csv do essay-br em Dict
    lista_redacoes = create_redacoes_dict()

    # Busca avaliacoes das redacoes e ids avaliados
    avaliacoes_redacoes = util.json_dealer("controle/avaliacoes.json", 'read')
    ids_avaliados = util.json_dealer("controle/lista_avaliadas.json", 'read')

    # Inicializa o cliente genai
    from .keys import genai_key
    genai_client = genai.init(api_key=genai_key)

    # Percorre redacoes a serem avaliadas
    write_counter = 0
    current_ids = []
    for redacao in tqdm(lista_redacoes, desc="Avaliando redações"):

        id = redacao['id']
        current_ids.append(id)

        if id in ids_avaliados['redacoes_avaliadas']: 
            # logging.debug(f"Redação {id} - Redação já avaliada.")
            continue

        try:
            
            is_valid, extracted_data = _avalia_redacao(redacao, genai_client)

            is_retry = False
            if not is_valid:
                logging.warning(f"Redação {id} - Response inválido, tentando novamente.")

                is_valid, extracted_data = _avalia_redacao(redacao, genai_client)

                if is_valid:
                    logging.info(f"Redacao {id} - Response válido após nova tentativa.")
                    is_retry = True
                else:
                    logging.error(f"Redação {id} - Response inválido novamente. Pulando...")
                    current_ids.pop()
                    continue
            
            avaliacoes_redacoes[id] = extracted_data

            write_counter+=1
            if write_counter >= 20:
                write_counter = 0
                is_response_save = save_response_data(avaliacoes_redacoes, id, extracted_data)
                if not is_response_save:
                    logging.error(f"Redação {id} - Response não pode ser salvo. Pulando...\nids perdidos: {current_ids}")
                    current_ids = []
                    continue
                else:    
                    is_id_save = save_id(ids_avaliados, id)
                    if not is_id_save:
                        for id in current_ids: del avaliacoes_redacoes[id]
                        util.json_dealer("controle/avaliacoes.json", 'write', avaliacoes_redacoes)
                        logging.error(f"Redação {id} - ID não pode ser salvo. Pulando...\nids perdidos: {current_ids}")
                        current_ids = []
                        continue
                current_ids = []

            if not is_retry: logging.info(f"Redação {id} - Avaliada com sucesso!")


        except Exception as e:
            logging.error(f"Redação {id} - Falha ao processar: {e}", exc_info=True)
            continue

def _avalia_redacao(redacao, genai_client):

    redacao_input = generate_redacao_input(redacao, PROMPT_INPUT_MODEL)

    response = genai.generate(
        client=genai_client,
        input=redacao_input
    ) 
    is_valid, extracted_data = validate_and_extract_response(response)

    return is_valid, extracted_data
