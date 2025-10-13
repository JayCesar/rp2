# Configura o logger
import logging

import tqdm

from .shared import genai 
from .shared import util
from .shared.csv_to_dict import create_redacoes_dict
from .shared.generate_redacao_input import generate_redacao_input
from .shared.validate_response import validate_response
from .shared.extract_response_data import extract_response_data
from .shared.save_response_data import save_response_data

# Molde de Prompt
PROMPT_INPUT_MODEL = \
'''

'''

def run():

    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(message)s'
    )

    lista_redacoes = create_redacoes_dict()

    # Busca redações já avaliadas

    redacoes_avaliadas = util.json_dealer("controle/lista_avaliadas.json", 'read')

    genai_client = genai.init(api_key=)
    for redacao in tqdm(lista_redacoes, desc="Avaliando redações"):

        id = redacao['id']

        if id in redacoes_avaliadas: 
            logging.debug(f"Redação {id} - Redação já avaliada.")
            continue

        try:
            
            is_valid, response = _avalia_redacao(redacao, genai_client)

            # caso n seja válida, tratar
            if not is_valid:
                logging.warning(f"Redação {id} - Response inválido, tentando novamente.")

                is_valid, response = _avalia_redacao(redacao, genai_client)

                if is_valid:
                    logging.info(f"Redacao {id} - Response válido após nova tentativa.")
                else:
                    logging.error(f"Redação {id} - Response inválido novamente. Pulando...")
                    continue
            
            # armazenar caso seja
            response_data = extract_response_data(response)

            # Salvar id da redação avaliada
            # save()

            # armazenar resultados em um json
            save_response_data(response_data)

        except Exception as e:
            logging.error(f"Redação {id} - Falha ao processar: {e}", exc_info=True)
            continue

def _avalia_redacao(redacao, genai_client):

    redacao_input = generate_redacao_input(redacao, PROMPT_INPUT_MODEL)

    response = genai.generate(
        client=genai_client,
        input=redacao_input
    ) 
    is_valid = validate_response(response)

    return is_valid, response
