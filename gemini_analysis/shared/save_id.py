from typing import List

from .util import json_dealer

def save_id(ids_avaliados: List, id: str) -> bool:

    ids_avaliados['redacoes_avaliadas'].append(id)
    return json_dealer('controle/lista_avaliadas.json', 'write', ids_avaliados)

def save_response_data(avaliacoes_redacoes, id, extracted_data):

    avaliacoes_redacoes[id] = extracted_data
    return json_dealer('controle/avaliacoes.json', 'write', avaliacoes_redacoes)