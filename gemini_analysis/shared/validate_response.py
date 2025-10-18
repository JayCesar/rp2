import json, logging

from typing import Dict, Any, Tuple
from ..models.Redacao import Redacao

def validate_and_extract_response(response:str, redacao_obj: Redacao) -> Tuple[bool, Redacao]:

    id_redacao = redacao_obj.id

    data: Dict[str, Any] = {}

    cleaned_response = response.strip()
    if cleaned_response.startswith('```json'):
        cleaned_response = cleaned_response[7:].strip()
    if cleaned_response.endswith('```'):
        cleaned_response = cleaned_response[:-3].strip()
    
    if cleaned_response.startswith('```'):
        cleaned_response = cleaned_response[3:].strip()

    try:
        data = json.loads(cleaned_response)
        
    except json.JSONDecodeError:
        logging.error(f"Redação {id_redacao} - Não foi possível converter a response para JSON.")
        print(f"\n{response}\n")
        return False, redacao_obj
    
    
    required_keys = ["nota_c1", "nota_final", "justificativa"]
    if not all(key in data for key in required_keys):
        logging.error(f"Redação {id_redacao} - JSON sem todos os campos necessários.")
        return False, redacao_obj
    
    try:
        data["nota_c1"] = int(data["nota_c1"])
        data["nota_final"] = int(data["nota_final"])
    except (ValueError, TypeError):
        logging.error(f"Redação {id_redacao} - JSON com valores de tipo inválido.")
        return False, redacao_obj
    
    redacao_obj.gemini_nota_c1 = int(data["nota_c1"])
    redacao_obj.gemini_nota_final = int(data["nota_final"])
    redacao_obj.gemini_descricao = data["justificativa"]

    return True, redacao_obj