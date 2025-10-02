import json
from typing import Dict

def json_dealer(path: str, operation: str, new_data: Dict = None):
    """
    Uma função para ler ou escrever dados em um arquivo JSON.

    Args:
        path (str): O caminho para o arquivo JSON (ex: 'dados.json').
        operation (str): A operação a ser realizada. Aceita 'read' ou 'write'.
        new_data (Dict | List, optional): Os dados a serem escritos no arquivo. 
                                           Obrigatório se a operação for 'write'. Default é None.

    """
    
    if operation == 'read':
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data
        except FileNotFoundError:
            print(f"Aviso: O arquivo '{path}' não foi encontrado. Retornando lista vazia.")
            return None
        except json.JSONDecodeError:
            print(f"Erro: O arquivo '{path}' está corrompido ou não é um JSON válido. Retornando None.")
            return None
    
    elif operation == 'write':
        if new_data is None:
            print("Erro: Para a operação 'write', o parâmetro 'new_data' é obrigatório.")
            return False
        
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(new_data, f, indent=4, ensure_ascii=False)
            print(f"Dados salvos com sucesso em '{path}'.")
            return True
        except Exception as e:
            print(f"Erro ao escrever no arquivo '{path}': {e}")
            return False
            
    else:
        print(f"Erro: Operação '{operation}' inválida. Use 'read' ou 'write'.")
        return None

