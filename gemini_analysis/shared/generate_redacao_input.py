from typing import Dict

def generate_redacao_input(redacao_dict: Dict, prompt_template: str) -> str:

    prompt_input: str = prompt_template.format(
        enunciado_titulo=redacao_dict['enunciado_titulo'],
        enunciado=redacao_dict['enunciado'],
        titulo=redacao_dict['titulo'],
        texto=redacao_dict['texto']
    )
    
    return prompt_input