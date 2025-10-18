from ..models.Redacao import Redacao

def generate_redacao_input(redacao_obj: Redacao, prompt_template: str) -> str:

    prompt_input: str = prompt_template.format(
        enunciado_titulo=redacao_obj.enunciado_titulo,
        enunciado=redacao_obj.enunciado,
        titulo=redacao_obj.titulo,
        texto=redacao_obj.texto
    )
    
    return prompt_input