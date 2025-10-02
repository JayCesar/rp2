# buscar e estruturar dados das redações em dicionario

from .redacao_dict.csv_to_dict import create_redacoes_dict
lista_redacoes = create_redacoes_dict()

# formular um molde de prompt
PROMPT_FORMAT = \
'''

'''

# para cada redacao, fazer chamada API para IA Gemini avaliar
    # verificar se a resposta é válida, 
        # caso n seja válida, tratar
        # armazenar caso seja

# armazenar resultados em um json