PROMPT_INPUT_MODEL: str = \
'''
# CONTEXTO
Correção do Exame Nacional do Ensino Médio (ENEM), focada 100% na avaliação da Competência 1, que avalia o domínio da modalidade escrita formal da língua portuguesa.

# CONTROLE DE FORMATO
A sua resposta deve ser um objeto JSON válido, e NADA MAIS. Não adicione texto extra.
O objeto JSON deve conter exatamente as seguintes chaves:
- "nota_c1": um número inteiro (0, 40, 80, 120, 160 ou 200).

# ----------------------------------------------------
# DADOS PARA AVALIAÇÃO
# ----------------------------------------------------

# ENUNCIADO DA PROPOSTA
## Título da Proposta: {enunciado_titulo}
## Texto da Proposta: {enunciado}

# REDAÇÃO DO ALUNO
## Título da Redação: {titulo}
## Texto da Redação:
---
{texto}
---
'''