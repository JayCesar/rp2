import spacy
import language_tool_python

# Inicialização das ferramentas
print("Carregando ferramentas...")
try:
    nlp = spacy.load('pt_core_news_lg')
    tool = language_tool_python.LanguageTool('pt-BR')
    print("Ferramentas carregadas com sucesso!")
except Exception as e:
    print(f"Erro ao carregar ferramentas: {e}")
    nlp = None
    tool = None

COLOQUIALISMOS = ['mano', 'tá ligado', 'tipo assim', 'né', 'daora']
CONECTIVOS_FORMAIS = ['ademais', 'outrossim', 'dessa forma', 'portanto', 'entretanto', 'contudo']

def extrair_features(texto: str) -> dict:
    features = {}
    if not nlp or not tool: return {}

    matches = tool.check(texto)
    
    features['erros_ortograficos'] = len([m for m in matches if 'MORFOLOGIK_RULE_PT_BR' in m.ruleId or 'MISSING_ACCENT' in m.ruleId])
    features['erros_pontuacao'] = len([m for m in matches if 'PUNCTUATION' in m.ruleId])
    features['erros_de_crase'] = len([m for m in matches if 'CR_CRAS' in m.ruleId])

    doc = nlp(texto)
    lemmas = [token.lemma_ for token in doc if token.is_alpha]
    if len(lemmas) > 0: features['riqueza_lexical'] = len(set(lemmas)) / len(lemmas)
    else: features['riqueza_lexical'] = 0

    sentencas = list(doc.sents)
    if len(sentencas) > 0:
        comprimentos = [len(sent) for sent in sentencas]
        features['comp_medio_sentenca'] = sum(comprimentos) / len(comprimentos)
    else:
        features['comp_medio_sentenca'] = 0

    texto_lower = texto.lower()
    features['contagem_coloquialismos'] = sum(1 for expressao in COLOQUIALISMOS if expressao in texto_lower)
    
    lemmas_lower = [lemma.lower() for lemma in lemmas]
    features['contagem_conectivos_formais'] = sum(1 for conectivo in CONECTIVOS_FORMAIS if conectivo in lemmas_lower)

    return features

texto_exemplo = """
O Brasil enfrenta muitos desafios atualmente, tipo assim, na area da saude. É preciso que medidas
sejam tomadas para resolver esta situação. Ademais, a educação tambem presisa de atenção.
Nós vemos que a falta de recursos é um problema grave. A solução para isso é complexa.
O governo deve investir mais, porem a sociedade precisa colaborar.
"""

features_corrigidas = extrair_features(texto_exemplo)

print("\n[ Dicionário de Features Extraídas ]")
for chave, valor in features_corrigidas.items(): print(f"- {chave}: {valor}")
