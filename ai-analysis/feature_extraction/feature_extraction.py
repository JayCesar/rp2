import spacy
import language_tool_python

nlp = spacy.load('pt_core_news_lg')
tool = language_tool_python.LanguageTool('pt-BR')

def extrair_todas_features(texto: str) -> list:
    """
    Função orquestradora que chama todas as funções de extração 
    e retorna um único vetor numérico.
    """
    
    features_combinadas = {}

    # Chama cada extrator e atualiza o dicionário principal
    features_combinadas.update(extrair_features_languagetool(texto, tool))
    features_combinadas.update(extrair_features_spacy(texto, nlp))
    features_combinadas.update(extrair_features_custom(texto))

    return list(features_combinadas.values())


def extrair_features_languagetool(texto: str, tool: language_tool_python.LanguageTool) -> dict:
    """Extrai features baseadas nas regras do LanguageTool."""
    features_lt = {}
    matches = tool.check(texto)
    
    features_lt['erros_ortograficos'] = len([
        m for m in matches if 'MORFOLOGIK_RULE_PT_BR' in m.ruleId or 'MISSING_ACCENT' in m.ruleId
    ])
    features_lt['erros_pontuacao'] = len([m for m in matches if 'PUNCTUATION' in m.ruleId])
    features_lt['erros_de_crase'] = len([m for m in matches if 'CR_CRAS' in m.ruleId])
    
    return features_lt

def extrair_features_spacy(texto: str, nlp) -> dict:
    """Extrai features estruturais e lexicais usando o spaCy."""
    features_spacy = {}
    doc = nlp(texto)
    
    # Riqueza Lexical
    lemmas = [token.lemma_ for token in doc if token.is_alpha]
    if len(lemmas) > 0:
        features_spacy['riqueza_lexical'] = len(set(lemmas)) / len(lemmas)
    else:
        features_spacy['riqueza_lexical'] = 0

    # Comprimento médio das sentenças
    sentencas = list(doc.sents)
    if len(sentencas) > 0:
        comprimentos = [len(sent) for sent in sentencas]
        features_spacy['comp_medio_sentenca'] = sum(comprimentos) / len(comprimentos)
    else:
        features_spacy['comp_medio_sentenca'] = 0
        
    return features_spacy

# Defina suas listas no escopo global para que a função possa acessá-las
COLOQUIALISMOS = ['mano', 'tá ligado', 'tipo assim', 'né', 'daora']
CONECTIVOS_FORMAIS = ['ademais', 'outrossim', 'dessa forma', 'portanto', 'entretanto', 'contudo']

def extrair_features_custom(texto: str) -> dict:
    """Extrai features baseadas em listas de palavras customizadas."""
    features_custom = {}
    texto_lower = texto.lower()
    
    features_custom['contagem_coloquialismos'] = sum(1 for exp in COLOQUIALISMOS if exp in texto_lower)
    features_custom['contagem_conectivos_formais'] = sum(1 for con in CONECTIVOS_FORMAIS if con in texto_lower)
    
    return features_custom