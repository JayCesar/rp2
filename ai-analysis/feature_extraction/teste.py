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

COLOQUIALISMOS = [
    "mano", "tá ligado", "tipo assim", "né", "daora", "abestado", "abiscoitar", "abufelar", "abotoar o paletó", "acabar em pizza", "acertar na mosca", "acoitar", "adoidado", "adoçar a boca", "afogado", "alçar a perna", "alguém me segura", "aluado", "amancebado", "amarrado", "amoado", "amofinado", "aparelho", "aperrear", "apombaiado", "arrebentar", "arregaçar as mangas", "arretado", "arrochar o nó", "arruinou", "asa dura", "avaixe", "avalie", "avexado", "avoar no mato", "azedou o caldo", "azagaia", "azangar", "azuretado", "babado", "baba ovo", "bah", "baixa da égua", "balada", "balão", "bater a caçoleta", "bater um rango", "bater uma bolinha", "bau", "beiçudo", "bereré", "bicho", "bicho-carpinteiro", "bicuda", "biscoiteiro", "bitelo", "bizonho", "boiar", "bola nas costas", "bolado", "bocada", "borracho", "borogodó", "botar as barbas de molho", "botar banca", "brabo", "breja", "brenha", "briba", "brocado", "brother", "brotar", "bruguelo", "buchuda", "bugre", "buliçoso", "buzão", "caatinga", "cabrunco", "cabuloso", "cachaça", "cacetinho", "cafuringa", "cair os butiá do bolso", "calabreso", "camela", "campo santo", "caô", "capar o gato", "capaz", "caraca", "carapanã", "carcunda", "casca de bala", "chapa", "chapar o coco", "chavecar", "chiar", "chutar o balde", "chutar o pau da barraca", "chuvinha de leve", "coisar", "colar", "colocar melancia na cabeça", "comer na gaveta", "confundir alhos com bugalhos", "crush", "curumim", "curtir", "cutucar a onça com vara curta", "da hora", "dar a volta por cima", "dar bolo", "dar de ombros", "dar uma banda", "dar uma mão", "dar uma segurada", "dar uma canja", "dar uma mãozinha", "dar uma olhada", "dar um tempo", "de boa", "de rocha", "deu a louca", "deu mole", "deu ruim", "diabéisso", "dispense", "dormir no macio", "embrazado", "embretar-se", "empacado", "empapar", "empedrar", "empatar", "engasgar", "escrachado", "esgualepado", "espantar", "esparrado", "estribado", "falar cobras e lagartos", "fazer uma vaquinha", "ficar de bubuia", "ficar de boa", "ficar de nhe nhe nhe", "ficar na moita", "ficar de olho", "firmeza", "flanelinha", "flopar", "friaca", "frisete", "fuzuê", "gaitxar", "gaiato", "gastura", "guacho", "guaipeca", "guasca", "guenzo", "guria", "guri", "iapois", "içar", "ilhado", "ir para o beleléu", "ixi", "jacú", "já é", "jão", "jardim da infância", "jeca", "jegue", "kiu", "lacrou", "lagartear", "larica", "lascar o cano", "lavar as mãos", "levar toco", "levar um bolo", "levar o farelo", "lindeiro", "lisca", "liso", "lombra", "macambúzio", "macho", "maleva", "mangar", "mandar bem", "mano", "manteiga", "marombado", "maria-vai-com-as-outras", "matar a cobra e mostrar o pau", "matar cachorro a grito", "mauricinho", "mec", "mermão", "migué", "mina", "miudinho", "mó barato", "molhado", "morgado", "morreu", "moscou", "muqui", "na boa", "na faixa", "na moleza", "na tora", "nas coxas", "no sapatinho", "nu", "o bicho está pegando", "o gato subiu no telhado", "olada", "olho gordo", "oxente", "pagar mico", "pagar pau", "pagar sapo", "pagar vexa", "paia", "papo reto", "parça", "partiu", "patife", "pau-d'água", "pau-pra-toda-obra", "pé-de-boi", "pé rachado", "pegar o beco", "pegar uma carona", "pegar uma treta", "peguete", "pelejar", "perrengue", "piá", "pisa menos", "pisar na bola", "pistola", "pitiú", "pocar", "pode pá", "pongar", "popudinho", "pôr minhoca na cabeça", "puxar o saco da cuia", "quebrado", "queimar o filme", "quem não tem cão caça com gato", "ranço", "rato", "rebolar no mato", "relho", "rolê", "sabe-tudo", "salve", "sangue bom", "se é louco", "se pique", "se ligar", "ser uma pedra no sapato", "shipper", "sinistro", "só o pó", "soltar a franga", "solito", "sussa", "sustança", "tá ligado", "tá liso", "tá forrado", "tá chovendo duro", "tá me tirando", "teú", "tchê", "tijolinho", "tirana", "tiração", "tô ligado", "top", "treta", "trem", "tri", "trocar ideia", "trovar", "tubão", "umborimbora", "vacilão", "vai dar zebra", "vazar", "vazar na braquiara", "véi", "vigia bem", "vixe", "vixe", "vôte", "vtzeiro", "xavecar", "zé ruela", "zoado", "zueira"
]
CONECTIVOS_FORMAIS = [
    "outrossim", "ademais", "além disso", "a propósito", "acima de tudo", "acerca de", "assim", "assim como", "assim sendo", "ao contrário", "ao passo que", "a fim de", "a saber", "a despeito de", "apesar de", "com efeito", "com isso", "com o fim de", "consequentemente", "contudo", "como resultado", "considerando que", "de acordo com", "de forma que", "de fato", "devido a", "diante disso", "dessa forma", "desse modo", "em contrapartida", "em outras palavras", "em vez de", "em virtude de", "em suma", "em síntese", "em particular", "em primeiro lugar", "em segundo lugar", "embora", "entretanto", "enquanto", "exceto", "finalmente", "graças a", "igualmente", "inclusive", "logo", "mas", "mediante", "no entanto", "na medida em que", "ou seja", "ou", "por conseguinte", "por exemplo", "por outro lado", "porém", "portanto", "pois", "primeiramente", "principalmente", "por isso", "para que", "salvo", "seja", "semelhantemente", "sob o prisma de", "sobretudo", "tal como", "tão logo", "uma vez que", "visto que"
]

def extrair_features(texto: str) -> dict:
    features = {}
    if not nlp or not tool: return {}

    matches = tool.check(texto)
    
    features['erros_ortograficos'] = len([m for m in matches if 'MORFOLOGIK_RULE_PT_BR' in m.ruleId or 'MISSING_ACCENT' in m.ruleId])

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

import matplotlib.pyplot as plt
from redacao_dict.csv_to_dict import create_redacoes_dict
from tqdm import tqdm

def run():

    dados_features = {
        'erros_ortograficos': [],
        'riqueza_lexical': [],
        'comp_medio_sentenca': [],
        'contagem_coloquialismos': [],
        'contagem_conectivos_formais': [],
        'nota_c1': [],
        'nota_final': []
    }

    redacoes_qt = 100
    redacoes_dict = create_redacoes_dict(redacoes_qt)

    for redacao in tqdm(redacoes_dict, desc="Extraindo e coletando dados"):
        features_corrigidas = extrair_features(redacao['texto'])
        for chave, valor in features_corrigidas.items():
            if chave in dados_features:
                dados_features[chave].append(valor)
        
        dados_features['nota_c1'].append(redacao.get('nota_c1', 0))
        dados_features['nota_final'].append(redacao.get('nota_final', 0))

    fig, axs = plt.subplots(2, 4, figsize=(20, 10))

    fig.suptitle('Distribuição de Features e Notas das Redações', fontsize=18)
    
    axs_flat = axs.flatten()
    chaves = list(dados_features.keys())

    for i, chave in enumerate(chaves):
        ax = axs_flat[i]
        
        n, bins, patches = ax.hist(dados_features[chave], bins=20, edgecolor='black', alpha=0.7)
        
        ax.set_title(chave.replace('_', ' ').title())
        ax.set_xlabel('Valor')
        ax.set_ylabel('Frequência')
        ax.grid(axis='y', alpha=0.75)

        for count, rect in zip(n, patches):
            if count > 0:
                height = rect.get_height()
                ax.text(
                    rect.get_x() + rect.get_width() / 2, 
                    height, 
                    int(count), 
                    ha='center', 
                    va='bottom',
                    fontsize=8
                )
    
    axs_flat[-1].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    plt.savefig(f"graficos_{redacoes_qt}.png")

    print("\n" + "-"*40)
    print("[ Valores Médios das Features e Notas ]")
    print("-"*40)

    for chave, valores in dados_features.items():
        media_calculada = sum(valores) / len(valores) if valores else 0
        print(f"  - {chave.replace('_', ' ').title()}: {media_calculada:.2f}")

    print("-"*40)
