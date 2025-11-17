import pathlib
import unicodedata
import re

import language_tool_python
import polars as pl
import utils

logger = utils.logger

# Configuration
MAX_SAMPLES = None  # Set to None to use all samples

spacy_model_name = "pt_core_news_md"
try:
    nlp = utils.spacy_model(spacy_model_name)
except OSError:
    logger.error(f"Failed to load spaCy model {spacy_model_name}")

logger.info("Initializing LanguageTool for Portuguese (Brazil)...")
tool = language_tool_python.LanguageTool("pt-BR")
logger.info("LanguageTool initialized successfully")


def _normalize(text: str) -> str:
    """Lowercase and strip diacritics for robust string comparison."""
    normalized = unicodedata.normalize("NFD", text)
    without_diacritics = "".join(
        ch for ch in normalized if not unicodedata.combining(ch)
    )
    return without_diacritics.lower()


COLLOQUIALISMS = [
    "mano",
    "tá ligado",
    "tipo assim",
    "né",
    "daora",
    "abestado",
    "abiscoitar",
    "abufelar",
    "abotoar o paletó",
    "acabar em pizza",
    "acertar na mosca",
    "acoitar",
    "adoidado",
    "adoçar a boca",
    "afogado",
    "alçar a perna",
    "alguém me segura",
    "aluado",
    "amancebado",
    "amarrado",
    "amoado",
    "amofinado",
    "aparelho",
    "aperrear",
    "apombaiado",
    "arrebentar",
    "arregaçar as mangas",
    "arretado",
    "arrochar o nó",
    "arruinou",
    "asa dura",
    "avaixe",
    "avalie",
    "avexado",
    "avoar no mato",
    "azedou o caldo",
    "azagaia",
    "azangar",
    "azuretado",
    "babado",
    "baba ovo",
    "bah",
    "baixa da égua",
    "balada",
    "balão",
    "bater a caçoleta",
    "bater um rango",
    "bater uma bolinha",
    "bau",
    "beiçudo",
    "bereré",
    "bicho",
    "bicho-carpinteiro",
    "bicuda",
    "biscoiteiro",
    "bitelo",
    "bizonho",
    "boiar",
    "bola nas costas",
    "bolado",
    "bocada",
    "borracho",
    "borogodó",
    "botar as barbas de molho",
    "botar banca",
    "brabo",
    "breja",
    "brenha",
    "briba",
    "brocado",
    "brother",
    "brotar",
    "bruguelo",
    "buchuda",
    "bugre",
    "buliçoso",
    "buzão",
    "caatinga",
    "cabrunco",
    "cabuloso",
    "cachaça",
    "cacetinho",
    "cafuringa",
    "cair os butiá do bolso",
    "calabreso",
    "camela",
    "campo santo",
    "caô",
    "capar o gato",
    "capaz",
    "caraca",
    "carapanã",
    "carcunda",
    "casca de bala",
    "chapa",
    "chapar o coco",
    "chavecar",
    "chiar",
    "chutar o balde",
    "chutar o pau da barraca",
    "chuvinha de leve",
    "coisar",
    "colar",
    "colocar melancia na cabeça",
    "comer na gaveta",
    "confundir alhos com bugalhos",
    "crush",
    "curumim",
    "curtir",
    "cutucar a onça com vara curta",
    "da hora",
    "dar a volta por cima",
    "dar bolo",
    "dar de ombros",
    "dar uma banda",
    "dar uma mão",
    "dar uma segurada",
    "dar uma canja",
    "dar uma mãozinha",
    "dar uma olhada",
    "dar um tempo",
    "de boa",
    "de rocha",
    "deu a louca",
    "deu mole",
    "deu ruim",
    "diabéisso",
    "dispense",
    "dormir no macio",
    "embrazado",
    "embretar-se",
    "empacado",
    "empapar",
    "empedrar",
    "empatar",
    "engasgar",
    "escrachado",
    "esgualepado",
    "espantar",
    "esparrado",
    "estribado",
    "falar cobras e lagartos",
    "fazer uma vaquinha",
    "ficar de bubuia",
    "ficar de boa",
    "ficar de nhe nhe nhe",
    "ficar na moita",
    "ficar de olho",
    "firmeza",
    "flanelinha",
    "flopar",
    "friaca",
    "frisete",
    "fuzuê",
    "gaitxar",
    "gaiato",
    "gastura",
    "guacho",
    "guaipeca",
    "guasca",
    "guenzo",
    "guria",
    "guri",
    "iapois",
    "içar",
    "ilhado",
    "ir para o beleléu",
    "ixi",
    "jacú",
    "já é",
    "jão",
    "jardim da infância",
    "jeca",
    "jegue",
    "kiu",
    "lacrou",
    "lagartear",
    "larica",
    "lascar o cano",
    "lavar as mãos",
    "levar toco",
    "levar um bolo",
    "levar o farelo",
    "lindeiro",
    "lisca",
    "liso",
    "lombra",
    "macambúzio",
    "macho",
    "maleva",
    "mangar",
    "mandar bem",
    "mano",
    "manteiga",
    "marombado",
    "maria-vai-com-as-outras",
    "matar a cobra e mostrar o pau",
    "matar cachorro a grito",
    "mauricinho",
    "mec",
    "mermão",
    "migué",
    "mina",
    "miudinho",
    "mó barato",
    "molhado",
    "morgado",
    "morreu",
    "moscou",
    "muqui",
    "na boa",
    "na faixa",
    "na moleza",
    "na tora",
    "nas coxas",
    "no sapatinho",
    "nu",
    "o bicho está pegando",
    "o gato subiu no telhado",
    "olada",
    "olho gordo",
    "oxente",
    "pagar mico",
    "pagar pau",
    "pagar sapo",
    "pagar vexa",
    "paia",
    "papo reto",
    "parça",
    "partiu",
    "patife",
    "pau-d'água",
    "pau-pra-toda-obra",
    "pé-de-boi",
    "pé rachado",
    "pegar o beco",
    "pegar uma carona",
    "pegar uma treta",
    "peguete",
    "pelejar",
    "perrengue",
    "piá",
    "pisa menos",
    "pisar na bola",
    "pistola",
    "pitiú",
    "pocar",
    "pode pá",
    "pongar",
    "popudinho",
    "pôr minhoca na cabeça",
    "puxar o saco da cuia",
    "quebrado",
    "queimar o filme",
    "quem não tem cão caça com gato",
    "ranço",
    "rato",
    "rebolar no mato",
    "relho",
    "rolê",
    "sabe-tudo",
    "salve",
    "sangue bom",
    "se é louco",
    "se pique",
    "se ligar",
    "ser uma pedra no sapato",
    "shipper",
    "sinistro",
    "só o pó",
    "soltar a franga",
    "solito",
    "sussa",
    "sustança",
    "tá ligado",
    "tá liso",
    "tá forrado",
    "tá chovendo duro",
    "tá me tirando",
    "teú",
    "tchê",
    "tijolinho",
    "tirana",
    "tiração",
    "tô ligado",
    "top",
    "treta",
    "trem",
    "tri",
    "trocar ideia",
    "trovar",
    "tubão",
    "umborimbora",
    "vacilão",
    "vai dar zebra",
    "vazar",
    "vazar na braquiara",
    "véi",
    "vigia bem",
    "vixe",
    "vixe",
    "vôte",
    "vtzeiro",
    "xavecar",
    "zé ruela",
    "zoado",
    "zueira",
]

FORMAL_CONJUNCTIONS = [
    "outrossim",
    "ademais",
    "além disso",
    "a propósito",
    "acima de tudo",
    "acerca de",
    "assim",
    "assim como",
    "assim sendo",
    "ao contrário",
    "ao passo que",
    "a fim de",
    "a saber",
    "a despeito de",
    "apesar de",
    "com efeito",
    "com isso",
    "com o fim de",
    "consequentemente",
    "contudo",
    "como resultado",
    "considerando que",
    "de acordo com",
    "de forma que",
    "de fato",
    "devido a",
    "diante disso",
    "dessa forma",
    "desse modo",
    "em contrapartida",
    "em outras palavras",
    "em vez de",
    "em virtude de",
    "em suma",
    "em síntese",
    "em particular",
    "em primeiro lugar",
    "em segundo lugar",
    "embora",
    "entretanto",
    "enquanto",
    "exceto",
    "finalmente",
    "graças a",
    "igualmente",
    "inclusive",
    "logo",
    "mas",
    "mediante",
    "no entanto",
    "na medida em que",
    "ou seja",
    "ou",
    "por conseguinte",
    "por exemplo",
    "por outro lado",
    "porém",
    "portanto",
    "pois",
    "primeiramente",
    "principalmente",
    "por isso",
    "para que",
    "salvo",
    "seja",
    "semelhantemente",
    "sob o prisma de",
    "sobretudo",
    "tal como",
    "tão logo",
    "uma vez que",
    "visto que",
]

NORMALIZED_COLLOQUIALISMS = [_normalize(exp) for exp in COLLOQUIALISMS]
NORMALIZED_FORMAL_CONJUNCTIONS = [_normalize(con) for con in FORMAL_CONJUNCTIONS]

COLLOQUIALISMS_PATTERN = re.compile(
    "|".join(re.escape(exp) for exp in NORMALIZED_COLLOQUIALISMS)
)
FORMAL_CONJUNCTIONS_PATTERN = re.compile(
    "|".join(re.escape(exp) for exp in NORMALIZED_FORMAL_CONJUNCTIONS)
)


def essay_metrics(essay_data, total_essay_count):
    essay_id = essay_data["essay_id"]
    if essay_id % 10 == 0:
        logger.info(f"Processing essay {essay_id}/{total_essay_count}...")

    essay = essay_data["essay_as_single_utf8_string"]

    # Basic text statistics
    doc = nlp(essay)
    word_count = len([token for token in doc if not token.is_punct])
    sentences = list(doc.sents)
    sentence_count = len(sentences)

    # LanguageTool error checking
    errors = tool.check(essay)
    # logger.info(f"Errors found in essay {essay_idx}: {errors}")

    error_counts = {}
    for error in errors:
        error_category = error.category
        if error_category in error_counts:
            error_counts[error_category] += 1
        else:
            error_counts[error_category] = 1

    total_error_count = sum(error_counts.values())

    # Lexical diversity
    lemmas = [token.lemma_ for token in doc if token.is_alpha]
    features_spacy = {}
    if len(lemmas) > 0:
        features_spacy["LEXICAL_DIVERSITY"] = len(set(lemmas)) / len(lemmas)
    else:
        # ensure float dtype for consistent schema across rows
        features_spacy["LEXICAL_DIVERSITY"] = 0.0

    # Sentence average length
    if sentence_count:
        sentence_lengths = [len(sentence) for sentence in sentences]
        features_spacy["AVERAGE_SENTENCE_LENGTH"] = sum(sentence_lengths) / len(
            sentence_lengths
        )
    else:
        # ensure float dtype for consistent schema across rows
        features_spacy["AVERAGE_SENTENCE_LENGTH"] = 0.0

    normalized_essay = _normalize(essay)

    features_custom = {
        "COLLOQUALISM_COUNT": sum(
            1 for _ in COLLOQUIALISMS_PATTERN.finditer(normalized_essay)
        ),
        "FORMAL_CONJUNCTION_COUNT": sum(
            1 for _ in FORMAL_CONJUNCTIONS_PATTERN.finditer(normalized_essay)
        ),
    }

    df = pl.DataFrame(
        error_counts
        | essay_data
        | {
            "TOTAL_ERROR_COUNT": total_error_count,
            "WORD_COUNT": word_count,
            "SENTENCE_COUNT": sentence_count,
        }
        | features_spacy
        | features_custom
    )

    # enforce float dtypes for spacy-derived ratio features
    df = df.with_columns(
        [
            pl.col("LEXICAL_DIVERSITY").cast(pl.Float64),
            pl.col("AVERAGE_SENTENCE_LENGTH").cast(pl.Float64),
        ]
    )

    return df


def essay_token_count(encoded_essay):
    essay_tokens = encoded_essay["input_ids"][0]

    return len(essay_tokens)


def main():
    logger.info("Starting LanguageTool feature extraction process...")

    dataset_parquet_file_path = (
        pathlib.Path.cwd()
        / "generated_datasets"
        / "extended_essay-br_preprocessed_for_LanguageTool.parquet"
    )
    logger.info(f"Checking dataset file: {dataset_parquet_file_path}")
    if not dataset_parquet_file_path.exists():
        logger.error(f"Dataset file not found at path {dataset_parquet_file_path}")
        return

    logger.info(f"Loading dataset from {dataset_parquet_file_path}...")
    relevant_columns = "c1", "essay_as_single_utf8_string", "prompt"
    dataset = (
        pl.scan_parquet(dataset_parquet_file_path)
        .select(relevant_columns)
        .drop_nulls()
        .unique()
        .with_row_index("essay_id")
    )

    # Apply row limit if specified
    if MAX_SAMPLES is not None:
        logger.info(f"Applying row limit: {MAX_SAMPLES}")
        dataset = dataset.head(MAX_SAMPLES)
        logger.info(f"Applied row limit. Processing at most {MAX_SAMPLES} essays")
    else:
        logger.info("No row limit applied. Processing all essays")
    dataset = dataset.collect()
    logger.info(f"Dataset loaded successfully. Shape: {dataset.shape}")

    total_essay_count = len(dataset)
    logger.info(f"Starting feature extraction for {total_essay_count} essays...")

    results = (
        essay_metrics(essay_data, total_essay_count)
        for essay_data in dataset.to_dicts()
    )

    logger.info("Concatenating results...")
    dataset_with_languagetool_metrics = pl.concat(
        results,
        how="diagonal",
    ).with_columns(pl.all().fill_null(strategy="zero"))

    logger.info(
        f"Feature extraction completed. Result shape: {dataset_with_languagetool_metrics.shape}"
    )
    logger.info(f"Final dataset preview:\n{dataset_with_languagetool_metrics.head()}")
    logger.info(f"Final dataset columns: {dataset_with_languagetool_metrics.columns}")

    # Save results to files
    project_root = pathlib.Path(__file__).parent.parent.parent
    assert project_root.name == "rp2"

    generated_datasets_directory = project_root / "generated_datasets"
    generated_datasets_directory.mkdir(exist_ok=True)

    dataset_with_languagetool_metrics_filename = "dataset_with_languagetool_metrics"
    dataset_with_languagetool_metrics_extensions = "parquet", "csv", "json"

    utils.save_dataset(
        dataset_with_languagetool_metrics,
        dataset_with_languagetool_metrics_filename,
        *dataset_with_languagetool_metrics_extensions,
    )

    logger.info("LanguageTool feature extraction process completed successfully!")


if __name__ == "__main__":
    main()
