import language_tool_python
import numpy as np
import pathlib
import polars as pl
import torch
import transformers
import spacy
import re


def essay_line_to_single_utf8_string(essay_line: str):
    """
    Sequence of transformations performed on each essay_line:
    1) Use eval() to turn the essay_line from a Python list string representation
       into a Python list
    2) Join all the sentences of the essay with a " " between them
    3) Remove all sequences of multiple (more than 1) whitespaces in a row and
       replace each of them with a single " "
    """

    return re.sub(r"\s\s+", " ", " ".join(eval(essay_line)))


def essay_metrics(essay, nlp, tool, tokenizer):
    # Basic text statistics
    doc = nlp(essay)
    word_count = len([token for token in doc if not token.is_punct])
    sentence_count = len(list(doc.sents))
    token_amount = essay_token_count(
        tokenizer(
            essay,
            return_tensors="pt",
        )
    )
    if token_amount > 512:
        print(
            f"[WARNING] essay's token count ({token_amount}) exceeds BERT's token limit (essay_token_amount > 512)"
        )

    errors = tool.check(essay)
    error_count = {}
    for error in errors:
        error_category = error.category
        if error_category in error_count:
            error_count[error_category] += 1
        else:
            error_count[error_category] = 1

        print(f"{error_category} error: {error}")

    total_error_count = 0
    for key, value in error_count.items():
        print(f"[DEBUG] {key} error count: {value}")

        total_error_count += value
    print(f"\n[DEBUG] Total error count: {total_error_count}")

    return pl.LazyFrame(
        error_count
        | {
            "total_error_count": total_error_count,
            "word_count": word_count,
            "sentence_count": sentence_count,
            "token_count": token_amount,
        }
    )


def essay_token_count(encoded_essay):
    essay_tokens = encoded_essay["input_ids"][0]

    return len(essay_tokens)


def main():
    nlp = spacy.load("pt_core_news_lg")
    tool = language_tool_python.LanguageTool("pt-BR")

    dataset_file_path = (
        pathlib.Path.cwd().joinpath("database").joinpath("extended_essay-br.csv")
    )
    if not dataset_file_path.exists():
        print(
            f"""[ERROR] Dataset file  not found at path {dataset_file_path} the
            script must be executed from the project's root directory"""
        )

        return

    bert_model_huggingface_repo = "neuralmind/bert-base-portuguese-cased"  # BERTimbau
    # bert_model_huggingface_repo = "ricardoz/BERTugues-base-portuguese-cased" # BERTugues

    tokenizer = transformers.AutoTokenizer.from_pretrained(bert_model_huggingface_repo)
    print("[DEBUG] Tokenizer loaded")
    model = transformers.AutoModel.from_pretrained(bert_model_huggingface_repo)
    print("[DEBUG] Model loaded")

    # encoded_essay = tokenizer(
    #     "O esporte é uma prática super popular que muitos conhecem, várias pessoas vivem da prática do esporte, da competição, entre países ou entre estados, muitas pessoas praticam ou já praticou algum tipo de esporte em sua vida. Porém, o esporte tem seu lado doloroso, atletas com fraturas, lesões, e para combater isso tomam anabolizantes para não sofrer tanto com os treinos e ter um bom resultado na competição, mas esse tipo de ação e proibido no esporte, porque pode ser detectado em exames de rotina dos atletas. Grandes competidores arriscam a própria saúde para chegar ao primeiro lugar, e com algum movimento em falso ou mal sucedido, pode colocar em risco a saúde e a própria carreira, porque são na maioria das vezes cobrados pelos treinadores e pelos fans e acabam se machucando, os atletas deveriam ser contemplados com uma lei para que não sejam forçados a fazer nada ou nenhum movimento arriscado, para que os riscos diminuam e não se tornem um peso para o atleta, que acima de tudo ele está competindo com lealdade e dedicação nos treinos. E com essa medida os esportistas se sentiriam melhor ao competir de maneira limpa, sem pressão ou trapassa , para que fazem isso porque gostem e porque se sintam de maneira que não comprometa a saúde e o bem estar do atleta.",
    #     return_tensors="pt",
    # )
    # print(essay_token_count(encoded_essay))
    # with torch.no_grad():
    #     model_output = model(**encoded_essay)
    # # Get the embedding for the [CLS] token (first token of the sentence)
    # vector = model_output.last_hidden_state[:, 0, :].numpy()
    # vectors_dimension_amount = np.shape(vector)[1]
    # print(vector)
    # print(f"[DEBUG] Vectors dimension amount: {vectors_dimension_amount}")

    relevant_columns = "c1", "essay", "prompt"
    dataset = (
        pl.scan_csv(dataset_file_path)
        .head(25)  # Comment this line to process the whole dataset
        .select(relevant_columns)
        .drop_nulls()
        .unique()
        .with_columns(
            pl.col("essay")
            .map_batches(
                lambda essay_column: pl.Series(
                    (
                        essay_line_to_single_utf8_string(essay_line)
                        for essay_line in essay_column
                    )
                ),
                return_dtype=pl.Utf8,
            )
            .alias("essay_as_single_string")
        )
    ).collect()
    print(dataset)

    dataset_with_languagetool_metrics = pl.concat(
        (
            essay_metrics(essay, nlp, tool, tokenizer).collect()
            for essay in dataset["essay_as_single_string"]
        ),
        how="diagonal",
    ).with_columns(pl.all().fill_null(strategy="zero"))
    print(
        "\n\n[DEBUG] dataset_with_languagetool_metrics:\n",
        dataset_with_languagetool_metrics,
    )

    dataset_file_path_parent = pathlib.Path.cwd().joinpath("generated_datasets")
    bert_model_name_path_suffix = bert_model_huggingface_repo.replace("/", "--")

    dataset_with_languagetool_metrics_parquet_file_path = (
        dataset_file_path_parent.joinpath(
            f"dataset_with_languagetool_metrics_{bert_model_name_path_suffix}.parquet"
        )
    )
    dataset_with_languagetool_metrics.write_parquet(
        dataset_with_languagetool_metrics_parquet_file_path
    )
    print(
        "[DEBUG] Metrics written to Parquet file: ",
        dataset_with_languagetool_metrics_parquet_file_path,
    )

    dataset_with_languagetool_metrics_csv_file_path = dataset_file_path_parent.joinpath(
        f"dataset_with_languagetool_metrics_{bert_model_name_path_suffix}.csv"
    )
    dataset_with_languagetool_metrics.write_csv(
        dataset_with_languagetool_metrics_csv_file_path
    )
    print(
        "[DEBUG] Metrics written to CSV file: ",
        dataset_with_languagetool_metrics_csv_file_path,
    )

    dataset_with_languagetool_metrics_json_file_path = (
        dataset_file_path_parent.joinpath(
            f"dataset_with_languagetool_metrics_{bert_model_name_path_suffix}.json"
        )
    )
    dataset_with_languagetool_metrics.write_json(
        dataset_with_languagetool_metrics_json_file_path
    )
    print(
        "[DEBUG] Metrics written to JSON file: ",
        dataset_with_languagetool_metrics_json_file_path,
    )


if __name__ == "__main__":
    main()
