import concurrent.futures
import language_tool_python
import logging
import pathlib
import polars as pl
import re
import spacy
import transformers

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROW_UPPER_LIMIT = None  # Set to None to use all samples


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


def essay_metrics(essay, essay_c1_score, essay_idx, nlp, tool, tokenizer):
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

    if essay_idx % 25 == 0:
        print(f"[DEBUG] essay_idx: {essay_idx}")

    errors = tool.check(essay)
    error_count = {}
    for error in errors:
        error_category = error.category
        if error_category in error_count:
            error_count[error_category] += 1
        else:
            error_count[error_category] = 1

    total_error_count = 0
    for error_count in error_count.values():
        total_error_count += error_count

    return pl.LazyFrame(
        error_count
        | {
            "c1": essay_c1_score,
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

    dataset_file_path = pathlib.Path.cwd() / "database" / "extended_essay-br.csv"
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

    relevant_columns = "c1", "essay", "prompt"
    max_rows = None  # Set to None to use all samples
    ROW_UPPER_LIMIT = 2**31 - 1
    dataset = (
        pl.scan_csv(dataset_file_path)
        .head(max_rows if max_rows is not None else ROW_UPPER_LIMIT)
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
            .alias("essay_as_single_utf8_string")
        )
        .collect()
    )

    dataset_with_languagetool_metrics = pl.concat(
        (
            essay_metrics(essay, c1, idx, nlp, tool, tokenizer).collect()
            for idx, (essay, c1) in enumerate(
                zip(dataset["essay_as_single_utf8_string"], dataset["c1"])
            )
        ),
        how="diagonal",
    ).with_columns(pl.all().fill_null(strategy="zero"))
    print(
        "\n\n[DEBUG] dataset_with_languagetool_metrics:\n",
        dataset_with_languagetool_metrics,
    )

    dataset_file_path_parent = pathlib.Path.cwd() / "generated_datasets"
    bert_model_name_path_suffix = bert_model_huggingface_repo.replace("/", "--")

    dataset_with_languagetool_metrics_parquet_file_path = (
        dataset_file_path_parent
        / f"dataset_with_languagetool_metrics_{bert_model_name_path_suffix}.parquet"
    )
    dataset_with_languagetool_metrics.write_parquet(
        dataset_with_languagetool_metrics_parquet_file_path
    )
    print(
        "[DEBUG] Metrics written to Parquet file: ",
        dataset_with_languagetool_metrics_parquet_file_path,
    )

    dataset_with_languagetool_metrics_csv_file_path = (
        dataset_file_path_parent
        / f"dataset_with_languagetool_metrics_{bert_model_name_path_suffix}.csv"
    )
    dataset_with_languagetool_metrics.write_csv(
        dataset_with_languagetool_metrics_csv_file_path
    )
    print(
        "[DEBUG] Metrics written to CSV file: ",
        dataset_with_languagetool_metrics_csv_file_path,
    )

    dataset_with_languagetool_metrics_json_file_path = (
        dataset_file_path_parent
        / f"dataset_with_languagetool_metrics_{bert_model_name_path_suffix}.json"
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
