import logging
import pathlib
import re
import subprocess
import sys

import polars as pl
import spacy

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


def spacy_model(spacy_model_name: str):
    def install_spacy_model(model_name: str) -> None:
        """Install a spaCy model if it's not already available.

        Args:
            model_name: Name of the spaCy model to install
        """
        logger.info(f"Installing spaCy model: {model_name}")
        try:
            subprocess.run(
                [sys.executable, "-m", "spacy", "download", model_name],
                check=True,
                capture_output=True,
                text=True,
            )
            logger.info(f"Successfully installed spaCy model: {model_name}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install spaCy model {model_name}: {e}")
            logger.error(f"Command output: {e.stdout}")
            logger.error(f"Command error: {e.stderr}")
            raise

    try:
        nlp = spacy.load(spacy_model_name)
        logger.info(f"Loaded spaCy model: {spacy_model_name}")
    except OSError:
        logger.warning(
            f"Portuguese spaCy model '{spacy_model_name}' not found. Installing automatically..."
        )
        install_spacy_model(spacy_model_name)
        try:
            nlp = spacy.load(spacy_model_name)
            logger.info(
                f"Successfully loaded spaCy model after installation: {spacy_model_name}"
            )
        except OSError as e:
            logger.error(f"Failed to load spaCy model even after installation: {e}")
            sys.exit(1)

    return nlp


def save_dataset(dataset: pl.DataFrame, filename: str, *extensions: str) -> None:
    project_root = pathlib.Path(__file__).parent.parent.parent
    assert project_root.name == "rp2"

    for extension in extensions:
        dataset_filepath = (
            project_root / "generated_datasets" / f"{filename}.{extension}"
        )
        logger.info(
            f"Writing dataset to {extension} file: {dataset_filepath}",
        )
        getattr(dataset, f"write_{extension}")(dataset_filepath)
        logger.info(
            f"Pre-processed dataset for BERT written to {extension} file: {dataset_filepath}"
        )
