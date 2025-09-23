import logging
import os
import pathlib
import re
import string
import sys

import numpy as np
import polars as pl
import spacy
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from sklearn.metrics import cohen_kappa_score
from transformers import AutoModel, AutoTokenizer

sys.path.append(os.path.join(os.path.dirname(__file__), "feature_extraction"))
from dataset_preprocessing_for_BERT import preprocess_text


def quadratic_weighted_kappa(y_true, y_pred, labels=None):
    """Calculate Quadratic Weighted Kappa (QWK) score.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: List of possible labels (optional)

    Returns:
        QWK score between -1 and 1, where 1 is perfect agreement
    """
    if labels is None:
        labels = sorted(list(set(y_true + y_pred)))

    # Create confusion matrix
    n_labels = len(labels)
    label_to_idx = {label: idx for idx, label in enumerate(labels)}

    confusion_matrix = np.zeros((n_labels, n_labels))
    for true_label, pred_label in zip(y_true, y_pred):
        true_idx = label_to_idx[true_label]
        pred_idx = label_to_idx[pred_label]
        confusion_matrix[true_idx, pred_idx] += 1

    # Normalize to get observed agreement matrix
    total = confusion_matrix.sum()
    if total == 0:
        return 0.0

    observed_matrix = confusion_matrix / total

    # Calculate expected agreement matrix
    row_marginals = confusion_matrix.sum(axis=1) / total
    col_marginals = confusion_matrix.sum(axis=0) / total
    expected_matrix = np.outer(row_marginals, col_marginals)

    # Create quadratic weight matrix
    weights = np.zeros((n_labels, n_labels))
    for i in range(n_labels):
        for j in range(n_labels):
            weights[i, j] = (i - j) ** 2 / (n_labels - 1) ** 2

    # Calculate weighted agreements
    observed_agreement = np.sum(weights * observed_matrix)
    expected_agreement = np.sum(weights * expected_matrix)

    # Calculate QWK
    if expected_agreement == 0:
        return 0.0

    qwk = 1 - (observed_agreement / expected_agreement)
    return qwk


# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load spaCy Portuguese model for NLP preprocessing
try:
    nlp = spacy.load("pt_core_news_lg")
except OSError:
    logger.error(
        "Portuguese spaCy model not found. Please install with: python -m spacy download pt_core_news_lg"
    )
    sys.exit(1)


def preprocess_text(text: str) -> str:
    """Preprocess text using NLP techniques: lemmatization, stop word removal, cleaning.

    Args:
        text: Raw text to preprocess

    Returns:
        Preprocessed text string
    """
    if not text or text.strip() == "":
        return text

    # Basic cleaning
    # Remove extra whitespace and normalize
    text = re.sub(r"\s+", " ", text.strip())

    # Remove URLs, emails, and special patterns
    text = re.sub(
        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
        "",
        text,
    )
    text = re.sub(r"\S+@\S+", "", text)

    # Remove excessive punctuation (keep some for sentence structure)
    text = re.sub(r"[!]{2,}", "!", text)
    text = re.sub(r"[?]{2,}", "?", text)
    text = re.sub(r"[.]{3,}", "...", text)

    # Process with spaCy
    try:
        doc = nlp(text)

        # Extract lemmatized tokens, removing stop words and non-alphabetic tokens
        processed_tokens = []
        for token in doc:
            # Skip stop words, punctuation, spaces, and very short tokens
            if (
                not token.is_stop
                and not token.is_punct
                and not token.is_space
                and len(token.text.strip()) > 1
                and token.text.strip() not in string.punctuation
            ):
                # Use lemma if available, otherwise use original token
                lemma = token.lemma_.lower().strip()
                if (
                    lemma and lemma != "-PRON-"
                ):  # spaCy sometimes returns -PRON- for pronouns
                    processed_tokens.append(lemma)
                else:
                    processed_tokens.append(token.text.lower().strip())

        # Join tokens back into text
        processed_text = " ".join(processed_tokens)

        # If preprocessing resulted in empty text, return original
        if not processed_text.strip():
            return text

        return processed_text

    except Exception as e:
        logger.warning(f"Error in text preprocessing: {e}. Returning original text.")
        return text


def _print_device_info():
    """Print device information before starting inference."""
    print(f"\nDevice Detection:")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU AVAILABLE - Will use: {gpu_name} ({gpu_memory:.1f} GB)")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        import platform

        cpu_info = platform.processor() or "Unknown CPU"
        print(f"GPU NOT AVAILABLE - Will use CPU: {cpu_info}")
        print(f"PyTorch CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.device_count() > 0:
            print(f"Note: {torch.cuda.device_count()} GPU(s) detected but not usable")


class BERTimbauForC1Prediction(nn.Module):
    """BERTimbau model with regression head for C1 prediction.

    Args:
        model_name: HuggingFace model name (e.g., 'neuralmind/bert-base-portuguese-cased')
        num_labels: Number of output labels (1 for regression)
        dropout_prob: Dropout probability for regularization
    """

    def __init__(self, model_name: str, num_labels: int = 1, dropout_prob: float = 0.1):
        super().__init__()  # Modern Python super() syntax
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_prob)
        self.regressor = nn.Linear(self.bert.config.hidden_size, num_labels)

        # Initialize weights with Xavier/Glorot initialization for better training
        nn.init.xavier_uniform_(self.regressor.weight)
        nn.init.zeros_(self.regressor.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass through the model.

        Args:
            input_ids: Token IDs from tokenizer
            attention_mask: Attention mask for padding tokens
            labels: Target C1 scores (optional, for training)

        Returns:
            dictionary containing 'loss' and 'logits'
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Use the [CLS] token representation (first token)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)

        # Regression head
        logits = self.regressor(pooled_output)

        loss = None
        if labels is not None:
            # Use MSE loss for regression
            loss_fn = nn.MSELoss()
            loss = loss_fn(logits.squeeze(-1), labels)

        return {"loss": loss, "logits": logits}


class BERTimbauC1Predictor:
    """Wrapper class for making C1 predictions with the fine-tuned model.

    Args:
        model_path: Path to the saved model directory
        device: Device to run inference on (auto-detected if None)
    """

    def __init__(
        self,
        model_path: str | pathlib.Path,
        device: torch.device | None = None,
    ):
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model_path = pathlib.Path(model_path)

        # Load the saved model
        checkpoint = torch.load(
            self.model_path / "bertimbau_c1_model.pth",
            map_location=self.device,
            weights_only=False,
        )

        # Initialize model
        self.model = BERTimbauForC1Prediction(checkpoint["model_name"])
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path / "tokenizer")
        self.max_length = checkpoint["max_length"]

        # Load normalization parameters
        if "normalization" in checkpoint:
            self.normalization = checkpoint["normalization"]
            self.c1_min = self.normalization["c1_min"]
            self.c1_max = self.normalization["c1_max"]
            logger.info(f"Using normalization: C1 range [{self.c1_min}, {self.c1_max}]")
        else:
            # Fallback for old models
            self.c1_min = 0
            self.c1_max = 200
            logger.warning("No normalization parameters found, using defaults")

        logger.info(f"Model loaded successfully from {model_path}")

    def predict_single_essay(
        self, essay_text: str, use_preprocessing: bool = True
    ) -> float:
        """Make a C1 prediction for a single essay.

        Args:
            essay_text: The essay text to predict
            use_preprocessing: Whether to apply NLP preprocessing

        Returns:
            Predicted C1 score
        """
        # Transform essay if it's in the original format (Python list string)
        if isinstance(essay_text, str) and essay_text.startswith("["):
            try:
                essay_text = essay_line_to_single_utf8_string(essay_text)
            except:
                # If transformation fails, use the text as is
                pass

        # Apply NLP preprocessing if enabled
        if use_preprocessing:
            essay_text = preprocess_text(essay_text)

        # Tokenize the essay
        encoding = self.tokenizer(
            essay_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Move to device
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        # Make prediction
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            normalized_prediction = outputs["logits"].squeeze().cpu().item()

            # Denormalize the prediction back to original C1 range
            prediction = (
                normalized_prediction * (self.c1_max - self.c1_min) + self.c1_min
            )

            # Clamp to valid range
            prediction = max(self.c1_min, min(self.c1_max, prediction))

        return prediction

    def predict_batch(
        self, essay_texts: list[str], use_preprocessing: bool = True
    ) -> list[float]:
        """Make C1 predictions for a batch of essays.

        Args:
            essay_texts: List of essay texts to predict
            use_preprocessing: Whether to apply NLP preprocessing

        Returns:
            List of predicted C1 scores
        """
        predictions = []

        for essay_text in essay_texts:
            prediction = self.predict_single_essay(
                essay_text, use_preprocessing=use_preprocessing
            )
            predictions.append(prediction)

        return predictions

    def predict_with_confidence(
        self, essay_text: str, num_samples: int = 5, use_preprocessing: bool = True
    ) -> dict[str, float | list[float]]:
        """ "
        Args:
            essay_text: The essay text to predict
            num_samples: Number of forward passes for uncertainty estimation
            use_preprocessing: Whether to apply NLP preprocessing

        Returns:
            Dictionary with prediction, confidence std, and individual samples
        """
        # Enable dropout for uncertainty estimation
        self.model.train()

        predictions = []
        for _ in range(num_samples):
            pred = self.predict_single_essay(
                essay_text, use_preprocessing=use_preprocessing
            )
            predictions.append(pred)

        # Switch back to eval mode
        self.model.eval()

        mean_prediction = np.mean(predictions)
        std_prediction = np.std(predictions)

        return {
            "prediction": mean_prediction,
            "confidence_std": std_prediction,
            "samples": predictions,
        }


def demo_predictions():
    """Demo function showing how to use the predictor."""
    print("Starting demo predictions...")

    # Check device early
    _print_device_info()

    project_root = pathlib.Path(__file__).parent.parent
    model_path = project_root / "models" / "bertimbau_c1_finetuned"

    if not model_path.exists():
        print(f"Model not found at {model_path}. Please train the model first.")
        return

    print(f"Model found at: {model_path}")

    # Initialize predictor
    print("Initializing predictor...")
    try:
        predictor = BERTimbauC1Predictor(model_path)
        print("Predictor initialized successfully!")
    except Exception as e:
        print(f"Error initializing predictor: {e}")
        return

    # Example essays for demonstration
    example_essays = [
        "['Este ensaio aborda a importância da educação na sociedade brasileira.', 'A educação é fundamental para o desenvolvimento de um país.', 'Investimentos em educação geram retornos significativos para a sociedade.']",
        "['O meio ambiente é uma questão crucial nos dias atuais.', 'É necessário desenvolver políticas sustentáveis para preservar nossos recursos naturais.', 'A conscientização ambiental deve começar nas escolas.']",
        "A tecnologia tem transformado a maneira como vivemos e trabalhamos. É importante que nos adaptemos às mudanças tecnológicas para não ficarmos para trás.",
    ]

    print("Making predictions for example essays:")

    for i, essay in enumerate(example_essays, 1):
        print(f"Processing essay {i}...")
        prediction = predictor.predict_single_essay(essay)
        print(f"Essay {i} - C1 prediction: {prediction:.4f}")

        # Also get prediction with confidence
        print(f"Getting confidence for essay {i}...")
        confidence_result = predictor.predict_with_confidence(essay)
        print(
            f"Essay {i} - With confidence: {confidence_result['prediction']:.4f} ± {confidence_result['confidence_std']:.4f}"
        )

    # Batch prediction example
    print("\nBatch prediction:")
    batch_predictions = predictor.predict_batch(example_essays)
    for i, pred in enumerate(batch_predictions, 1):
        print(f"Essay {i} - Batch prediction: {pred:.4f}")


def predict_from_csv(max_samples: int | None = None):
    """Function to make predictions on a CSV file."""
    print("Starting CSV predictions...")

    # Check device early
    _print_device_info()

    project_root = pathlib.Path(__file__).parent.parent
    model_path = project_root / "models" / "bertimbau_c1_finetuned"
    csv_path = (
        project_root
        / "generated_datasets"
        / "extended_essay-br_preprocessed_for_BERT.csv"
    )

    if not model_path.exists():
        print(f"Model not found at {model_path}. Please train the model first.")
        return

    print(f"Loading data from: {csv_path}")
    relevant_columns = ["essay_as_single_utf8_string", "c1"]
    SAMPLE_SIZE_UPPER_BOUND = 2**31 - 1
    df = (
        pl.scan_csv(csv_path)
        .select(relevant_columns)
        .head(max_samples if max_samples is not None else SAMPLE_SIZE_UPPER_BOUND)
        .collect()
    )
    print(f"Loaded {len(df)} rows from CSV")

    for idx, essay in enumerate(df["essay_as_single_utf8_string"]):
        print(f"{idx}:\n{essay}\n\n")

    print(
        f"[DEBUG] Printing first {max_samples if max_samples else len(df)} rows of the CSV:"
    )
    print(df.head(max_samples if max_samples else len(df)))

    # Initialize predictor
    print("Initializing predictor...")
    predictor = BERTimbauC1Predictor(model_path)

    print("Making predictions on sample essays from the dataset:")

    predictions = []
    actual_c1_values = []
    for i, row in enumerate(df.iter_rows(named=True), 1):
        if row["essay_as_single_utf8_string"] is not None:
            print(f"Processing essay {i}...")
            prediction = predictor.predict_single_essay(
                row["essay_as_single_utf8_string"]
            )
            predictions.append(prediction)
            actual_c1_values.append(row["c1"])

            print(f"Essay {i} - Actual C1: {row['c1']}, Predicted C1: {prediction:.4f}")

    # Calculate some basic metrics
    if len(predictions) > 0:
        mse = np.mean(
            [
                (pred - actual) ** 2
                for pred, actual in zip(predictions, actual_c1_values)
            ]
        )
        mae = np.mean(
            [abs(pred - actual) for pred, actual in zip(predictions, actual_c1_values)]
        )

        # Calculate Cohen's Kappa - round predictions and true labels to nearest C1 score levels
        def round_to_c1_levels(scores):
            """Round scores to nearest valid C1 levels (0, 40, 80, 120, 160, 200)"""
            c1_levels = [0, 40, 80, 120, 160, 200]
            rounded = []
            for score in scores:
                # Clamp to valid range first
                score = max(0, min(200, score))
                # Find closest C1 level
                closest_level = min(c1_levels, key=lambda x: abs(x - score))
                rounded.append(closest_level)
            return rounded

        true_labels_rounded = round_to_c1_levels(actual_c1_values)
        predictions_rounded = round_to_c1_levels(predictions)

        try:
            kappa = cohen_kappa_score(true_labels_rounded, predictions_rounded)
        except Exception as e:
            # If kappa calculation fails (e.g., only one class), set to 0
            kappa = 0.0

        # Calculate Quadratic Weighted Kappa
        try:
            qwk = quadratic_weighted_kappa(
                true_labels_rounded,
                predictions_rounded,
                labels=[0, 40, 80, 120, 160, 200],
            )
        except Exception as e:
            qwk = 0.0

        # Calculate Pearson correlation
        try:
            pearson_corr, pearson_p = pearsonr(actual_c1_values, predictions)
            # Handle case where correlation is NaN
            if np.isnan(pearson_corr):
                pearson_corr = 0.0
        except Exception as e:
            pearson_corr = 0.0

        print(f"\nSample evaluation metrics:")
        print(f"MSE: {mse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"Cohen's Kappa: {kappa:.4f}")
        print(f"Quadratic Weighted Kappa (QWK): {qwk:.4f}")
        print(f"Pearson Correlation: {pearson_corr:.4f}")


def main():
    """Main function with different usage examples."""
    import argparse

    parser = argparse.ArgumentParser(description="BERTimbau C1 Prediction Inference")
    parser.add_argument(
        "--mode",
        choices=["demo", "csv"],
        default="demo",
        help="Mode to run: 'demo' for example predictions, 'csv' for CSV file predictions",
    )
    parser.add_argument("--text", type=str, help="Single text to predict")

    args = parser.parse_args()

    if args.text:
        # Single text prediction
        print("Single text prediction mode...")

        # Check device early
        _print_device_info()

        project_root = pathlib.Path(__file__).parent.parent
        model_path = project_root / "models" / "bertimbau_c1_finetuned"

        if not model_path.exists():
            logger.error(
                f"Model not found at {model_path}. Please train the model first."
            )
            return

        predictor = BERTimbauC1Predictor(model_path)
        prediction = predictor.predict_single_essay(args.text)
        print(f"C1 Prediction: {prediction:.4f}")

    elif args.mode == "demo":
        demo_predictions()
    elif args.mode == "csv":
        predict_from_csv(100)


if __name__ == "__main__":
    main()
