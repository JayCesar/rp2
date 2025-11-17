"""Metrics computation and target scaling utilities

Provides comprehensive evaluation metrics for regression tasks and target scaling.
"""

import logging
from typing import Literal

import numpy as np
import scipy.stats
import sklearn.metrics
import torch

logger = logging.getLogger(__name__)

# Score constants for ENEM essays
SCORE_MIN = 0
SCORE_MAX = 200
SCORE_STEP = 40
VALID_SCORES = [0, 40, 80, 120, 160, 200]


def snap_to_step(score: float, step: int = SCORE_STEP) -> int:
    """Snap score to nearest step increment for evaluation.
    
    Args:
        score: Continuous score
        step: Step size (default: 40 for ENEM)
        
    Returns:
        Score snapped to nearest step
        
    Example:
        >>> snap_to_step(125, 40)
        120
    """
    return int(round(score / step) * step)


def round_to_valid_scores(scores: list[float]) -> list[int]:
    """Round scores to nearest valid ENEM score levels.
    
    Args:
        scores: List of continuous scores
        
    Returns:
        List of scores rounded to [0, 40, 80, 120, 160, 200]
        
    Example:
        >>> round_to_valid_scores([45, 125, 185])
        [40, 120, 200]
    """
    rounded = []
    for score in scores:
        # Clamp to valid range first
        score = max(SCORE_MIN, min(SCORE_MAX, score))
        # Find closest valid level
        closest = min(VALID_SCORES, key=lambda x: abs(x - score))
        rounded.append(closest)
    return rounded


def quadratic_weighted_kappa(
    y_true: list[int],
    y_pred: list[int],
    labels: list[int] | None = None
) -> float:
    """Calculate Quadratic Weighted Kappa (QWK) score.
    
    QWK measures inter-rater agreement with quadratic weights,
    where disagreements are penalized quadratically.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: List of possible labels (auto-detected if None)
        
    Returns:
        QWK score between -1 and 1, where 1 is perfect agreement
        
    Example:
        >>> qwk = quadratic_weighted_kappa([0, 40, 80], [0, 40, 120])
        >>> print(f"QWK: {qwk:.3f}")
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
    
    return 1 - (observed_agreement / expected_agreement)


class TargetScaler:
    """Target scaler for normalization/standardization of scores.
    
    Supports three modes:
    - 'none': No scaling
    - 'minmax': Scale to [0, 1] range
    - 'standard': Z-score normalization
    
    Example:
        >>> scaler = TargetScaler("minmax")
        >>> scaler.fit(np.array([0, 40, 80, 120, 160, 200]))
        >>> scaled = scaler.transform(np.array([40, 120]))
        >>> original = scaler.inverse_transform(scaled)
    """
    
    def __init__(self, mode: Literal["none", "minmax", "standard"] = "none"):
        self.mode = mode
        self.fitted = False
        self.min_val = None
        self.max_val = None
        self.mean_val = None
        self.std_val = None
    
    def fit(self, targets: np.ndarray) -> "TargetScaler":
        """Fit the scaler to target values."""
        if self.mode == "minmax":
            self.min_val = targets.min()
            self.max_val = targets.max()
        elif self.mode == "standard":
            self.mean_val = targets.mean()
            self.std_val = targets.std()
        
        self.fitted = True
        return self
    
    def transform(self, targets: np.ndarray) -> np.ndarray:
        """Transform targets using fitted scaler."""
        if not self.fitted and self.mode != "none":
            raise ValueError("Scaler must be fitted before transform")
        
        if self.mode == "none":
            return targets
        elif self.mode == "minmax":
            return (targets - self.min_val) / (self.max_val - self.min_val + 1e-8)
        elif self.mode == "standard":
            return (targets - self.mean_val) / (self.std_val + 1e-8)
        
        return targets
    
    def inverse_transform(self, targets: np.ndarray) -> np.ndarray:
        """Inverse transform targets to original scale."""
        if not self.fitted and self.mode != "none":
            raise ValueError("Scaler must be fitted before inverse_transform")
        
        if self.mode == "none":
            return targets
        elif self.mode == "minmax":
            return targets * (self.max_val - self.min_val) + self.min_val
        elif self.mode == "standard":
            return targets * self.std_val + self.mean_val
        
        return targets
    
    def state_dict(self) -> dict:
        """Get state dictionary for serialization."""
        return {
            "mode": self.mode,
            "fitted": self.fitted,
            "min_val": self.min_val,
            "max_val": self.max_val,
            "mean_val": self.mean_val,
            "std_val": self.std_val,
        }
    
    def load_state_dict(self, state: dict) -> None:
        """Load state from dictionary."""
        self.mode = state["mode"]
        self.fitted = state["fitted"]
        self.min_val = state.get("min_val")
        self.max_val = state.get("max_val")
        self.mean_val = state.get("mean_val")
        self.std_val = state.get("std_val")


class MetricsAccumulator:
    """Running metrics accumulator for regression evaluation.
    
    Accumulates predictions and targets across batches,
    then computes comprehensive regression metrics.
    
    Example:
        >>> metrics = MetricsAccumulator()
        >>> for batch in data_loader:
        ...     preds, targets = model(batch), batch["targets"]
        ...     metrics.update(preds, targets, batch["ids"])
        >>> results = metrics.compute_metrics()
        >>> print(f"MAE: {results['mae']:.2f}, RMSE: {results['rmse']:.2f}")
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self) -> None:
        """Reset accumulated predictions and targets."""
        self.predictions = []
        self.targets = []
        self.ids = []
    
    def update(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        ids: list[str]
    ) -> None:
        """Update with batch predictions and targets.
        
        Args:
            preds: Predicted scores [batch_size]
            targets: True scores [batch_size]
            ids: Sample IDs for tracking
        """
        self.predictions.extend(preds.detach().cpu().numpy())
        self.targets.extend(targets.detach().cpu().numpy())
        self.ids.extend(ids)
    
    def compute_metrics(
        self,
        target_scaler: TargetScaler | None = None
    ) -> dict[str, float]:
        """Compute all regression metrics.
        
        Metrics include:
        - MAE, RMSE, R²
        - Pearson correlation
        - Cohen's Kappa and QWK
        - Step accuracy (40-point grid)
        
        Args:
            target_scaler: Optional scaler for inverse transform
            
        Returns:
            Dictionary of metric names to values
        """
        if not self.predictions:
            return {}
        
        preds = np.array(self.predictions)
        targets = np.array(self.targets)
        
        # Inverse transform if scaler was used
        if target_scaler and target_scaler.fitted and target_scaler.mode != "none":
            preds = target_scaler.inverse_transform(preds)
            targets = target_scaler.inverse_transform(targets)
        
        # Clamp predictions to valid range
        preds_clamped = np.clip(preds, SCORE_MIN, SCORE_MAX)
        
        # Standard regression metrics
        mae = np.mean(np.abs(preds_clamped - targets))
        mse = np.mean((preds_clamped - targets) ** 2)
        rmse = np.sqrt(mse)
        
        # R-squared
        ss_res = np.sum((targets - preds_clamped) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        # Step-aligned metrics
        preds_snapped = np.array([snap_to_step(p) for p in preds_clamped])
        targets_snapped = np.array([snap_to_step(t) for t in targets])
        step_accuracy = np.mean(preds_snapped == targets_snapped)
        mae_step = np.mean(np.abs(preds_snapped - targets_snapped))
        
        # Round to valid scores for kappa calculations
        true_labels = round_to_valid_scores(targets.tolist())
        pred_labels = round_to_valid_scores(preds_clamped.tolist())
        
        # Cohen's Kappa
        kappa = sklearn.metrics.cohen_kappa_score(true_labels, pred_labels)
        
        # Quadratic Weighted Kappa
        qwk = quadratic_weighted_kappa(true_labels, pred_labels, labels=VALID_SCORES)
        
        # Pearson correlation
        try:
            pearson_corr, _ = scipy.stats.pearsonr(targets, preds_clamped)
            if np.isnan(pearson_corr):
                pearson_corr = 0.0
        except Exception:
            pearson_corr = 0.0
        
        return {
            "loss": float(mse),
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
            "kappa": float(kappa),
            "qwk": float(qwk),
            "pearson_corr": float(pearson_corr),
            "step_accuracy": float(step_accuracy),
            "mae_step": float(mae_step),
            "count": len(preds),
        }
    
    def get_predictions_df(
        self,
        target_scaler: TargetScaler | None = None
    ) -> list[dict[str, str | float | int]]:
        """Get predictions as list of dicts for DataFrame creation.
        
        Args:
            target_scaler: Optional scaler for inverse transform
            
        Returns:
            List of dictionaries with id, target, pred, pred_snapped
        """
        if not self.predictions:
            return []
        
        preds = np.array(self.predictions)
        targets = np.array(self.targets)
        
        # Inverse transform if scaler was used
        if target_scaler and target_scaler.fitted and target_scaler.mode != "none":
            preds = target_scaler.inverse_transform(preds)
            targets = target_scaler.inverse_transform(targets)
        
        # Clamp and snap predictions
        preds_clamped = np.clip(preds, SCORE_MIN, SCORE_MAX)
        preds_snapped = np.array([snap_to_step(p) for p in preds_clamped])
        
        return [
            {
                "id": id_,
                "target": float(target),
                "pred": float(pred),
                "pred_snapped": int(pred_snap),
            }
            for id_, target, pred, pred_snap in zip(
                self.ids, targets, preds_clamped, preds_snapped
            )
        ]
