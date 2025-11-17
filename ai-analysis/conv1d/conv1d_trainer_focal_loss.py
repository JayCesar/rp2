"""Conv1D Trainer with Focal Loss for Classification

This module provides a Trainer class for Conv1D classification models using
FocalLoss on 6-class C1 score classification {0, 40, 80, 120, 160, 200}.

Mirrors the structure of BLSTM trainer but adapted for Conv1D:
- Uses Conv1DClassifier instead of Conv1DRegressor
- Computes FocalLoss on class indices
- Converts predictions back to score space for metrics
- Tracks best model by validation MAE (like regression)
"""

from __future__ import annotations

import pathlib
import time
from typing import Optional

import torch
import torch.nn as nn
import tqdm
from torch.utils.data import DataLoader

# Import common utilities
try:
    from ..common import MetricsAccumulator, TargetScaler, ensure_dir
    from .conv1d import ModelConfig, SerializationConfig, TrainConfig
except ImportError:
    # Fallback for direct execution
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
    from common import MetricsAccumulator, TargetScaler, ensure_dir  # type: ignore
    from conv1d import ModelConfig, SerializationConfig, TrainConfig  # type: ignore

# Import focal loss-specific components and mappings
try:
    from .conv1d_focal_loss import (
        Conv1DClassifier,
        scores_to_class_indices,
        logits_to_scores,
    )
except ImportError:
    from conv1d_focal_loss import (
        Conv1DClassifier,
        scores_to_class_indices,
        logits_to_scores,
    )

# Import FocalLoss
try:
    from ..common.focal_loss import FocalLoss
except ImportError:
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
    from common.focal_loss import FocalLoss  # type: ignore

# Shared alpha helper
try:
    from ..common.class_frequencies import calculate_alpha_from_frequency
except ImportError:
    import sys as _sys

    _sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
    from common.class_frequencies import calculate_alpha_from_frequency  # type: ignore


class Conv1DFLTrainer:
    """Trainer for Conv1D classification with FocalLoss."""

    def __init__(
        self,
        model: Conv1DClassifier,
        train_loader: DataLoader,
        val_loader: DataLoader,
        model_config: ModelConfig,
        train_config: TrainConfig,
        serialization_config: SerializationConfig,
        target_scaler: Optional[TargetScaler] = None,
        device: Optional[torch.device] = None,
        class_frequencies: Optional[list[int]] = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model_config = model_config
        self.train_config = train_config
        self.serialization_config = serialization_config
        self.target_scaler = target_scaler

        # Device setup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.model = self.model.to(self.device)

        # Optimizer
        self.optimizer = self._setup_optimizer()

        # AMP
        self.use_amp = train_config.use_amp and self.device.type == "cuda"
        # Use torch.amp.GradScaler for parity with conv1d/trainer.py
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # Loss with inverse frequency alpha weights
        if class_frequencies is not None:
            alpha = calculate_alpha_from_frequency(class_frequencies)
        else:
            alpha = [0.25] * 6  # Fallback: uniform weights
        
        self.criterion = FocalLoss(
            gamma=2,
            alpha=alpha,
            task_type="multi-class",
            num_classes=6,
        )

        # State
        self.current_epoch = 0
        self.best_val_mae = float("inf")
        self.patience_counter = 0
        self.training_history: list[dict[str, float]] = []
        self.best_val_predictions: list[dict[str, str | float | int]] = []

        # Output dir
        ensure_dir(self.serialization_config.output_dir)

    def _setup_optimizer(self) -> torch.optim.Optimizer:
        if self.train_config.optimizer == "adamw":
            kwargs = {
                "lr": self.train_config.lr,
                "weight_decay": self.train_config.weight_decay,
            }
            if self.device.type == "cuda":
                try:
                    return torch.optim.AdamW(
                        self.model.parameters(), fused=True, **kwargs
                    )
                except Exception:
                    return torch.optim.AdamW(self.model.parameters(), **kwargs)
            else:
                return torch.optim.AdamW(self.model.parameters(), **kwargs)
        raise ValueError(f"Unknown optimizer: {self.train_config.optimizer}")

    def _train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        total_samples = 0

        pbar = tqdm.tqdm(
            self.train_loader,
            desc=f"Training Epoch {self.current_epoch + 1}",
            leave=False,
        )

        for batch in pbar:
            tokens = batch["tokens"].to(self.device, non_blocking=True)
            lengths = batch["lengths"]
            targets = batch["targets"].to(self.device, non_blocking=True)

            # For features mode, lengths may be None; pass as-is
            if lengths is not None:
                lengths = lengths.to(self.device, non_blocking=True)

            # Convert scores → class indices
            target_classes = scores_to_class_indices(targets).long()

            bs = tokens.shape[0]
            total_samples += bs

            self.optimizer.zero_grad(set_to_none=True)

            if self.use_amp:
                with torch.autocast(
                    "cuda", enabled=True, dtype=self.train_config.amp_dtype
                ):
                    logits = self.model(tokens, lengths)
                    loss = self.criterion(logits, target_classes)
                self.scaler.scale(loss).backward()
                if self.train_config.grad_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.train_config.grad_clip_norm
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(tokens, lengths)
                loss = self.criterion(logits, target_classes)
                loss.backward()
                if self.train_config.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.train_config.grad_clip_norm
                    )
                self.optimizer.step()

            total_loss += loss.item() * bs
            pbar.set_postfix({"loss": loss.item()})

        return total_loss / max(1, total_samples)

    def _validate_epoch(
        self,
    ) -> tuple[float, dict[str, float], list[dict[str, str | float | int]]]:
        self.model.eval()
        metrics = MetricsAccumulator()
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in self.val_loader:
                tokens = batch["tokens"].to(self.device, non_blocking=True)
                lengths = batch["lengths"]
                targets = batch["targets"]
                ids = batch["ids"]

                if lengths is not None:
                    lengths = lengths.to(self.device, non_blocking=True)

                bs = tokens.shape[0]
                total_samples += bs

                if self.use_amp:
                    with torch.autocast(
                        "cuda", enabled=True, dtype=self.train_config.amp_dtype
                    ):
                        logits = self.model(tokens, lengths)
                        pred_scores = logits_to_scores(logits)
                else:
                    logits = self.model(tokens, lengths)
                    pred_scores = logits_to_scores(logits)

                # Validation focal loss on class indices
                target_classes = scores_to_class_indices(targets.to(self.device)).long()
                loss = self.criterion(logits, target_classes)
                total_loss += loss.item() * bs

                # Metrics on score space
                metrics.update(pred_scores.cpu().float(), targets, ids)

        computed = metrics.compute_metrics(self.target_scaler)
        preds_df = metrics.get_predictions_df(self.target_scaler)
        avg_loss = total_loss / max(1, total_samples)
        return avg_loss, computed, preds_df

    def _save_checkpoint(
        self, is_best: bool = False, metrics: dict[str, float] | None = None
    ) -> None:
        ckpt = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict() if self.use_amp else None,
            "model_config": self.model_config.to_dict(),
            "train_config": self.train_config.to_dict(),
            "target_scaler_state": self.target_scaler.state_dict()
            if self.target_scaler
            else None,
            "best_val_mae": self.best_val_mae,
            "training_history": self.training_history,
            "metrics": metrics or {},
            "loss_type": "FocalLoss",
            "focal_gamma": getattr(self.criterion, "gamma", None),
        }

        latest = pathlib.Path(self.serialization_config.output_dir) / "latest.pt"
        torch.save(ckpt, latest)

        if is_best:
            best = pathlib.Path(self.serialization_config.output_dir) / "best.pt"
            torch.save(ckpt, best)

    def train(self) -> dict[str, float]:
        start_time = time.time()
        best_metrics: dict[str, float] = {}

        try:
            for epoch in range(self.train_config.epochs):
                self.current_epoch = epoch
                epoch_start = time.time()

                train_loss = self._train_epoch()
                val_loss, val_metrics, val_predictions = self._validate_epoch()

                val_mae = val_metrics["mae"]
                is_best = val_mae < self.best_val_mae

                if is_best:
                    self.best_val_mae = val_mae
                    self.patience_counter = 0
                    best_metrics = val_metrics.copy()
                    self.best_val_predictions = val_predictions
                else:
                    self.patience_counter += 1

                self._save_checkpoint(is_best=is_best, metrics=val_metrics)

                epoch_time = time.time() - epoch_start
                hist = {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_mae": val_mae,
                    "val_rmse": val_metrics["rmse"],
                    "kappa": val_metrics["kappa"],
                    "qwk": val_metrics["qwk"],
                    "r2": val_metrics["r2"],
                    "pearson_corr": val_metrics["pearson_corr"],
                    "step_accuracy": val_metrics["step_accuracy"],
                    "learning_rate": self.optimizer.param_groups[0]["lr"],
                    "epoch_time": epoch_time,
                }
                self.training_history.append(hist)

                # Early stopping
                if self.patience_counter >= self.train_config.early_stopping_patience:
                    break

            total_time = time.time() - start_time
            return best_metrics

        except KeyboardInterrupt:
            return best_metrics
        except Exception:
            raise
