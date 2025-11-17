"""BiLSTM Trainer with FocalLoss for Classification

This mirrors :mod:`trainer_cross_entropy_loss` but uses FocalLoss instead of
CrossEntropyLoss and is designed to plug into the shared gamma-search helper.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import torch
import tqdm
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
from torch.utils.data import DataLoader

# Import from main BLSTM implementation
from blstm import (
    MetricsAccumulator,
    ModelConfig,
    SerializationConfig,
    TargetScaler,
    TrainConfig,
    ensure_dir,
)

# BLSTM classification components / mappings
from blstm_focal_loss import (
    NUM_CLASSES,
    BiLSTMClassifier,
    logits_to_scores,
    scores_to_class_indices,
)

# FocalLoss implementation (shared with Conv1D)
try:
    from ..common.focal_loss import FocalLoss
except ImportError:  # pragma: no cover - direct script execution fallback
    import sys as _sys
    import pathlib as _pathlib

    _sys.path.insert(0, str(_pathlib.Path(__file__).parent.parent))
    from common.focal_loss import FocalLoss  # type: ignore

# Optional alpha helper (for single-run usage); gamma-search overrides criterion
try:
    from ..common.class_frequencies import calculate_alpha_from_frequency
except ImportError:  # pragma: no cover - direct script execution fallback
    import sys as _sys2
    import pathlib as _pathlib2

    _sys2.path.insert(0, str(_pathlib2.Path(__file__).parent.parent))
    from common.class_frequencies import calculate_alpha_from_frequency  # type: ignore


logger = logging.getLogger(__name__)


class BiLSTMFLTrainer:
    """Trainer for BiLSTM classification with FocalLoss.

    The interface mirrors :class:`BiLSTMCETrainer` so that higher-level code
    (including the gamma-sweep helper) can treat both trainers uniformly.
    """

    def __init__(
        self,
        model: BiLSTMClassifier,
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
        logger.info(f"Training on device: {self.device}")

        # Optimizer & scheduler
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()

        # Mixed precision
        self.use_amp = train_config.use_amp and self.device.type == "cuda"
        self.scaler = torch.GradScaler("cuda", enabled=self.use_amp)

        # Initial FocalLoss (gamma/alpha may be overridden by gamma-search)
        if class_frequencies is not None:
            alpha = calculate_alpha_from_frequency(class_frequencies)
        else:
            alpha = None

        self.criterion: torch.nn.Module = FocalLoss(
            gamma=2.0,
            alpha=alpha,
            task_type="multi-class",
            num_classes=NUM_CLASSES,
        )
        logger.info("Using FocalLoss for 6-class classification")

        # Training state
        self.current_epoch = 0
        self.best_val_mae = float("inf")
        self.patience_counter = 0
        self.training_history: list[dict[str, float]] = []
        self.best_val_predictions: list[dict[str, str | float | int]] = []

        # Ensure output directory exists
        ensure_dir(self.serialization_config.output_dir)

    # -----------------------
    # Setup helpers
    # -----------------------
    def _setup_optimizer(self) -> torch.optim.Optimizer:
        if self.train_config.optimizer == "adamw":
            use_fused = self.device.type == "cuda" and hasattr(
                torch.optim.AdamW, "__init__"
            )
            kwargs = {
                "lr": self.train_config.lr,
                "weight_decay": self.train_config.weight_decay,
            }
            if use_fused:
                try:
                    opt = torch.optim.AdamW(self.model.parameters(), fused=True, **kwargs)
                    logger.info("Using fused AdamW optimizer")
                except Exception:  # pragma: no cover - fallback
                    opt = torch.optim.AdamW(self.model.parameters(), **kwargs)
                    logger.info("Using standard AdamW optimizer")
            else:
                opt = torch.optim.AdamW(self.model.parameters(), **kwargs)
                logger.info("Using standard AdamW optimizer")
            return opt
        raise ValueError(f"Unknown optimizer: {self.train_config.optimizer}")

    def _setup_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler | None:
        if self.train_config.scheduler == "plateau":
            sched = ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=self.train_config.plateau_factor,
                patience=self.train_config.plateau_patience,
            )
            logger.info("Using ReduceLROnPlateau scheduler")
            return sched
        if self.train_config.scheduler == "onecycle":
            total_steps = len(self.train_loader) * self.train_config.epochs
            sched = OneCycleLR(
                self.optimizer,
                max_lr=self.train_config.lr,
                total_steps=total_steps,
                pct_start=self.train_config.onecycle_pct_start,
            )
            logger.info(f"Using OneCycleLR scheduler (total_steps={total_steps})")
            return sched
        if self.train_config.scheduler == "none":
            logger.info("No scheduler used")
            return None
        raise ValueError(f"Unknown scheduler: {self.train_config.scheduler}")

    # -----------------------
    # Epoch loops
    # -----------------------
    def _train_epoch(self) -> float:
        """Train for one epoch and return average loss."""

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

            # Handle feature mode: lengths may be None
            if lengths is None:
                tokens = tokens.unsqueeze(1)
                lengths = torch.ones(
                    tokens.shape[0], dtype=torch.long, device=self.device
                )
            else:
                lengths = lengths.to(self.device, non_blocking=True)

            # Convert scores to class indices for FocalLoss
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

            if isinstance(self.scheduler, OneCycleLR):
                self.scheduler.step()

        return total_loss / max(1, total_samples)

    def _validate_epoch(
        self,
    ) -> tuple[float, dict[str, float], list[dict[str, str | float | int]]]:
        """Validate for one epoch and return loss, metrics, and predictions."""

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

                if lengths is None:
                    tokens = tokens.unsqueeze(1)
                    lengths = torch.ones(
                        tokens.shape[0], dtype=torch.long, device=self.device
                    )
                else:
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
                target_classes = scores_to_class_indices(
                    targets.to(self.device)
                ).long()
                loss = self.criterion(logits, target_classes)
                total_loss += loss.item() * bs

                metrics.update(pred_scores.cpu(), targets, ids)

        computed = metrics.compute_metrics(self.target_scaler)
        preds = metrics.get_predictions_df(self.target_scaler)
        avg_loss = total_loss / max(1, total_samples)
        return avg_loss, computed, preds

    # -----------------------
    # Checkpointing / public API
    # -----------------------
    def _save_checkpoint(
        self, is_best: bool = False, metrics: dict[str, float] | None = None
    ) -> None:
        """Save model checkpoint to ``latest.pt`` and, if best, to ``best.pt``."""

        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict()
            if self.scheduler
            else None,
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

        latest_path = Path(self.serialization_config.output_dir) / "latest.pt"
        torch.save(checkpoint, latest_path)

        if is_best:
            best_path = Path(self.serialization_config.output_dir) / "best.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"New best model saved with val_mae: {self.best_val_mae:.6f}")

        logger.info(f"Checkpoint saved: {latest_path}")

    def train(self) -> dict[str, float]:
        """Train the model and return best validation metrics."""

        logger.info(f"Starting FL training for {self.train_config.epochs} epochs")
        start_time = time.time()

        best_metrics: dict[str, float] = {}

        try:
            for epoch in range(self.train_config.epochs):
                self.current_epoch = epoch
                epoch_start_time = time.time()

                train_loss = self._train_epoch()
                val_loss, val_metrics, val_predictions = self._validate_epoch()

                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["mae"])

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

                epoch_time = time.time() - epoch_start_time
                history_entry = {
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
                self.training_history.append(history_entry)

                current_epoch = epoch + 1
                logger.info(f"Epoch {current_epoch}/{self.train_config.epochs}")
                logger.info(f"Epoch {current_epoch} - Train Loss: {train_loss:.4f}")
                logger.info(
                    f"Epoch {current_epoch} - Val Loss: {val_metrics['loss']:.4f}"
                )
                logger.info(
                    f"Epoch {current_epoch} - Val MAE: {val_metrics['mae']:.4f}"
                )
                logger.info(
                    f"Epoch {current_epoch} - Val RMSE: {val_metrics['rmse']:.4f}"
                )
                logger.info(
                    f"Epoch {current_epoch} - Val R²: {val_metrics['r2']:.4f}"
                )
                logger.info(
                    f"Epoch {current_epoch} - Val Kappa: {val_metrics['kappa']:.4f}"
                )
                logger.info(
                    f"Epoch {current_epoch} - Val QWK: {val_metrics['qwk']:.4f}"
                )
                logger.info(
                    f"Epoch {current_epoch} - Val Pearson: {val_metrics['pearson_corr']:.4f}"
                )
                logger.info(
                    f"Epoch {current_epoch} - Val Step Acc: {val_metrics.get('step_accuracy', 0.0):.4f}"
                )
                logger.info(f"Epoch {current_epoch} - Time: {epoch_time:.1f}s")

                if (
                    self.patience_counter
                    >= self.train_config.early_stopping_patience
                ):
                    logger.info(
                        "Early stopping triggered after %d epochs (patience %d/%d)",
                        epoch + 1,
                        self.patience_counter,
                        self.train_config.early_stopping_patience,
                    )
                    break

            total_time = time.time() - start_time
            logger.info(f"FL training completed in {total_time:.1f}s")
            logger.info(f"Best validation MAE: {self.best_val_mae:.6f}")
            return best_metrics

        except KeyboardInterrupt:  # pragma: no cover - manual interruption
            logger.info("Training interrupted by user")
            return best_metrics
        except Exception as exc:  # pragma: no cover - surfaced to caller
            logger.error(f"Training failed with error: {exc}")
            raise
