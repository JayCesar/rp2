#!/usr/bin/env python3
"""BiLSTM Training on Vectorized Essays with FocalLoss (gamma sweep)

Runs a FocalLoss gamma sweep for Component 1 BLSTM classification on
vectorized essays (token embeddings), mirroring the Conv1D focal-loss script.

If the fused BLSTM parquet dataset is missing, this script will attempt to
create it automatically via ``generated_datasets/fuse_blstm_parquet.py``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import polars as pl
import torch

# Local BLSTM imports
sys.path.append(".")
from blstm import (  # type: ignore
    ModelConfig,
    SerializationConfig,
    TargetScaler,
    TrainConfig,
    get_device,
    set_seed,
)

from blstm_cross_entropy_loss import (  # type: ignore
    BiLSTMClassifier,
)

from trainer_focal_loss import (  # type: ignore
    BiLSTMFLTrainer,
)

# Common utilities
sys.path.append(str(Path(__file__).parent.parent))
from common import (  # type: ignore
    DEFAULT_GAMMA_VALUES,
    EssayDataset,
    create_data_loader,
    metric_key_for_selection,
    run_gamma_search,
    split_dataset,
)
from common.class_frequencies import (  # type: ignore
    calculate_alpha_from_frequency,
    get_class_frequencies,
    print_class_distribution,
)


project_root = Path(__file__).parent.parent.parent
assert project_root.name == "rp2"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _ensure_fused_blstm_parquet() -> Path:
    """Ensure fused BLSTM parquet exists, auto-running fuse script if needed."""

    fused = project_root / "generated_datasets" / "extended_essay-br_preprocessed_for_BLSTM.parquet"
    if fused.exists():
        return fused

    # Try to auto-fuse using generated_datasets/fuse_blstm_parquet.py
    logger.info("Fused BLSTM parquet not found; attempting automatic fusion...")
    sys.path.append(str(project_root / "generated_datasets"))
    try:
        import fuse_blstm_parquet  # type: ignore

        rc = fuse_blstm_parquet.fuse()
    except Exception as exc:  # pragma: no cover - surfaced to user
        raise SystemExit(
            "Failed to import or execute fuse_blstm_parquet.py for BLSTM dataset "
            f"fusion: {exc!r}"
        ) from exc

    if rc != 0:
        raise SystemExit(
            "Automatic fusion of extended_essay-br_preprocessed_for_BLSTM failed. "
            "Please run generated_datasets/fuse_blstm_parquet.py manually and "
            "inspect its output."
        )

    if not fused.exists():
        raise SystemExit(
            "Fuse script completed but fused BLSTM parquet is still missing: "
            f"{fused}"
        )

    logger.info("[ok] Fused BLSTM parquet created: %s", fused)
    return fused


def _load_vectorized_dataframe(max_samples: Optional[int]) -> pl.DataFrame:
    """Load full vectorized-essays dataset for BLSTM FocalLoss training/search."""

    parquet_file = _ensure_fused_blstm_parquet()
    json_file = (
        project_root
        / "generated_datasets"
        / "extended_essay-br_preprocessed_for_BLSTM.json"
    )

    data_file: Optional[Path]
    if parquet_file.exists():
        data_file = parquet_file
        logger.info(f"[gamma-search] Using fused BLSTM parquet file: {data_file}")
    elif json_file.exists():
        data_file = json_file
        logger.info(f"[gamma-search] Using JSON file: {data_file}")
    else:
        raise FileNotFoundError(
            f"Data file not found. Looked for: {parquet_file} and {json_file}"
        )

    DEFAULT_MAX_SAMPLE_SIZE = 2**31 - 1
    try:
        relevant_columns = "c1", "essay_token_embeddings", "essay_token_length"
        df = (
            pl.scan_parquet(data_file)
            .select(relevant_columns)
            .head(DEFAULT_MAX_SAMPLE_SIZE)
            .drop_nulls()
            .unique()
            .collect()
        )
    except Exception:
        relevant_columns = "c1", "essay_token_embeddings"
        df = (
            pl.scan_parquet(data_file)
            .select(relevant_columns)
            .head(DEFAULT_MAX_SAMPLE_SIZE)
            .drop_nulls()
            .unique()
            .collect()
        )

    logger.info(
        "[gamma-search] Loaded BLSTM vectorized dataset with %d essays", len(df)
    )
    logger.info(
        "[gamma-search] Essay token embeddings schema: %s",
        df.schema["essay_token_embeddings"],
    )
    return df


def _split_vectorized_dataset_focal(
    *, max_samples: Optional[int]
) -> Tuple[EssayDataset, Any, Any, Any]:
    """Create EssayDataset and train/val/test splits with optional train cap."""

    import torch.utils.data as tud

    df = _load_vectorized_dataframe(max_samples=None)
    dataset = EssayDataset(df)

    train_ds, val_ds, test_ds = split_dataset(dataset, seed=42)

    if max_samples is not None and max_samples > 0 and len(train_ds) > max_samples:
        indices = train_ds.indices  # type: ignore[attr-defined]
        if isinstance(indices, range):
            indices = list(indices)
        indices = indices[:max_samples]
        train_ds = tud.Subset(train_ds.dataset, indices)  # type: ignore[arg-type]

    return dataset, train_ds, val_ds, test_ds


def main() -> int:
    """Entry point: run a FocalLoss gamma sweep on vectorized essays (BLSTM)."""

    print("\n" + "=" * 50)
    print("Component 1 BiLSTM FL Training on Vectorized Essays (gamma sweep)")
    print("=" * 50)

    parser = argparse.ArgumentParser(
        description=(
            "BiLSTM FocalLoss training on vectorized essays using a mandatory gamma sweep"
        ),
    )
    parser.add_argument(
        "--gamma-grid",
        type=str,
        default=None,
        help=(
            "comma-separated list of gamma values; if omitted, uses DEFAULT_GAMMA_VALUES"
        ),
    )
    parser.add_argument(
        "--epochs-per-gamma",
        type=int,
        default=None,
        help=(
            "number of epochs to train for each gamma; if omitted, uses TrainConfig.epochs"
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help=(
            "optional cap on training samples per gamma; validation/test always use full data"
        ),
    )

    args = parser.parse_args()

    if args.gamma_grid is not None:
        raw_items = [item.strip() for item in args.gamma_grid.split(",")]
        if not any(raw_items):
            raise SystemExit("error: --gamma-grid must contain at least one value")
        gamma_values: list[float] = []
        for raw in raw_items:
            if not raw:
                continue
            try:
                gamma = float(raw)
            except ValueError:
                raise SystemExit(
                    f"error: invalid gamma value '{raw}' in --gamma-grid='{args.gamma_grid}'"
                )
            if gamma < 0.0:
                raise SystemExit(
                    f"error: gamma values must be non-negative; got {gamma}"
                )
            gamma_values.append(gamma)
        if not gamma_values:
            raise SystemExit(
                "error: --gamma-grid must contain at least one numeric gamma value"
            )
    else:
        gamma_values = list(DEFAULT_GAMMA_VALUES)

    print("\n[gamma-search] Gamma grid for FocalLoss sweep (BLSTM vectorized essays):")
    print("  values: " + ", ".join(f"{g:.6g}" for g in gamma_values))
    print(f"  total gamma values to test: {len(gamma_values)}")

    if args.epochs_per_gamma is not None and args.epochs_per_gamma <= 0:
        raise SystemExit(
            f"error: --epochs-per-gamma must be a positive integer; got {args.epochs_per_gamma}"
        )

    if args.max_samples is not None and args.max_samples <= 0:
        raise SystemExit(
            f"error: --max-samples must be a positive integer; got {args.max_samples}"
        )

    device = get_device("auto")
    set_seed(42)
    logger.info(f"Using device: {device}")

    try:
        full_dataset, train_dataset, val_dataset, test_dataset = _split_vectorized_dataset_focal(
            max_samples=args.max_samples
        )

        all_scores = full_dataset.data["c1"].to_numpy()
        target_scaler = TargetScaler("none")
        target_scaler.fit(np.array(all_scores))

        class_freqs = get_class_frequencies(train_dataset)
        logger.info(
            "[gamma-search] Training set class distribution (BLSTM vectorized essays, capped if requested):"
        )
        print_class_distribution(class_freqs)
        alpha = calculate_alpha_from_frequency(class_freqs)

        def dataloaders_factory_for_sweep(max_samples_inner: Optional[int]):
            # max_samples_inner ignored; args.max_samples applied during initial split
            from common.setup import get_optimal_workers  # type: ignore

            train_config = TrainConfig()
            num_workers = get_optimal_workers(device)
            pin_memory = device.type == "cuda"

            train_loader = create_data_loader(
                train_dataset,
                batch_size=train_config.batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
            val_loader = create_data_loader(
                val_dataset,
                batch_size=train_config.batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
            test_loader = create_data_loader(
                test_dataset,
                batch_size=train_config.batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
            return train_loader, val_loader, test_loader

        def trainer_factory_blstm_vectorized(out_dir, train_loader, val_loader, test_loader):  # type: ignore[override]
            from blstm_train_on_vectorized_essays_cross_entropy_loss import (  # type: ignore
                create_component1_config_for_vectorized_essays,
                create_training_config,
            )

            model_config: ModelConfig = create_component1_config_for_vectorized_essays()
            train_config: TrainConfig = create_training_config()
            if args.epochs_per_gamma is not None:
                train_config.epochs = args.epochs_per_gamma

            serialization_config = SerializationConfig(
                output_dir=out_dir, save_best_only=True, keep_last_k=3
            )

            model = BiLSTMClassifier(model_config)

            trainer = BiLSTMFLTrainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                model_config=model_config,
                train_config=train_config,
                serialization_config=serialization_config,
                target_scaler=target_scaler,
                device=device,
                class_frequencies=None,
            )
            return trainer

        output_root = (
            Path(__file__).parent
            / "runs"
            / "vectorized_essays_focal_loss"
            / "blstm_model"
            / "gamma_sweep"
        )

        results_by_gamma, best_gamma, best_metrics = run_gamma_search(
            trainer_factory=trainer_factory_blstm_vectorized,
            dataloaders_factory=dataloaders_factory_for_sweep,
            gamma_values=gamma_values,
            num_classes=6,
            output_root=output_root,
            seed=42,
            alpha=alpha,
            max_samples=args.max_samples,
        )

        print("\nGamma sweep results (BLSTM vectorized essays, FocalLoss):")
        print("gamma\tqwk\tkappa\tmae")
        for gamma in sorted(
            results_by_gamma.keys(),
            key=lambda g: metric_key_for_selection(results_by_gamma[g]),
        ):
            m = results_by_gamma[gamma]
            qwk = float(m.get("qwk", float("nan")))
            kappa = float(m.get("kappa", float("nan")))
            mae = float(m.get("mae", float("nan")))
            print(f"{gamma:.3g}\t{qwk:.4f}\t{kappa:.4f}\t{mae:.4f}")

        best_dir = output_root / f"gamma_{str(best_gamma).replace('-', 'neg_').replace('.', '_')}"
        print("\nBest gamma:")
        print(f"  gamma = {best_gamma}")
        print(f"  metrics = {best_metrics}")
        print(f"  artifacts in: {best_dir}")

        return 0

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 1
    except SystemExit:
        raise
    except Exception as exc:
        logger.error(f"Training failed: {exc}", exc_info=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
