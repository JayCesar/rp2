#!/usr/bin/env python3
"""BiLSTM Training on Features with FocalLoss (gamma sweep)

Runs a FocalLoss gamma sweep for Component 1 BLSTM classification on
linguistic features, mirroring the Conv1D focal-loss scripts.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import polars as pl
import polars.selectors as cs
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

# Common utilities (data + gamma search)
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


def _load_features_dataframe() -> pl.DataFrame:
    """Load full features dataset (c1 + SCREAMING_SNAKE_CASE columns)."""

    parquet_file = (
        project_root
        / "generated_datasets"
        / "dataset_with_languagetool_metrics.parquet"
    )
    json_file = (
        project_root / "generated_datasets" / "dataset_with_languagetool_metrics.json"
    )

    data_file: Optional[Path]
    if parquet_file.exists():
        data_file = parquet_file
        logger.info(f"[gamma-search] Using parquet file: {data_file}")
    elif json_file.exists():
        data_file = json_file
        logger.info(f"[gamma-search] Using JSON file: {data_file}")
    else:
        raise FileNotFoundError(
            f"Data file not found. Looked for: {parquet_file} and {json_file}"
        )

    relevant_columns = [pl.col("c1"), cs.matches(r"^[A-Z0-9_]+$")]
    DEFAULT_MAX_SAMPLE_SIZE = 2**31 - 1
    return (
        pl.scan_parquet(data_file)
        .select(relevant_columns)
        .head(DEFAULT_MAX_SAMPLE_SIZE)
        .drop_nulls()
        .unique()
        .collect()
    )


def _split_features_dataset_focal(
    *, max_samples: Optional[int]
) -> Tuple[EssayDataset, any, any, any]:
    """Create EssayDataset and train/val/test splits with optional train cap.

    The optional ``max_samples`` cap is applied **only** to the training split,
    mirroring the Conv1D focal-loss implementation.
    """

    import torch.utils.data as tud

    df = _load_features_dataframe()
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
    """Entry point: run a FocalLoss gamma sweep on features (BLSTM)."""

    print("\n" + "=" * 50)
    print("Component 1 BiLSTM FL Training on Features (gamma sweep)")
    print("=" * 50)

    parser = argparse.ArgumentParser(
        description=(
            "BiLSTM FocalLoss training on features using a mandatory gamma sweep"
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

    # Parse gamma grid
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

    print("\n[gamma-search] Gamma grid for FocalLoss sweep (BLSTM features):")
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
        # Pre-split dataset once
        full_dataset, train_dataset, val_dataset, test_dataset = _split_features_dataset_focal(
            max_samples=args.max_samples
        )

        # Target scaler (kept for API parity; classification uses raw scores)
        all_scores = full_dataset.data["c1"].to_numpy()
        target_scaler = TargetScaler("none")
        target_scaler.fit(np.array(all_scores))

        # Compute class frequencies / alpha on (optionally capped) training split
        class_freqs = get_class_frequencies(train_dataset)
        logger.info(
            "[gamma-search] Training set class distribution (BLSTM features, capped if requested):"
        )
        print_class_distribution(class_freqs)
        alpha = calculate_alpha_from_frequency(class_freqs)

        # DataLoaders factory reusing pre-split datasets
        def dataloaders_factory_for_sweep(max_samples_inner: Optional[int]):
            # max_samples_inner is ignored: args.max_samples already applied above
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

        # Trainer factory capturing device, target scaler, epochs override
        def trainer_factory_blstm_features(out_dir, train_loader, val_loader, test_loader):  # type: ignore[override]
            from blstm_train_on_features_cross_entropy_loss import (  # type: ignore
                create_component1_config_for_features,
                create_training_config,
            )

            ds = train_loader.dataset
            base_ds = getattr(ds, "dataset", ds)

            model_config: ModelConfig = create_component1_config_for_features()
            try:
                input_dim = len(base_ds.feature_cols)
            except Exception:
                first_row = base_ds.data.row(0, named=True)
                input_dim = len([k for k in first_row.keys() if k != "c1"])
            model_config.input_dim = input_dim

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
            / "features_focal_loss"
            / "blstm_model"
            / "gamma_sweep"
        )

        results_by_gamma, best_gamma, best_metrics = run_gamma_search(
            trainer_factory=trainer_factory_blstm_features,
            dataloaders_factory=dataloaders_factory_for_sweep,
            gamma_values=gamma_values,
            num_classes=6,
            output_root=output_root,
            seed=42,
            alpha=alpha,
            max_samples=args.max_samples,
        )

        # Summary table sorted by selection key
        print("\nGamma sweep results (BLSTM features, FocalLoss):")
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
