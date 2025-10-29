"""I/O utilities for file operations and dataset management

Provides functions for directory management and dataset persistence.
"""

import logging
import pathlib
from typing import Union

import polars as pl

logger = logging.getLogger(__name__)


def ensure_dir(path: Union[str, pathlib.Path]) -> pathlib.Path:
    """Create directory if it doesn't exist.
    
    Args:
        path: Directory path to create
        
    Returns:
        Path object for the created/existing directory
        
    Example:
        >>> output_dir = ensure_dir("runs/experiment_1")
        >>> model.save(output_dir / "model.pt")
    """
    path = pathlib.Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_dataset(
    dataset: pl.DataFrame,
    filename: str,
    *extensions: str,
    output_dir: Union[str, pathlib.Path] = "generated_datasets"
) -> None:
    """Save Polars DataFrame in multiple formats.
    
    Saves dataset to generated_datasets directory in specified formats.
    Automatically determines the appropriate write method based on extension.
    
    Args:
        dataset: Polars DataFrame to save
        filename: Base filename without extension
        *extensions: File extensions to save (e.g., 'csv', 'parquet', 'json')
        output_dir: Output directory (default: 'generated_datasets')
        
    Example:
        >>> save_dataset(df, "preprocessed_essays", "csv", "parquet")
        # Saves to generated_datasets/preprocessed_essays.csv and .parquet
    """
    # Resolve output directory
    if isinstance(output_dir, str):
        # Assume relative to project root
        project_root = pathlib.Path(__file__).parent.parent.parent
        output_dir = project_root / output_dir
    else:
        output_dir = pathlib.Path(output_dir)
    
    ensure_dir(output_dir)
    
    for extension in extensions:
        extension = extension.lstrip('.')  # Remove leading dot if present
        filepath = output_dir / f"{filename}.{extension}"
        
        logger.info(f"Writing dataset to {extension} file: {filepath}")
        
        # Get the appropriate write method
        write_method = getattr(dataset, f"write_{extension}", None)
        if write_method is None:
            logger.warning(f"Unsupported extension '{extension}', skipping")
            continue
            
        write_method(filepath)
        logger.info(f"Dataset written to {extension} file: {filepath}")


def load_dataset(
    filename: str,
    extension: str = "parquet",
    input_dir: Union[str, pathlib.Path] = "generated_datasets",
    lazy: bool = False
) -> Union[pl.DataFrame, pl.LazyFrame]:
    """Load dataset from file with automatic format detection.
    
    Args:
        filename: Filename with or without extension
        extension: File extension if not in filename
        input_dir: Input directory
        lazy: Whether to return LazyFrame for lazy evaluation
        
    Returns:
        Polars DataFrame or LazyFrame
        
    Example:
        >>> df = load_dataset("preprocessed_essays", lazy=True)
        >>> df = df.select(["c1", "essay"]).collect()
    """
    # Resolve input directory
    if isinstance(input_dir, str):
        project_root = pathlib.Path(__file__).parent.parent.parent
        input_dir = project_root / input_dir
    else:
        input_dir = pathlib.Path(input_dir)
    
    # Handle filename with or without extension
    if not filename.endswith(f".{extension}"):
        filename = f"{filename}.{extension}"
    
    filepath = input_dir / filename
    
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset not found: {filepath}")
    
    logger.info(f"Loading dataset from {filepath}")
    
    # Use scan for lazy loading, read for eager
    if lazy:
        if extension == "parquet":
            return pl.scan_parquet(filepath)
        elif extension == "csv":
            return pl.scan_csv(filepath)
        else:
            logger.warning(f"Lazy loading not supported for {extension}, using eager")
            return pl.read_json(filepath) if extension == "json" else pl.read_csv(filepath)
    else:
        if extension == "parquet":
            return pl.read_parquet(filepath)
        elif extension == "csv":
            return pl.read_csv(filepath)
        elif extension == "json":
            return pl.read_json(filepath)
        else:
            raise ValueError(f"Unsupported extension: {extension}")
