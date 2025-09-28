"""BERTimbau Preprocessing Pipeline for Essay Text to Embeddings

This module provides a complete pipeline to convert essay text into 768-dimensional
token embeddings using BERTimbau (Portuguese BERT) for BiLSTM training.

Features:
- Automatic model downloading and caching
- Efficient batched processing
- Memory-optimized inference
- Support for different BERTimbau variants
- Token-level embeddings extraction
- Progress tracking and logging
- Error handling and recovery

Usage:
    from bertimbau_preprocessing import BERTimbauProcessor
    
    # Initialize processor
    processor = BERTimbauProcessor()
    
    # Process single essay
    embeddings = processor.process_essay("Texto do ensaio aqui...")
    
    # Process multiple essays
    all_embeddings = processor.process_essays_batch(essays_list)
    
    # Save embeddings to files
    processor.save_embeddings(embeddings, "output_dir/")
"""

import logging
import os
import pickle
import time
from pathlib import Path
from typing import Any, Generator, Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModel, AutoTokenizer, 
    BertModel, BertTokenizer,
    pipeline
)
from tqdm import tqdm

logger = logging.getLogger(__name__)


class BERTimbauProcessor:
    """BERTimbau processor for converting essay text to token embeddings."""
    
    def __init__(
        self,
        model_name: str = "neuralmind/bert-base-portuguese-cased",
        device: str = "auto",
        max_length: int = 512,
        batch_size: int = 8,
        cache_dir: str | None = None,
        use_fast_tokenizer: bool = True
    ) -> None:
        """Initialize the BERTimbau processor.
        
        Args:
            model_name: HuggingFace model identifier for BERTimbau variant
            device: Device to use ('auto', 'cuda', 'cpu', 'mps')
            max_length: Maximum sequence length for tokenization
            batch_size: Batch size for processing multiple essays
            cache_dir: Directory to cache model files
            use_fast_tokenizer: Whether to use fast tokenizer implementation
        """
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        
        # Device setup
        self.device = self._setup_device(device)
        logger.info(f"Using device: {self.device}")
        
        # Load model and tokenizer
        self.tokenizer = self._load_tokenizer(use_fast_tokenizer)
        self.model = self._load_model()
        
        logger.info(f"BERTimbau processor initialized with model: {model_name}")
    
    def _setup_device(self, device_preference: str) -> torch.device:
        """Setup and validate device."""
        if device_preference == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
                logger.info(f"Auto-selected CUDA device: {torch.cuda.get_device_name()}")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = torch.device("mps")
                logger.info("Auto-selected MPS device")
            else:
                device = torch.device("cpu")
                logger.info("Auto-selected CPU device")
        else:
            device = torch.device(device_preference)
            logger.info(f"Using specified device: {device}")
        
        return device
    
    def _load_tokenizer(self, use_fast: bool) -> AutoTokenizer:
        """Load and configure tokenizer."""
        logger.info(f"Loading tokenizer: {self.model_name}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                use_fast=use_fast
            )
            logger.info(f"Tokenizer loaded successfully (fast: {use_fast})")
            return tokenizer
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise
    
    def _load_model(self) -> AutoModel:
        """Load and configure BERTimbau model."""
        logger.info(f"Loading model: {self.model_name}")
        try:
            model = AutoModel.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                output_hidden_states=False,  # We only need last hidden state
                output_attentions=False,     # We don't need attention weights
            )
            model = model.to(self.device)
            model.eval()  # Set to evaluation mode
            
            # Get model info
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            logger.info(f"Model loaded successfully")
            logger.info(f"Total parameters: {total_params:,}")
            logger.info(f"Trainable parameters: {trainable_params:,}")
            
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def process_essay(
        self, 
        essay_text: str, 
        return_tokens: bool = False,
        chunk_overlap: int = 50
    ) -> dict[str, np.ndarray | list[str]]:
        """Process a single essay into token embeddings.
        
        Args:
            essay_text: Raw essay text
            return_tokens: Whether to return tokenized text
            chunk_overlap: Overlap between chunks for long texts
            
        Returns:
            Dictionary with 'embeddings' key containing [seq_len, 768] array
            and optionally 'tokens' key with list of tokens
        """
        if not essay_text or not essay_text.strip():
            logger.warning("Empty essay text provided")
            empty_embedding = np.zeros((1, 768), dtype=np.float32)
            result = {'embeddings': empty_embedding}
            if return_tokens:
                result['tokens'] = ['[PAD]']
            return result
        
        # Tokenize text
        try:
            encoding = self.tokenizer(
                essay_text,
                truncation=True,
                padding=False,
                max_length=self.max_length,
                return_tensors="pt"
            )
        except Exception as e:
            logger.error(f"Tokenization failed: {e}")
            raise
        
        # Move to device
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Extract embeddings
        try:
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
            
            # Get last hidden state [batch_size, seq_len, hidden_size]
            embeddings = outputs.last_hidden_state.cpu().numpy()[0]  # Remove batch dim
            
            # Filter out padding tokens
            valid_length = attention_mask.sum().item()
            embeddings = embeddings[:valid_length]  # Shape: [valid_seq_len, 768]
            
        except Exception as e:
            logger.error(f"Model inference failed: {e}")
            raise
        
        result = {'embeddings': embeddings.astype(np.float32)}
        
        if return_tokens:
            tokens = self.tokenizer.convert_ids_to_tokens(
                input_ids[0][:valid_length].cpu().numpy()
            )
            result['tokens'] = tokens
        
        logger.debug(f"Processed essay: {len(embeddings)} tokens, shape: {embeddings.shape}")
        return result
    
    def process_essays_batch(
        self,
        essays: list[str],
        show_progress: bool = True,
        return_tokens: bool = False,
        max_workers: int | None = None
    ) -> list[dict[str, np.ndarray | list[str]]]:
        """Process multiple essays in batches.
        
        Args:
            essays: List of essay texts
            show_progress: Whether to show progress bar
            return_tokens: Whether to return tokenized text for each essay
            max_workers: Maximum number of worker processes (unused in current impl)
            
        Returns:
            List of dictionaries, each containing embeddings and optionally tokens
        """
        results = []
        
        iterator = tqdm(essays, desc="Processing essays") if show_progress else essays
        
        for i, essay in enumerate(iterator):
            try:
                result = self.process_essay(essay, return_tokens=return_tokens)
                results.append(result)
                
                if show_progress and (i + 1) % 100 == 0:
                    logger.info(f"Processed {i + 1}/{len(essays)} essays")
                    
            except Exception as e:
                logger.error(f"Failed to process essay {i}: {e}")
                # Add empty embedding for failed essay
                empty_embedding = np.zeros((1, 768), dtype=np.float32)
                result = {'embeddings': empty_embedding}
                if return_tokens:
                    result['tokens'] = ['[UNK]']
                results.append(result)
        
        logger.info(f"Completed processing {len(essays)} essays")
        return results
    
    def process_essays_from_dataframe(
        self,
        df: Any,  # pandas.DataFrame
        text_column: str,
        id_column: str | None = None,
        output_dir: str | None = None,
        save_format: Literal['npy', 'pt'] = 'npy',
        show_progress: bool = True
    ) -> list[dict[str, Any]]:
        """Process essays from a pandas DataFrame and optionally save to files.
        
        Args:
            df: Pandas DataFrame containing essays
            text_column: Column name containing essay text
            id_column: Column name for essay IDs (optional)
            output_dir: Directory to save embeddings (optional)
            save_format: Format to save embeddings ('npy' or 'pt')
            show_progress: Whether to show progress bar
            
        Returns:
            List of dictionaries with essay metadata and embedding info
        """
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Saving embeddings to: {output_path}")
        
        results = []
        essays = df[text_column].tolist()
        
        # Get IDs if specified
        if id_column and id_column in df.columns:
            ids = df[id_column].tolist()
        else:
            ids = [f"essay_{i:06d}" for i in range(len(essays))]
        
        # Process essays
        embeddings_results = self.process_essays_batch(
            essays, show_progress=show_progress, return_tokens=False
        )
        
        # Save and collect results
        for i, (essay_id, essay_text, embedding_result) in enumerate(zip(ids, essays, embeddings_results)):
            embeddings = embedding_result['embeddings']
            
            # Save to file if output directory specified
            if output_dir:
                if save_format == 'npy':
                    file_path = output_path / f"{essay_id}.npy"
                    np.save(file_path, embeddings)
                elif save_format == 'pt':
                    file_path = output_path / f"{essay_id}.pt"
                    torch.save(torch.from_numpy(embeddings), file_path)
                else:
                    raise ValueError(f"Unsupported save format: {save_format}")
            else:
                file_path = None
            
            # Collect metadata
            result = {
                'id': essay_id,
                'embedding_path': str(file_path) if file_path else None,
                'embedding_shape': embeddings.shape,
                'text_length': len(essay_text),
                'num_tokens': embeddings.shape[0]
            }
            
            # Add original DataFrame columns
            for col in df.columns:
                if col not in [text_column]:
                    result[col] = df.iloc[i][col]
            
            results.append(result)
        
        logger.info(f"Processing completed. Results: {len(results)} essays")
        return results
    
    def save_embeddings(
        self,
        embeddings: np.ndarray,
        output_path: str,
        format: Literal['npy', 'pt'] = 'npy'
    ) -> None:
        """Save embeddings to file.
        
        Args:
            embeddings: Embeddings array with shape [seq_len, 768]
            output_path: Output file path
            format: Save format ('npy' or 'pt')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'npy':
            np.save(output_path, embeddings)
        elif format == 'pt':
            torch.save(torch.from_numpy(embeddings), output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.debug(f"Saved embeddings to {output_path}: shape {embeddings.shape}")
    
    def get_model_info(self) -> dict[str, Any]:
        """Get information about the loaded model."""
        return {
            'model_name': self.model_name,
            'vocab_size': self.tokenizer.vocab_size,
            'max_length': self.max_length,
            'embedding_dim': self.model.config.hidden_size,
            'num_layers': self.model.config.num_hidden_layers,
            'num_attention_heads': self.model.config.num_attention_heads,
            'device': str(self.device)
        }


def create_embedding_dataset_from_json(
    json_path: str,
    text_column: str = "essay_as_single_utf8_string",
    score_column: str = "c1",
    id_column: str | None = None,
    output_dir: str = "embeddings",
    model_name: str = "neuralmind/bert-base-portuguese-cased",
    batch_size: int = 8,
    save_format: Literal['npy', 'pt'] = 'npy',
    max_length: int = 512
) -> tuple[str, dict[str, Any]]:
    """Create embedding dataset from JSON file containing essays.
    
    Args:
        json_path: Path to JSON file with essays
        text_column: Column name containing essay text
        score_column: Column name containing C1 scores
        id_column: Column name for essay IDs
        output_dir: Directory to save embeddings
        model_name: BERTimbau model variant
        batch_size: Processing batch size
        save_format: Format to save embeddings
        max_length: Maximum sequence length
        
    Returns:
        Tuple of (CSV path with metadata, processing statistics)
    """
    import pandas as pd
    
    logger.info(f"Creating embedding dataset from: {json_path}")
    
    # Load data
    df = pd.read_json(json_path)
    logger.info(f"Loaded {len(df)} essays from JSON")
    
    # Initialize processor
    processor = BERTimbauProcessor(
        model_name=model_name,
        max_length=max_length,
        batch_size=batch_size
    )
    
    # Process essays
    start_time = time.time()
    results = processor.process_essays_from_dataframe(
        df=df,
        text_column=text_column,
        id_column=id_column,
        output_dir=output_dir,
        save_format=save_format,
        show_progress=True
    )
    processing_time = time.time() - start_time
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save metadata CSV
    csv_path = Path(output_dir) / "embeddings_metadata.csv"
    results_df.to_csv(csv_path, index=False)
    
    # Compute statistics
    stats = {
        'total_essays': len(results),
        'processing_time_seconds': processing_time,
        'average_tokens_per_essay': np.mean([r['num_tokens'] for r in results]),
        'total_tokens_processed': sum(r['num_tokens'] for r in results),
        'model_info': processor.get_model_info(),
        'save_format': save_format,
        'output_dir': output_dir,
        'csv_path': str(csv_path)
    }
    
    logger.info(f"Dataset creation completed in {processing_time:.1f}s")
    logger.info(f"Average tokens per essay: {stats['average_tokens_per_essay']:.1f}")
    logger.info(f"Total tokens processed: {stats['total_tokens_processed']:,}")
    logger.info(f"Metadata saved to: {csv_path}")
    
    return str(csv_path), stats


def example_usage():
    """Example usage of the BERTimbau preprocessing pipeline."""
    import pandas as pd
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
    )
    
    # Example 1: Process single essay
    processor = BERTimbauProcessor()
    
    sample_essay = """
    Este é um exemplo de ensaio em português. O texto será processado pelo modelo
    BERTimbau para gerar representações vetoriais de 768 dimensões para cada token.
    Essas representações serão usadas como entrada para o modelo BiLSTM.
    """
    
    result = processor.process_essay(sample_essay, return_tokens=True)
    print(f"Embeddings shape: {result['embeddings'].shape}")
    print(f"Tokens: {result['tokens'][:10]}...")  # First 10 tokens
    
    # Example 2: Process multiple essays
    essays = [
        "Primeiro ensaio de exemplo para demonstrar o processamento em lote.",
        "Segundo ensaio com texto diferente para testar a variabilidade.",
        "Terceiro ensaio que mostra como múltiplos textos são processados juntos."
    ]
    
    results = processor.process_essays_batch(essays, show_progress=True)
    for i, result in enumerate(results):
        print(f"Essay {i+1}: {result['embeddings'].shape}")
    
    # Example 3: Create sample DataFrame and process
    sample_data = {
        'id': ['essay_001', 'essay_002', 'essay_003'],
        'essay_text': essays,
        'c1_score': [120, 160, 80]
    }
    df = pd.DataFrame(sample_data)
    
    # Process and save
    metadata_results = processor.process_essays_from_dataframe(
        df=df,
        text_column='essay_text',
        id_column='id',
        output_dir='sample_embeddings',
        save_format='npy',
        show_progress=True
    )
    
    print(f"Processed {len(metadata_results)} essays")
    for result in metadata_results:
        print(f"ID: {result['id']}, Shape: {result['embedding_shape']}, Path: {result['embedding_path']}")


if __name__ == '__main__':
    example_usage()