#!/usr/bin/env python3
"""Process Essays Dataset with BERTimbau

This script processes the existing essay dataset from JSON format,
converting essay texts to 768-dimensional BERTimbau embeddings
for use with the BiLSTM model.

Usage:
    python process_essays_dataset.py [options]
    
    # Process with default settings
    python process_essays_dataset.py
    
    # Process with custom settings
    python process_essays_dataset.py --input data.json --output embeddings_dir --batch-size 16
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

from bertimbau_preprocessing import BERTimbauProcessor, create_embedding_dataset_from_json


def setup_logging(log_level: str = 'INFO') -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('essay_processing.log')
        ]
    )


def find_essay_datasets() -> list[Path]:
    """Find available essay dataset files."""
    current_dir = Path.cwd()
    dataset_patterns = [
        "generated_datasets/*.json",
        "*.json"
    ]
    
    found_files = []
    for pattern in dataset_patterns:
        found_files.extend(current_dir.glob(pattern))
    
    # Filter for likely essay datasets
    essay_files = []
    for file in found_files:
        if any(keyword in file.name.lower() for keyword in ['essay', 'redacao', 'dataset']):
            essay_files.append(file)
    
    return essay_files


def inspect_dataset(file_path: str) -> dict:
    """Inspect dataset to understand its structure."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list) and len(data) > 0:
            sample = data[0]
            columns = list(sample.keys()) if isinstance(sample, dict) else []
            
            # Look for text and score columns
            text_column = None
            score_column = None
            id_column = None
            
            for col in columns:
                col_lower = col.lower()
                if 'essay' in col_lower or 'text' in col_lower or 'redacao' in col_lower:
                    text_column = col
                elif 'c1' in col_lower or 'score' in col_lower or 'nota' in col_lower:
                    score_column = col
                elif 'id' in col_lower:
                    id_column = col
            
            return {
                'total_records': len(data),
                'columns': columns,
                'text_column': text_column,
                'score_column': score_column,
                'id_column': id_column,
                'sample_record': sample
            }
        else:
            return {'error': 'Unsupported data format'}
            
    except Exception as e:
        return {'error': str(e)}


def main():
    """Main processing function."""
    parser = argparse.ArgumentParser(
        description='Process essays dataset with BERTimbau embeddings',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process default dataset
    python process_essays_dataset.py
    
    # Specify input and output
    python process_essays_dataset.py --input data.json --output embeddings
    
    # Use different model and batch size  
    python process_essays_dataset.py --model neuralmind/bert-large-portuguese-cased --batch-size 4
    
    # List available datasets
    python process_essays_dataset.py --list-datasets
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        help='Input JSON file with essays (auto-detected if not specified)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='embeddings',
        help='Output directory for embeddings (default: embeddings)'
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='neuralmind/bert-base-portuguese-cased',
        choices=[
            'neuralmind/bert-base-portuguese-cased',
            'neuralmind/bert-large-portuguese-cased',
            'ricardoz/BERTugues-base-portuguese-cased'
        ],
        help='BERTimbau model variant to use'
    )
    
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=8,
        help='Batch size for processing (default: 8)'
    )
    
    parser.add_argument(
        '--max-length',
        type=int,
        default=512,
        help='Maximum sequence length (default: 512)'
    )
    
    parser.add_argument(
        '--format',
        choices=['npy', 'pt'],
        default='npy',
        help='Output format for embeddings (default: npy)'
    )
    
    parser.add_argument(
        '--text-column',
        type=str,
        help='Column name containing essay text (auto-detected if not specified)'
    )
    
    parser.add_argument(
        '--score-column',
        type=str,
        help='Column name containing C1 scores (auto-detected if not specified)'
    )
    
    parser.add_argument(
        '--id-column',
        type=str,
        help='Column name containing essay IDs (auto-detected if not specified)'
    )
    
    parser.add_argument(
        '--list-datasets',
        action='store_true',
        help='List available datasets and exit'
    )
    
    parser.add_argument(
        '--inspect',
        action='store_true',
        help='Inspect dataset structure and exit'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # List datasets if requested
    if args.list_datasets:
        logger.info("Searching for essay datasets...")
        datasets = find_essay_datasets()
        
        if datasets:
            print("\nAvailable datasets:")
            for i, dataset in enumerate(datasets, 1):
                print(f"  {i}. {dataset}")
        else:
            print("No essay datasets found in current directory.")
        return 0
    
    # Determine input file
    input_file = args.input
    if not input_file:
        datasets = find_essay_datasets()
        if not datasets:
            logger.error("No essay datasets found. Please specify --input or check your directory.")
            return 1
        input_file = str(datasets[0])
        logger.info(f"Auto-selected dataset: {input_file}")
    
    if not Path(input_file).exists():
        logger.error(f"Input file not found: {input_file}")
        return 1
    
    # Inspect dataset if requested
    if args.inspect:
        logger.info(f"Inspecting dataset: {input_file}")
        info = inspect_dataset(input_file)
        
        if 'error' in info:
            logger.error(f"Failed to inspect dataset: {info['error']}")
            return 1
        
        print(f"\nDataset Information:")
        print(f"  File: {input_file}")
        print(f"  Total records: {info['total_records']:,}")
        print(f"  Columns: {', '.join(info['columns'])}")
        print(f"  Detected text column: {info['text_column']}")
        print(f"  Detected score column: {info['score_column']}")
        print(f"  Detected ID column: {info['id_column']}")
        print(f"\nSample record:")
        for key, value in list(info['sample_record'].items())[:5]:
            value_str = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
            print(f"  {key}: {value_str}")
        
        return 0
    
    # Get dataset info for column detection
    logger.info(f"Analyzing dataset structure: {input_file}")
    info = inspect_dataset(input_file)
    
    if 'error' in info:
        logger.error(f"Failed to read dataset: {info['error']}")
        return 1
    
    # Determine column names
    text_column = args.text_column or info['text_column']
    score_column = args.score_column or info['score_column']
    id_column = args.id_column or info['id_column']
    
    if not text_column:
        logger.error("Could not detect text column. Please specify --text-column")
        return 1
    
    logger.info(f"Using columns - Text: {text_column}, Score: {score_column}, ID: {id_column}")
    logger.info(f"Processing {info['total_records']:,} essays")
    logger.info(f"Model: {args.model}")
    logger.info(f"Output directory: {args.output}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Max length: {args.max_length}")
    logger.info(f"Output format: {args.format}")
    
    # Confirm processing for large datasets
    if info['total_records'] > 1000:
        response = input(f"\nProcess {info['total_records']:,} essays? This may take a while. (y/N): ")
        if response.lower() != 'y':
            logger.info("Processing cancelled.")
            return 0
    
    try:
        # Process dataset
        logger.info("Starting BERTimbau processing...")
        start_time = time.time()
        
        csv_path, stats = create_embedding_dataset_from_json(
            json_path=input_file,
            text_column=text_column,
            score_column=score_column,
            id_column=id_column,
            output_dir=args.output,
            model_name=args.model,
            batch_size=args.batch_size,
            save_format=args.format,
            max_length=args.max_length
        )
        
        total_time = time.time() - start_time
        
        # Print summary
        print("\n" + "="*60)
        print("PROCESSING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Total essays processed: {stats['total_essays']:,}")
        print(f"Total processing time: {total_time:.1f} seconds")
        print(f"Average time per essay: {total_time/stats['total_essays']:.2f} seconds")
        print(f"Average tokens per essay: {stats['average_tokens_per_essay']:.1f}")
        print(f"Total tokens processed: {stats['total_tokens_processed']:,}")
        print(f"Model used: {stats['model_info']['model_name']}")
        print(f"Output directory: {args.output}/")
        print(f"Metadata CSV: {csv_path}")
        print("\nEmbedding files are ready for BiLSTM training!")
        print("="*60)
        
        # Save processing summary
        summary_path = Path(args.output) / 'processing_summary.json'
        with open(summary_path, 'w') as f:
            json.dump({
                'input_file': input_file,
                'processing_time': total_time,
                'args': vars(args),
                'stats': stats
            }, f, indent=2, default=str)
        
        logger.info(f"Processing summary saved to: {summary_path}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user.")
        return 1
    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    exit(main())