import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from collections import Counter

def analyze_data_distribution():
    """Analyze the distribution of C1 scores and essay characteristics."""
    
    print("🔍 Analyzing Data Distribution")
    print("=" * 50)
    
    # Load data
    project_root = Path(__file__).parent.parent
    csv_path = project_root / "generated_datasets" / "extended_essay-br_preprocessed_for_BERT.csv"
    
    print(f"Loading data from: {csv_path}")
    
    relevant_columns = ["essay_as_single_utf8_string", "c1"]
    df = (pl.scan_csv(csv_path).select(relevant_columns)
        .filter((pl.col("c1") > 0)) # Remove samples with C1 score of 0, as they are not reliable enough
        .drop_nulls().collect()
    )
    
    print(f"Total samples: {len(df)}")
    
    # Analyze C1 score distribution
    print("\n📊 C1 Score Distribution:")
    c1_counts = df["c1"].value_counts().sort("c1")
    print(c1_counts)
    
    # Calculate statistics
    c1_values = df["c1"].to_numpy()
    print(f"\nC1 Score Statistics:")
    print(f"  Min: {np.min(c1_values)}")
    print(f"  Max: {np.max(c1_values)}")
    print(f"  Mean: {np.mean(c1_values):.2f}")
    print(f"  Median: {np.median(c1_values):.2f}")
    print(f"  Std: {np.std(c1_values):.2f}")
    
    # Check if dataset is imbalanced
    unique_scores = np.unique(c1_values)
    print(f"\nUnique C1 scores: {unique_scores}")
    
    # Sample essays from different C1 levels
    print(f"\n📝 Sample Essays by C1 Level:")
    
    sample_essays = {}
    for c1_score in [40, 80, 120, 160, 200]:
        sample_df = df.filter(pl.col("c1") == c1_score).head(2)
        if len(sample_df) > 0:
            sample_essays[c1_score] = []
            for row in sample_df.to_dicts():
                essay_text = row["essay_as_single_utf8_string"]
                sample_essays[c1_score].append(essay_text)
    
    # Display sample essays
    for c1_score, essays in sample_essays.items():
        print(f"\n--- C1 Score: {c1_score} ---")
        for i, essay in enumerate(essays[:2], 1):
            preview = essay[:200] + "..." if len(essay) > 200 else essay
            print(f"Essay {i}: {preview}")
            print(f"Length: {len(essay)} characters")
    
    # Analyze essay lengths by C1 score
    print(f"\n📏 Essay Length Analysis:")
    
    essay_lengths = {}
    for c1_score in unique_scores:
        sample_df = df.filter(pl.col("c1") == c1_score)
        lengths = []
        
        for row in sample_df.to_dicts():
            essay_text = row["essay_as_single_utf8_string"]
            lengths.append(len(essay_text))
        
        if lengths:
            essay_lengths[c1_score] = {
                'mean': np.mean(lengths),
                'median': np.median(lengths),
                'std': np.std(lengths),
                'count': len(lengths)
            }
    
    for c1_score, stats in essay_lengths.items():
        print(f"C1={c1_score}: Mean={stats['mean']:.0f}, Median={stats['median']:.0f}, "
              f"Std={stats['std']:.0f}, Count={stats['count']}")
    
    # Check if there are meaningful differences
    print(f"\n🔍 Analysis Summary:")
    
    # Check if the model is just predicting the mean
    predicted_mean = 27.56  # From the inference output
    actual_mean = np.mean(c1_values)
    print(f"  Actual C1 mean: {actual_mean:.2f}")
    print(f"  Model prediction mean: {predicted_mean:.2f}")
    print(f"  Difference: {abs(predicted_mean - actual_mean):.2f}")
    
    # Check data balance
    min_count = min([stats['count'] for stats in essay_lengths.values()])
    max_count = max([stats['count'] for stats in essay_lengths.values()])
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    print(f"  Class imbalance ratio: {imbalance_ratio:.2f}")
    
    if imbalance_ratio > 10:
        print("  ⚠️  HIGH CLASS IMBALANCE detected!")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    if abs(predicted_mean - actual_mean) < 5:
        print("  ❌ Model appears to be predicting near the dataset mean")
        print("     → Need better training to learn actual patterns")
    
    if imbalance_ratio > 5:
        print("  ❌ Dataset is imbalanced")
        print("     → Use stratified sampling or weighted loss")
    
    # Check if essays actually differ between C1 levels
    length_correlation = []
    for c1_score, stats in essay_lengths.items():
        length_correlation.append((c1_score, stats['mean']))
    
    length_correlation.sort()
    print(f"  Essay length vs C1 correlation:")
    for c1, length in length_correlation:
        print(f"    C1={c1} → Avg Length={length:.0f}")

if __name__ == "__main__":
    analyze_data_distribution()
