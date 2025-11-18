"""
Dataset Structure Analyzer
Analyzes CSV files to extract schema, data types, unique values, and statistics.
Output is formatted for AI coding assistants to understand data structure.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Any
import sys

def analyze_column(series: pd.Series, max_unique_samples: int = 10) -> Dict[str, Any]:
    """
    Analyze a single column and extract metadata.
    
    Args:
        series: Pandas Series to analyze
        max_unique_samples: Maximum unique values to include
    
    Returns:
        Column metadata dictionary
    """
    col_info = {
        'name': series.name,
        'dtype': str(series.dtype),
        'non_null_count': int(series.count()),
        'null_count': int(series.isnull().sum()),
        'null_percentage': float(series.isnull().sum() / len(series) * 100),
    }
    
    # Type-specific analysis
    if pd.api.types.is_numeric_dtype(series):
        col_info['statistics'] = {
            'min': float(series.min()),
            'max': float(series.max()),
            'mean': float(series.mean()),
            'median': float(series.median()),
            'std': float(series.std())
        }
        
        # Check if likely categorical (few unique values)
        n_unique = series.nunique()
        if n_unique < 50:
            col_info['unique_values'] = sorted(series.dropna().unique().tolist())[:max_unique_samples]
            col_info['value_counts'] = series.value_counts().head(10).to_dict()
    
    elif pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        n_unique = series.nunique()
        col_info['unique_count'] = int(n_unique)
        
        if n_unique < 100:
            # Small cardinality - show all unique values
            col_info['unique_values'] = sorted([str(x) for x in series.dropna().unique()])[:max_unique_samples]
            col_info['value_counts'] = {str(k): int(v) for k, v in series.value_counts().head(20).items()}
        else:
            # High cardinality - show samples
            col_info['sample_values'] = [str(x) for x in series.dropna().head(max_unique_samples).tolist()]
            col_info['most_common'] = {str(k): int(v) for k, v in series.value_counts().head(10).items()}
    
    elif pd.api.types.is_datetime64_any_dtype(series):
        col_info['date_range'] = {
            'min': str(series.min()),
            'max': str(series.max())
        }
    
    return col_info

def analyze_csv(file_path: Path, name: str = None) -> Dict[str, Any]:
    """
    Analyze a CSV file and extract comprehensive metadata.
    
    Args:
        file_path: Path to CSV file
        name: Human-readable name for the dataset
    
    Returns:
        Dataset metadata dictionary
    """
    print(f"\nAnalyzing: {file_path.name}")
    
    try:
        # Load CSV
        df = pd.read_csv(file_path, low_memory=False)
        
        # Basic info
        dataset_info = {
            'name': name or file_path.stem,
            'file': str(file_path),
            'shape': {'rows': int(df.shape[0]), 'columns': int(df.shape[1])},
            'columns': []
        }
        
        # Analyze each column
        print(f"  Rows: {df.shape[0]:,} | Columns: {df.shape[1]}")
        
        for col in df.columns:
            col_info = analyze_column(df[col])
            dataset_info['columns'].append(col_info)
            
            # Progress indicator
            print(f"     {col_info['name']} ({col_info['dtype']})")
        
        return dataset_info
    
    except Exception as e:
        print(f"   Error: {e}")
        return None

def generate_ai_prompt(datasets: List[Dict[str, Any]]) -> str:
    """
    Generate AI-friendly prompt with dataset structure.
    
    Args:
        datasets: List of analyzed dataset dictionaries
    
    Returns:
        Formatted prompt string
    """
    prompt = "# Dataset Structure Information\n\n"
    prompt += "This project uses the Free Music Archive (FMA) dataset with the following structure:\n\n"
    
    for ds in datasets:
        if ds is None:
            continue
        
        prompt += f"## {ds['name']}\n\n"
        prompt += f"**File**: `{Path(ds['file']).name}`  \n"
        prompt += f"**Shape**: {ds['shape']['rows']:,} rows × {ds['shape']['columns']} columns\n\n"
        
        prompt += "### Columns\n\n"
        
        for col in ds['columns']:
            prompt += f"#### `{col['name']}` ({col['dtype']})\n\n"
            prompt += f"- **Non-null**: {col['non_null_count']:,} ({100 - col['null_percentage']:.1f}%)\n"
            
            if 'statistics' in col:
                stats = col['statistics']
                prompt += f"- **Range**: [{stats['min']:.2f}, {stats['max']:.2f}]\n"
                prompt += f"- **Mean**: {stats['mean']:.2f} | **Std**: {stats['std']:.2f}\n"
            
            if 'unique_values' in col:
                prompt += f"- **Unique values** ({len(col['unique_values'])}): {col['unique_values']}\n"
            
            if 'value_counts' in col:
                prompt += f"- **Value distribution**:\n"
                for val, count in list(col['value_counts'].items())[:5]:
                    prompt += f"  - `{val}`: {count:,}\n"
            
            if 'sample_values' in col:
                prompt += f"- **Sample values**: {col['sample_values'][:5]}\n"
            
            prompt += "\n"
        
        prompt += "---\n\n"
    
    return prompt

def main():
    """Main execution."""
    print("=" * 70)
    print("FMA DATASET STRUCTURE ANALYZER")
    print("=" * 70)
    
    # Define data directory
    data_dir = Path(__file__).parent.parent / 'data' / 'metadata'
    
    if not data_dir.exists():
        print(f"\n Data directory not found: {data_dir}")
        print("\nRun this first:")
        print("  cd data/metadata/")
        print("  curl -L -o fma_metadata.zip https://os.unil.cloud.switch.ch/fma/fma_metadata.zip")
        print("  unzip fma_metadata.zip")
        sys.exit(1)
    
    # Analyze key CSV files
    csv_files = {
        'tracks.csv': 'Track Metadata',
        'genres.csv': 'Genre Definitions',
        'features.csv': 'Audio Features'
    }
    
    datasets = []
    
    for filename, name in csv_files.items():
        file_path = data_dir / filename
        
        if file_path.exists():
            dataset_info = analyze_csv(file_path, name)
            if dataset_info:
                datasets.append(dataset_info)
        else:
            print(f"\n  File not found: {filename}")
    
    if not datasets:
        print("\n No CSV files found to analyze")
        sys.exit(1)
    
    # Save full analysis as JSON
    output_json = Path(__file__).parent.parent / 'data' / 'dataset_structure.json'
    with open(output_json, 'w') as f:
        json.dump(datasets, f, indent=2)
    
    print(f"\n Saved full analysis to: {output_json}")
    
    # Generate AI prompt
    ai_prompt = generate_ai_prompt(datasets)
    
    # Save AI prompt
    output_prompt = Path(__file__).parent.parent / 'data' / 'DATASET_STRUCTURE.md'
    with open(output_prompt, 'w') as f:
        f.write(ai_prompt)
    
    print(f" Saved AI-friendly prompt to: {output_prompt}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nAnalyzed {len(datasets)} datasets:")
    for ds in datasets:
        print(f"  - {ds['name']}: {ds['shape']['rows']:,} rows × {ds['shape']['columns']} cols")
    
    print(f"\nTo use with AI tools, share the content of:")
    print(f"  {output_prompt}")
    
    print("\n" + "=" * 70)

if __name__ == '__main__':
    main()
