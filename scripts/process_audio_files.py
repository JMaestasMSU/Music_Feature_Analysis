"""
Process audio files and extract features for genre classification.
This script shows the complete workflow from raw audio to ML-ready features.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import os

# Determine project root dynamically
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / 'data'

# Add parent directory to path
sys.path.append(str(PROJECT_ROOT))

from preprocessing.feature_extraction import (
    extract_features,
    extract_features_batch,
    features_to_array
)


def resolve_path(path_str: str) -> Path:
    """
    Resolve a path string to absolute Path object.
    Handles both relative and absolute paths.
    
    Args:
        path_str: Path as string (can be relative or absolute)
    
    Returns:
        Resolved absolute Path object
    """
    path = Path(path_str)
    
    # If absolute, use as-is
    if path.is_absolute():
        return path
    
    # If relative, try multiple base paths
    # 1. Try relative to current working directory
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path.resolve()
    
    # 2. Try relative to project root
    project_path = PROJECT_ROOT / path
    if project_path.exists():
        return project_path.resolve()
    
    # 3. Try relative to data directory
    data_path = DATA_DIR / path
    if data_path.exists():
        return data_path.resolve()
    
    # 4. If file doesn't exist yet, assume relative to project root
    return project_path.resolve()


def process_single_file(audio_path: str, sr: int = 22050, duration: float = 30.0):
    """
    Example: Process a single audio file.
    
    Args:
        audio_path: Path to audio file (.mp3, .wav, .au, etc.)
        sr: Sample rate (22050 Hz default)
        duration: Duration to process (30 seconds default)
    
    Returns:
        Feature dictionary
    """
    audio_path = resolve_path(audio_path)
    
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    print(f"Processing: {audio_path}")
    
    # Extract features
    features = extract_features(str(audio_path), sr=sr, duration=duration)
    
    print("\nExtracted features:")
    print(f"  MFCCs (13): {features['mfcc'][:3]}... (first 3)")
    print(f"  Spectral Centroid: {features['spectral_centroid']:.2f} Hz")
    print(f"  Spectral Rolloff: {features['spectral_rolloff']:.2f} Hz")
    print(f"  Zero Crossing Rate: {features['zcr']:.6f}")
    print(f"  RMS Energy: {features['rms_energy']:.6f}")
    print(f"  Chroma (12): {features['chroma'][:3]}... (first 3)")
    
    # Convert to flat array for ML
    feature_array = features_to_array(features)
    print(f"\nFeature vector shape: {feature_array.shape}")
    print(f"Feature vector: {feature_array}")
    
    return features


def process_from_precomputed_features(
    features_csv_path: str,
    tracks_csv_path: str,
    output_path: str = 'data/processed/ml_ready_features.pkl'
):
    """
    Alternative: Use pre-computed features from FMA metadata.
    This is much faster than processing raw audio!
    
    Args:
        features_csv_path: Path to features.csv (pre-computed)
        tracks_csv_path: Path to tracks.csv (genre labels)
        output_path: Where to save processed data
    """
    print("=" * 70)
    print("LOADING PRE-COMPUTED FEATURES")
    print("=" * 70)
    
    # Resolve all paths
    features_path = resolve_path(features_csv_path)
    tracks_path = resolve_path(tracks_csv_path)
    output_file = resolve_path(output_path)
    
    # Validate input files exist
    if not features_path.exists():
        raise FileNotFoundError(
            f"Features file not found: {features_path}\n"
            f"  Looked in:\n"
            f"    - {Path.cwd() / features_csv_path}\n"
            f"    - {PROJECT_ROOT / features_csv_path}\n"
            f"    - {DATA_DIR / features_csv_path}"
        )
    
    if not tracks_path.exists():
        raise FileNotFoundError(
            f"Tracks file not found: {tracks_path}\n"
            f"  Looked in:\n"
            f"    - {Path.cwd() / tracks_csv_path}\n"
            f"    - {PROJECT_ROOT / tracks_csv_path}\n"
            f"    - {DATA_DIR / tracks_csv_path}"
        )
    
    # Load pre-computed features
    print(f"\nLoading features from: {features_path}")
    features_df = pd.read_csv(features_path, index_col=0, header=[0, 1, 2])
    
    # Load track metadata
    print(f"Loading metadata from: {tracks_path}")
    tracks_df = pd.read_csv(tracks_path, index_col=0, header=[0, 1])
    
    # Merge
    print("\nMerging features with genres...")
    genre_col = ('track', 'genre_top')
    
    # Get features (select relevant feature groups)
    feature_groups = ['mfcc', 'spectral_centroid', 'spectral_rolloff', 
                      'zcr', 'rmse', 'chroma_cens']
    
    selected_features = []
    for group in feature_groups:
        if group in features_df.columns.levels[0]:
            group_features = features_df[group]
            selected_features.append(group_features)
    
    if selected_features:
        features_combined = pd.concat(selected_features, axis=1)
    else:
        features_combined = features_df
    
    # Add genres
    features_combined['genre'] = tracks_df[genre_col]
    
    # Remove rows with missing values
    features_combined = features_combined.dropna()
    
    print(f"\nFinal dataset shape: {features_combined.shape}")
    print(f"Genres: {features_combined['genre'].nunique()}")
    
    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'wb') as f:
        pickle.dump(features_combined, f)
    
    print(f"\nProcessed features saved to: {output_file}")
    
    return features_combined


def process_fma_dataset(
    audio_dir: str,
    metadata_path: str,
    output_path: str = 'data/processed/extracted_features.pkl',
    max_files: int = None,
    sr: int = 22050,
    duration: float = 30.0
):
    """
    Process FMA dataset: extract features from all audio files.
    
    Args:
        audio_dir: Directory containing audio files (e.g., data/raw/fma_small/)
        metadata_path: Path to tracks.csv with genre information
        output_path: Where to save extracted features
        max_files: Limit number of files (for testing)
        sr: Sample rate
        duration: Duration per file
    """
    print("=" * 70)
    print("PROCESSING FMA DATASET")
    print("=" * 70)
    
    # Resolve paths
    audio_path = resolve_path(audio_dir)
    metadata_file = resolve_path(metadata_path)
    output_file = resolve_path(output_path)
    
    # Validate
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio directory not found: {audio_path}")
    
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
    
    # 1. Load metadata
    print(f"\nLoading metadata from: {metadata_file}")
    tracks_df = pd.read_csv(metadata_file, index_col=0, header=[0, 1])

    # Get genre information
    genre_col = ('track', 'genre_top')
    if genre_col not in tracks_df.columns:
        print(f"Warning: {genre_col} not found in metadata")
        genre_col = tracks_df.columns[0]  # Use first column as fallback

    # 2. Find all audio files
    audio_path = Path(audio_dir)
    audio_files = list(audio_path.glob("**/*.mp3"))

    if max_files:
        audio_files = audio_files[:max_files]
        print(f"\nProcessing {max_files} files (test mode)")
    else:
        print(f"\nFound {len(audio_files)} audio files")

    # 3. Extract features using preprocessing.extract_features_batch()
    print("\nExtracting features...")
    features_list = extract_features_batch([str(f) for f in audio_files], sr=sr, duration=duration)

    # 4. Save to pickle file
    results = []
    for i, audio_file in enumerate(audio_files):
        try:
            # Get track ID from filename
            track_id = int(audio_file.stem)

            # Get genre
            if track_id in tracks_df.index:
                genre = tracks_df.loc[track_id, genre_col]
            else:
                genre = 'unknown'

            # Store result
            results.append({
                'track_id': track_id,
                'file_path': str(audio_file),
                'genre': genre,
                'features': features_list[i],
            })

        except Exception as e:
            print(f"\nError processing {audio_file}: {e}")
            continue

    print(f"\nSuccessfully processed {len(results)} files")

    # Convert to DataFrame
    df_results = pd.DataFrame(results)

    # Save to pickle
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'wb') as f:
        pickle.dump(df_results, f)
    
    print(f"\nFeatures saved to: {output_file}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("FEATURE EXTRACTION SUMMARY")
    print("=" * 70)
    print(f"Total tracks processed: {len(results)}")
    print(f"Feature vector size: {results[0]['features'].shape[0]}")
    print(f"\nGenre distribution:")
    print(df_results['genre'].value_counts())
    print("=" * 70)

    return df_results


def main():
    """CLI entry point with argparse"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Extract audio features for genre classification',
        epilog=f"""
Examples:
  # From project root
  python scripts/process_audio_files.py --mode precomputed \\
      --features-csv data/metadata/features.csv \\
      --output data/processed/features.pkl

  # From scripts directory
  cd scripts/
  python process_audio_files.py --mode precomputed \\
      --features-csv data/metadata/features.csv \\
      --output data/processed/features.pkl

  # Using absolute paths
  python scripts/process_audio_files.py --mode precomputed \\
      --features-csv /full/path/to/features.csv \\
      --output /full/path/to/output.pkl

Paths are resolved dynamically from:
  1. Current working directory
  2. Project root: {PROJECT_ROOT}
  3. Data directory: {DATA_DIR}
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--mode', choices=['single', 'batch', 'precomputed'], default='single',
                        help='Processing mode')
    parser.add_argument('--audio-file', type=str, help='Path to single audio file')
    parser.add_argument('--audio-dir', type=str, default='data/raw/fma_small',
                        help='Directory with audio files (relative or absolute)')
    parser.add_argument('--metadata', type=str, default='data/metadata/tracks.csv',
                        help='Path to tracks.csv (relative or absolute)')
    parser.add_argument('--features-csv', type=str, default='data/metadata/features.csv',
                        help='Path to pre-computed features.csv (relative or absolute)')
    parser.add_argument('--output', type=str, default='data/processed/features.pkl',
                        help='Output file path (relative or absolute)')
    parser.add_argument('--max-files', type=int, help='Maximum files to process (for testing)')

    args = parser.parse_args()
    
    # Debug: Show where script thinks project root is
    if os.getenv('DEBUG'):
        print(f"DEBUG: Script directory: {SCRIPT_DIR}")
        print(f"DEBUG: Project root: {PROJECT_ROOT}")
        print(f"DEBUG: Data directory: {DATA_DIR}")
        print(f"DEBUG: Current working directory: {Path.cwd()}")
        print()

    try:
        if args.mode == 'single':
            if not args.audio_file:
                print("Error: --audio-file required for single mode")
                sys.exit(1)

            process_single_file(args.audio_file)

        elif args.mode == 'batch':
            process_fma_dataset(
                audio_dir=args.audio_dir,
                metadata_path=args.metadata,
                output_path=args.output,
                max_files=args.max_files
            )

        elif args.mode == 'precomputed':
            process_from_precomputed_features(
                features_csv_path=args.features_csv,
                tracks_csv_path=args.metadata,
                output_path=args.output
            )
    
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print(f"\nTip: Run from project root or provide absolute paths")
        print(f"  Project root: {PROJECT_ROOT}")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
