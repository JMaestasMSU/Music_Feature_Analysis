#!/usr/bin/env python3
"""
Prepare training data for CNN model.
Converts extracted features (mel spectrograms) from pickle format to numpy arrays.
"""

import pickle
import json
import numpy as np
from pathlib import Path
import argparse


def load_extracted_features(pkl_path):
    """Load the extracted features pickle file."""
    print(f"Loading features from: {pkl_path}")
    with open(pkl_path, 'rb') as f:
        df = pickle.load(f)
    print(f"  Loaded {len(df)} tracks with {len(df.columns)} columns")
    return df


def prepare_single_label_data(df, output_dir):
    """
    Prepare data for single-label classification.
    Each track has one genre label.
    """
    print("\nPreparing single-label data...")
    
    # Extract mel spectrograms
    mel_columns = [col for col in df.columns if col.startswith('mel_')]
    if not mel_columns:
        raise ValueError("No mel spectrogram columns found in dataframe")
    
    spectrograms = df[mel_columns].values
    print(f"  Spectrograms shape (before reshape): {spectrograms.shape}")
    
    # Reshape to (n_samples, height, width) for CNN
    # Assuming mel spectrograms are flattened, need to determine dimensions
    # Standard mel spectrogram: 128 mel bins x variable time frames
    n_samples = len(spectrograms)
    n_features = spectrograms.shape[1]
    
    # Common mel spectrogram configurations
    # Try to infer shape (128 mel bins is most common)
    n_mels = 128
    n_frames = n_features // n_mels
    
    if n_features == n_mels * n_frames:
        spectrograms = spectrograms.reshape(n_samples, n_mels, n_frames)
        print(f"  Reshaped spectrograms to: {spectrograms.shape}")
    else:
        print(f"  Warning: Could not reshape spectrograms cleanly")
        print(f"  n_features={n_features}, n_mels={n_mels}, n_frames={n_frames}")
        # Keep as 2D for now
        spectrograms = spectrograms.reshape(n_samples, -1, 1)
        print(f"  Using fallback shape: {spectrograms.shape}")
    
    # Extract labels
    if 'genre_idx' in df.columns:
        labels = df['genre_idx'].values
    else:
        # Create genre indices
        unique_genres = sorted(df['genre'].unique())
        genre_to_idx = {genre: idx for idx, genre in enumerate(unique_genres)}
        labels = df['genre'].map(genre_to_idx).values
    
    print(f"  Labels shape: {labels.shape}")
    print(f"  Number of unique genres: {len(np.unique(labels))}")
    
    # Get genre names
    if 'genre' in df.columns:
        unique_genres = sorted(df['genre'].unique())
        genre_names = {i: genre for i, genre in enumerate(unique_genres)}
    else:
        genre_names = {i: f"Genre_{i}" for i in range(len(np.unique(labels)))}
    
    return spectrograms, labels, genre_names


def prepare_multi_label_data(df, output_dir):
    """
    Prepare data for multi-label classification.
    Each track can have multiple genre labels.
    
    Note: This requires genre information to be structured as multi-label.
    If your data only has single labels, this will convert to one-hot encoding.
    """
    print("\nPreparing multi-label data...")
    
    # Extract mel spectrograms
    mel_columns = [col for col in df.columns if col.startswith('mel_')]
    if not mel_columns:
        raise ValueError("No mel spectrogram columns found in dataframe")
    
    spectrograms = df[mel_columns].values
    print(f"  Spectrograms shape (before reshape): {spectrograms.shape}")
    
    # Reshape spectrograms
    n_samples = len(spectrograms)
    n_features = spectrograms.shape[1]
    n_mels = 128
    n_frames = n_features // n_mels
    
    if n_features == n_mels * n_frames:
        spectrograms = spectrograms.reshape(n_samples, n_mels, n_frames)
        print(f"  Reshaped spectrograms to: {spectrograms.shape}")
    else:
        spectrograms = spectrograms.reshape(n_samples, -1, 1)
        print(f"  Using fallback shape: {spectrograms.shape}")
    
    # Create multi-label encoding
    # For now, convert single labels to one-hot (multi-label format)
    unique_genres = sorted(df['genre'].unique())
    num_genres = len(unique_genres)
    genre_to_idx = {genre: idx for idx, genre in enumerate(unique_genres)}
    
    # Create one-hot encoded labels
    labels = np.zeros((n_samples, num_genres), dtype=np.float32)
    for i, genre in enumerate(df['genre']):
        genre_idx = genre_to_idx[genre]
        labels[i, genre_idx] = 1.0
    
    print(f"  Labels shape: {labels.shape}")
    print(f"  Number of genres: {num_genres}")
    
    # Create genre names list
    genre_names = unique_genres
    
    return spectrograms, labels, genre_names


def save_training_data(spectrograms, labels, genre_names, output_dir, multi_label=False):
    """Save prepared data to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving data to: {output_dir}")
    
    # Save spectrograms
    spec_path = output_dir / 'spectrograms.npy'
    np.save(spec_path, spectrograms)
    print(f"  Saved spectrograms: {spec_path}")
    print(f"    Shape: {spectrograms.shape}")
    print(f"    Size: {spec_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Save labels
    if multi_label:
        labels_path = output_dir / 'labels_multilabel.npy'
    else:
        labels_path = output_dir / 'labels.npy'
    np.save(labels_path, labels)
    print(f"  Saved labels: {labels_path}")
    print(f"    Shape: {labels.shape}")
    
    # Save genre names
    if multi_label:
        genre_path = output_dir / 'genre_names_50.json'  # Adjust name based on actual count
    else:
        genre_path = output_dir / 'genre_names.json'
    
    # Convert genre_names to proper format
    if isinstance(genre_names, dict):
        genre_list = [genre_names[i] for i in range(len(genre_names))]
    else:
        genre_list = list(genre_names)
    
    with open(genre_path, 'w') as f:
        json.dump(genre_list, f, indent=2)
    print(f"  Saved genre names: {genre_path}")
    print(f"    Count: {len(genre_list)}")
    
    print("\n" + "=" * 70)
    print("SUCCESS! Training data prepared.")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Prepare training data from extracted features')
    parser.add_argument('--input', type=str, default='data/processed/extracted_features.pkl',
                        help='Path to extracted features pickle file')
    parser.add_argument('--output-dir', type=str, default='data/processed',
                        help='Output directory for processed data')
    parser.add_argument('--multi-label', action='store_true',
                        help='Prepare data for multi-label classification')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("PREPARING TRAINING DATA")
    print("=" * 70)
    
    # Load features
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    df = load_extracted_features(input_path)
    
    # Prepare data based on classification type
    if args.multi_label:
        spectrograms, labels, genre_names = prepare_multi_label_data(df, args.output_dir)
    else:
        spectrograms, labels, genre_names = prepare_single_label_data(df, args.output_dir)
    
    # Save data
    save_training_data(spectrograms, labels, genre_names, args.output_dir, args.multi_label)
    
    print("\nNext steps:")
    if args.multi_label:
        print("  - Train model: python scripts/train_multilabel_cnn.py --multi-label true")
    else:
        print("  - Train model: python scripts/train_multilabel_cnn.py --multi-label false")


if __name__ == '__main__':
    main()
