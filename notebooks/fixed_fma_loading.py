# Fixed FMA dataset loading code for 01_EDA.ipynb
# This replaces cell ID 7e56f7fd

# Load REAL audio dataset (FMA) or generate synthetic data
import pandas as pd
import numpy as np
from pathlib import Path

# Define data paths
DATA_DIR = Path("../data")
FEATURES_PATH = DATA_DIR / "metadata" / "features.csv"
TRACKS_PATH = DATA_DIR / "metadata" / "tracks.csv"

# Check if real data exists
if FEATURES_PATH.exists() and TRACKS_PATH.exists():
    print("✓ Found FMA dataset - loading real data")

    # Load pre-computed features (3-level columns)
    features_df = pd.read_csv(FEATURES_PATH, index_col=0, header=[0, 1, 2])

    # Load track metadata (2-level columns)
    tracks_df = pd.read_csv(TRACKS_PATH, index_col=0, header=[0, 1])

    print(f"  Features shape: {features_df.shape}")
    print(f"  Tracks shape: {tracks_df.shape}")

    # Build flat feature dataframe with proper column names
    # Extract features matching preprocessing/feature_extraction.py
    data_dict = {}

    # 1. MFCCs (13 coefficients)
    if 'mfcc' in features_df.columns.levels[0]:
        mfcc_data = features_df['mfcc']
        # Get all MFCC means (or first 13 columns)
        for i in range(min(13, len(mfcc_data.columns))):
            col_name = mfcc_data.columns[i]
            # Use 'mean' stat if available, otherwise just use the column
            if isinstance(col_name, tuple) and 'mean' in str(col_name):
                data_dict[f'mfcc_{i}'] = mfcc_data[col_name]
            elif len(mfcc_data.columns) > i:
                # Just take the i-th column and average if needed
                col_data = mfcc_data.iloc[:, i]
                data_dict[f'mfcc_{i}'] = col_data

    # 2. Spectral centroid
    if 'spectral_centroid' in features_df.columns.levels[0]:
        spec_cent = features_df['spectral_centroid']
        # Find 'mean' column
        for col in spec_cent.columns:
            if isinstance(col, tuple) and 'mean' in str(col):
                data_dict['spectral_centroid'] = spec_cent[col]
                break
        else:
            # Just take mean of all columns
            data_dict['spectral_centroid'] = spec_cent.mean(axis=1)

    # 3. Spectral rolloff
    if 'spectral_rolloff' in features_df.columns.levels[0]:
        spec_roll = features_df['spectral_rolloff']
        for col in spec_roll.columns:
            if isinstance(col, tuple) and 'mean' in str(col):
                data_dict['spectral_rolloff'] = spec_roll[col]
                break
        else:
            data_dict['spectral_rolloff'] = spec_roll.mean(axis=1)

    # 4. Zero crossing rate (zcr)
    if 'zcr' in features_df.columns.levels[0]:
        zcr_data = features_df['zcr']
        for col in zcr_data.columns:
            if isinstance(col, tuple) and 'mean' in str(col):
                data_dict['zero_crossing_rate'] = zcr_data[col]
                break
        else:
            data_dict['zero_crossing_rate'] = zcr_data.mean(axis=1)

    # 5. RMS energy (rmse or rms)
    for rms_name in ['rmse', 'rms']:
        if rms_name in features_df.columns.levels[0]:
            rms_data = features_df[rms_name]
            for col in rms_data.columns:
                if isinstance(col, tuple) and 'mean' in str(col):
                    data_dict['rms_energy'] = rms_data[col]
                    break
            else:
                data_dict['rms_energy'] = rms_data.mean(axis=1)
            break

    # 6. Chroma features (12 coefficients)
    for chroma_name in ['chroma_stft', 'chroma_cens', 'chroma_cqt', 'chroma']:
        if chroma_name in features_df.columns.levels[0]:
            chroma_data = features_df[chroma_name]
            # Get all chroma means (or first 12 columns)
            for i in range(min(12, len(chroma_data.columns))):
                col_name = chroma_data.columns[i]
                if isinstance(col_name, tuple) and 'mean' in str(col_name):
                    data_dict[f'chroma_{i}'] = chroma_data[col_name]
                elif len(chroma_data.columns) > i:
                    col_data = chroma_data.iloc[:, i]
                    data_dict[f'chroma_{i}'] = col_data
            break

    # Create flat dataframe from extracted features
    df = pd.DataFrame(data_dict, index=features_df.index)

    # Add genre from tracks (avoiding the merge error)
    genre_col = ('track', 'genre_top')
    if genre_col in tracks_df.columns:
        # Direct assignment using index alignment
        df['genre'] = tracks_df[genre_col]
    else:
        # Try to find genre column
        genre_candidates = [col for col in tracks_df.columns if 'genre' in str(col).lower()]
        if genre_candidates:
            df['genre'] = tracks_df[genre_candidates[0]]
        else:
            print("  Warning: Could not find genre column")
            df['genre'] = 'Unknown'

    # Create genre_idx (numeric labels)
    unique_genres = df['genre'].dropna().unique()
    genre_to_idx = {genre: idx for idx, genre in enumerate(sorted(unique_genres))}
    df['genre_idx'] = df['genre'].map(genre_to_idx)

    # Drop rows with missing genres
    df = df.dropna(subset=['genre', 'genre_idx'])
    df['genre_idx'] = df['genre_idx'].astype(int)

    # Update GENRES list to match real data
    GENRES = sorted(unique_genres)

    # Define feature_cols (all numeric columns except genre_idx)
    feature_cols = [col for col in df.columns if col not in ['genre', 'genre_idx']]

    print(f"\nLoaded {len(df)} tracks with real audio features")
    print(f"  Genres ({len(GENRES)}): {GENRES[:5]}... (showing first 5)")
    print(f"  Features ({len(feature_cols)}): {feature_cols[:6]}... (showing first 6)")

else:
    print("✗ FMA dataset not found - generating synthetic data")
    print("To use real data:")
    print("  1. See data/README.md for download instructions")
    print("  2. Run: python scripts/process_audio_files.py --mode precomputed")
    print()

    # Generate synthetic data with ALL features matching preprocessing/feature_extraction.py
    np.random.seed(42)
    dataset = []

    for genre_idx, genre in enumerate(GENRES):
        for sample_idx in range(100):
            sample = {
                'file_id': f"{genre}_{sample_idx:03d}",
                'genre': genre,
                'genre_idx': genre_idx,
            }

            # Add 13 MFCC features
            for i in range(13):
                sample[f'mfcc_{i}'] = np.random.normal(-5 - genre_idx + i*0.5, 2)

            # Add spectral features
            sample['spectral_centroid'] = np.random.normal(2000 + genre_idx * 200, 500)
            sample['spectral_rolloff'] = np.random.normal(7000 + genre_idx * 300, 1500)
            sample['zero_crossing_rate'] = np.random.beta(2, 5) + (genre_idx * 0.01)
            sample['rms_energy'] = np.random.gamma(2, 0.05) + (genre_idx * 0.02)

            # Add 12 chroma features
            for i in range(12):
                sample[f'chroma_{i}'] = np.random.normal(0.3 + genre_idx * 0.01 + i*0.02, 0.1)

            dataset.append(sample)

    df = pd.DataFrame(dataset)

    # Define feature_cols (all numeric columns except genre_idx and file_id)
    feature_cols = [col for col in df.columns if col not in ['file_id', 'genre', 'genre_idx']]

    print(f"Generated synthetic dataset: {df.shape}")
    print(f"  Features ({len(feature_cols)}): {feature_cols[:6]}... (showing first 6)")
    print(f"  Note: Results are for demonstration only")
