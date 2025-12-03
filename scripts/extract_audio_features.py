"""
Extract meaningful audio features from FMA dataset
Creates features suitable for EDA visualizations and CNN modeling
"""

import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import pickle
import json
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import argparse
import warnings
warnings.filterwarnings('ignore')

# Project paths
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / 'data'
RAW_AUDIO_DIR = DATA_DIR / 'raw'
TRACKS_CSV = DATA_DIR / 'metadata' / 'tracks.csv'
OUTPUT_PKL = DATA_DIR / 'processed' / 'extracted_features.pkl'
SPECTROGRAMS_DIR = DATA_DIR / 'processed' / 'spectrograms'
TARGET_GENRES_CONFIG = PROJECT_ROOT / 'config' / 'target_genres.json'

# Audio parameters
SAMPLE_RATE = 22050
DURATION = 30  # seconds
N_MFCC = 20
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512


def get_audio_path(track_id):
    """Get path to audio file given track ID"""
    tid_str = f'{int(track_id):06d}'
    # Try multiple possible locations (matches prepare_cnn_spectrograms.py)
    possible_paths = [
        RAW_AUDIO_DIR / tid_str[:3] / f'{tid_str}.mp3',  # Direct in raw/
        RAW_AUDIO_DIR / 'fma_large' / tid_str[:3] / f'{tid_str}.mp3',  # FMA large structure
        RAW_AUDIO_DIR / 'fma_medium' / tid_str[:3] / f'{tid_str}.mp3',  # FMA medium structure
        RAW_AUDIO_DIR / 'fma_small' / tid_str[:3] / f'{tid_str}.mp3',  # FMA small structure
    ]
    for path in possible_paths:
        if path.exists():
            return path
    # Return first path as default (for error messages)
    return possible_paths[0]


def extract_features_from_audio(audio_path):
    """Extract comprehensive audio features from a single file"""
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, duration=DURATION)

        features = {}

        # 1. Temporal features
        features['duration'] = len(y) / sr
        features['rms_energy_mean'] = float(np.mean(librosa.feature.rms(y=y)))
        features['rms_energy_std'] = float(np.std(librosa.feature.rms(y=y)))

        # 2. Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y)
        features['zcr_mean'] = float(np.mean(zcr))
        features['zcr_std'] = float(np.std(zcr))

        # 3. Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
        features['spectral_centroid_std'] = float(np.std(spectral_centroids))

        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        features['spectral_rolloff_mean'] = float(np.mean(spectral_rolloff))
        features['spectral_rolloff_std'] = float(np.std(spectral_rolloff))

        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        features['spectral_bandwidth_mean'] = float(np.mean(spectral_bandwidth))
        features['spectral_bandwidth_std'] = float(np.std(spectral_bandwidth))

        # 4. MFCCs (20 coefficients with mean and std)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        for i in range(N_MFCC):
            features[f'mfcc_{i}_mean'] = float(np.mean(mfccs[i]))
            features[f'mfcc_{i}_std'] = float(np.std(mfccs[i]))

        # 5. Chroma features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features['chroma_mean'] = float(np.mean(chroma))
        features['chroma_std'] = float(np.std(chroma))

        # 6. Tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features['tempo'] = float(tempo)

        # 7. Mel spectrogram (for visualization and CNN input)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS,
                                                   n_fft=N_FFT, hop_length=HOP_LENGTH)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Store spectrogram statistics
        features['mel_spec_mean'] = float(np.mean(mel_spec_db))
        features['mel_spec_std'] = float(np.std(mel_spec_db))
        features['mel_spec_max'] = float(np.max(mel_spec_db))
        features['mel_spec_min'] = float(np.min(mel_spec_db))

        return features, mel_spec_db, y

    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None, None, None


def process_single_track(args_tuple):
    """
    Process a single track for multiprocessing.

    Args:
        args_tuple: (track_id, genre_value) tuple

    Returns:
        dict: Extracted features or None if failed
    """
    track_id, genre_value = args_tuple

    audio_path = get_audio_path(track_id)
    if not audio_path.exists():
        return None

    features, mel_spec, y = extract_features_from_audio(audio_path)
    if features is None:
        return None

    # Add metadata
    features['track_id'] = track_id
    features['genre'] = genre_value

    return features


def save_spectrogram_example(track_id, mel_spec, genre, save_dir):
    """Save spectrogram visualization for a sample track"""
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(mel_spec, x_axis='time', y_axis='mel',
                                    sr=SAMPLE_RATE, hop_length=HOP_LENGTH,
                                    fmax=8000, ax=ax)
    ax.set_title(f'Mel Spectrogram - {genre} (Track {track_id})', fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    fig.colorbar(img, ax=ax, format='%+2.0f dB')

    # Sanitize genre name for use in filename (remove invalid characters)
    # Use URL-style encoding to avoid conflicts: space→_, slash→-SLASH-, hyphen stays as-is
    safe_genre = genre.replace('/', '-SLASH-').replace('\\', '-SLASH-').replace(' ', '_')
    save_path = save_dir / f'{safe_genre}_track_{track_id}.png'
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    return save_path


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Extract audio features from FMA dataset')
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Number of parallel workers (default: CPU count)')
    parser.add_argument('--samples-per-genre', type=int, default=100,
                        help='Number of samples per genre to process (default: 100)')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum total samples to process (overrides samples-per-genre)')
    parser.add_argument('--use-subgenres', action='store_true',
                        help='Use detailed subgenres instead of top-level genres (50+ genres)')
    parser.add_argument('--min-samples-per-genre', type=int, default=10,
                        help='Minimum number of samples per genre to include (default: 10)')
    parser.add_argument('--num-spectrogram-examples', type=int, default=50,
                        help='Total number of spectrogram images to generate (default: 50)')
    parser.add_argument('--spectrograms-per-genre', type=int, default=None,
                        help='Number of spectrograms per genre (overrides num-spectrogram-examples for even distribution)')
    args = parser.parse_args()

    print("=" * 70)
    print("EXTRACTING AUDIO FEATURES FROM RAW FILES")
    print("=" * 70)

    # Detect audio directory structure
    print(f"\n1. Detecting audio file location...")
    audio_subdirs = []
    for subdir in ['fma_large', 'fma_medium', 'fma_small', '']:
        test_dir = RAW_AUDIO_DIR / subdir if subdir else RAW_AUDIO_DIR
        if test_dir.exists():
            mp3_files = list(test_dir.rglob("*.mp3"))
            if mp3_files:
                audio_subdirs.append((subdir if subdir else 'root', len(mp3_files)))

    if audio_subdirs:
        print(f"   Found audio files:")
        for subdir, count in audio_subdirs:
            print(f"     {subdir}: {count:,} files")
    else:
        print(f"   ERROR: No audio files found in {RAW_AUDIO_DIR}")
        print(f"   Please run download_fma.py first")
        return 1

    # Load track metadata
    print(f"\n2. Loading track metadata from: {TRACKS_CSV.relative_to(PROJECT_ROOT)}")
    tracks_df = pd.read_csv(TRACKS_CSV, index_col=0, header=[0, 1])

    # Get genre information based on user choice
    if args.use_subgenres:
        print("   Using subgenres (detailed genre classification)")
        # Use the 'genres' column which contains lists of genre IDs
        # First load the genres mapping
        genres_csv = DATA_DIR / "metadata" / "genres.csv"
        genres_df = pd.read_csv(genres_csv, index_col=0)

        # Get tracks with genres
        genre_col = ('track', 'genres')
        tracks_with_genre = tracks_df[tracks_df[genre_col].notna()].copy()

        # Parse the genres (they're stored as stringified lists like "[21, 456]")
        import ast
        tracks_with_genre['genre_ids'] = tracks_with_genre[genre_col].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else []
        )

        # Take the first genre for each track (or we could do multi-label for all)
        tracks_with_genre['genre_id'] = tracks_with_genre['genre_ids'].apply(
            lambda x: x[0] if x else None
        )
        tracks_with_genre = tracks_with_genre[tracks_with_genre['genre_id'].notna()]

        # Map genre IDs to genre titles
        genre_id_to_title = genres_df['title'].to_dict()
        tracks_with_genre['genre'] = tracks_with_genre['genre_id'].map(genre_id_to_title)

        # Remove any tracks where genre mapping failed
        tracks_with_genre = tracks_with_genre[tracks_with_genre['genre'].notna()]
    else:
        print("   Using top-level genres (16 main categories)")
        genre_col = ('track', 'genre_top')
        tracks_with_genre = tracks_df[tracks_df[genre_col].notna()].copy()
        tracks_with_genre['genre'] = tracks_with_genre[genre_col]

    print(f"   Found {len(tracks_with_genre)} tracks with genre labels")
    print(f"   Genres: {tracks_with_genre['genre'].nunique()}")

    # Filter to target genres for academic project
    if args.use_subgenres:
        # Load target genres from config
        if TARGET_GENRES_CONFIG.exists():
            with open(TARGET_GENRES_CONFIG, 'r') as f:
                genre_config = json.load(f)
                target_genres = genre_config['genres']
            print(f"   Filtering to {len(target_genres)} target genres from config...")
        else:
            print(f"   Warning: {TARGET_GENRES_CONFIG} not found, using all genres")
            target_genres = None
        
        if target_genres:
            tracks_with_genre = tracks_with_genre[tracks_with_genre['genre'].isin(target_genres)]
            print(f"   After genre filtering: {len(tracks_with_genre)} tracks, {tracks_with_genre['genre'].nunique()} genres")
            
            # Show which target genres were found
            found_genres = set(tracks_with_genre['genre'].unique())
            missing_genres = set(target_genres) - found_genres
            if missing_genres:
                print(f"   Note: {len(missing_genres)} target genres not found in dataset: {list(missing_genres)[:3]}...")

    # Filter out genres with too few samples
    if args.min_samples_per_genre > 0:
        genre_counts = tracks_with_genre['genre'].value_counts()
        valid_genres = genre_counts[genre_counts >= args.min_samples_per_genre].index
        tracks_with_genre = tracks_with_genre[tracks_with_genre['genre'].isin(valid_genres)]
        print(f"   After filtering (min {args.min_samples_per_genre} samples/genre): {len(tracks_with_genre)} tracks, {len(valid_genres)} genres")

    # Sample tracks for processing (use subset for faster processing)
    print(f"\n3. Sampling tracks ({args.samples_per_genre} per genre)...")

    sampled_tracks = []
    for genre in tracks_with_genre['genre'].unique():
        genre_tracks = tracks_with_genre[tracks_with_genre['genre'] == genre]
        n_samples = min(args.samples_per_genre, len(genre_tracks))
        sampled = genre_tracks.sample(n=n_samples, random_state=42)
        sampled_tracks.append(sampled)

    tracks_to_process = pd.concat(sampled_tracks)

    # Apply max_samples limit if specified
    if args.max_samples:
        tracks_to_process = tracks_to_process.head(args.max_samples)

    print(f"   Selected {len(tracks_to_process)} tracks for processing")

    # Prepare track data for multiprocessing
    track_data = []
    for track_id, row in tracks_to_process.iterrows():
        genre = row['genre']
        if isinstance(genre, pd.Series):
            genre_value = str(genre.iloc[0])
        elif isinstance(genre, str):
            genre_value = genre
        else:
            genre_value = str(genre)
        track_data.append((track_id, genre_value))

    # Extract features with multiprocessing
    print("\n4. Extracting audio features with multiprocessing...")
    num_workers = args.num_workers if args.num_workers else cpu_count()
    print(f"   Using {num_workers} parallel workers")
    SPECTROGRAMS_DIR.mkdir(parents=True, exist_ok=True)

    features_list = []

    with Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_single_track, track_data),
            total=len(track_data),
            desc="Processing audio"
        ))

    # Collect valid results
    for result in results:
        if result is not None:
            features_list.append(result)

    # Save spectrogram examples - do this separately for visualization
    print("\n   Generating example spectrograms...")
    spectrograms_saved = {}

    if args.spectrograms_per_genre:
        # Per-genre distribution: save N spectrograms per genre
        print(f"   Target: {args.spectrograms_per_genre} spectrograms per genre")
        for track_id, genre_value in tqdm(track_data, desc="Saving examples"):
            if genre_value not in spectrograms_saved:
                spectrograms_saved[genre_value] = 0

            if spectrograms_saved[genre_value] < args.spectrograms_per_genre:
                audio_path = get_audio_path(track_id)
                if audio_path.exists():
                    _, mel_spec, _ = extract_features_from_audio(audio_path)
                    if mel_spec is not None:
                        save_spectrogram_example(track_id, mel_spec, genre_value, SPECTROGRAMS_DIR)
                        spectrograms_saved[genre_value] += 1
    else:
        # Random mix: save up to num_spectrogram_examples total
        print(f"   Target: {args.num_spectrogram_examples} spectrograms (random mix)")
        total_saved = 0
        for track_id, genre_value in tqdm(track_data, desc="Saving examples"):
            if total_saved >= args.num_spectrogram_examples:
                break

            audio_path = get_audio_path(track_id)
            if audio_path.exists():
                _, mel_spec, _ = extract_features_from_audio(audio_path)
                if mel_spec is not None:
                    save_spectrogram_example(track_id, mel_spec, genre_value, SPECTROGRAMS_DIR)
                    if genre_value not in spectrograms_saved:
                        spectrograms_saved[genre_value] = 0
                    spectrograms_saved[genre_value] += 1
                    total_saved += 1

    # Create DataFrame
    print(f"\n5. Creating feature dataframe...")

    if not features_list:
        print("\n   ERROR: No features were extracted!")
        print("   This usually means audio files were not found.")
        print(f"   Check that audio files exist in: {RAW_AUDIO_DIR}")
        return 1
    df_features = pd.DataFrame(features_list)

    # Create genre_idx
    unique_genres = sorted(df_features['genre'].unique())
    genre_to_idx = {genre: idx for idx, genre in enumerate(unique_genres)}
    df_features['genre_idx'] = df_features['genre'].map(genre_to_idx)

    print(f"   Shape: {df_features.shape}")
    print(f"   Features: {len(df_features.columns) - 3}")  # Exclude track_id, genre, genre_idx
    print(f"   Genres: {len(unique_genres)}")

    print(f"\n   Genre distribution:")
    for genre, count in df_features['genre'].value_counts().sort_index().items():
        print(f"     {genre:20s}: {count:4d} tracks")

    # Save
    OUTPUT_PKL.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PKL, 'wb') as f:
        pickle.dump(df_features, f)

    # Save genre names file (preserving order from target_genres.json for consistency)
    # This ensures spectrogram filenames and genre indices match
    if args.use_subgenres and TARGET_GENRES_CONFIG.exists():
        with open(TARGET_GENRES_CONFIG, 'r') as f:
            genre_config = json.load(f)
            config_genres = genre_config['genres']
        
        # Filter to only genres that actually have spectrograms
        genres_with_data = [g for g in config_genres if g in df_features['genre'].values]
        
        genre_names_file = OUTPUT_PKL.parent / f'genre_names_{len(genres_with_data)}.json'
        with open(genre_names_file, 'w') as f:
            json.dump(genres_with_data, f, indent=2)
        print(f"\nSaved genre names: {genre_names_file.relative_to(PROJECT_ROOT)}")
        print(f"  Genre order preserved from config file (not alphabetically sorted)")

    print("\nSUCCESS!")
    print(f"  Features: {OUTPUT_PKL.relative_to(PROJECT_ROOT)}")
    print(f"  Size: {OUTPUT_PKL.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"  Spectrograms: {SPECTROGRAMS_DIR.relative_to(PROJECT_ROOT)}/")
    print(f"  Count: {sum(spectrograms_saved.values())} example spectrograms saved")
    print("=" * 70)

    print("\nNext steps:")
    print("  - Prepare CNN spectrograms: python scripts/prepare_cnn_spectrograms.py")
    print("  - Train model: python scripts/train_multilabel_cnn.py")
    print("  - Run EDA notebook: jupyter notebook notebooks/01_EDA.ipynb")
    print("  - View spectrograms in:", SPECTROGRAMS_DIR.relative_to(PROJECT_ROOT))

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
