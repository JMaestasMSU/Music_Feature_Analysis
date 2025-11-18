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
from pathlib import Path
from tqdm import tqdm
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
    return RAW_AUDIO_DIR / tid_str[:3] / f'{tid_str}.mp3'


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

    save_path = save_dir / f'{genre}_track_{track_id}.png'
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    return save_path


def main():
    print("=" * 70)
    print("EXTRACTING AUDIO FEATURES FROM RAW FILES")
    print("=" * 70)

    # Load track metadata
    print(f"\n1. Loading track metadata from: {TRACKS_CSV.relative_to(PROJECT_ROOT)}")
    tracks_df = pd.read_csv(TRACKS_CSV, index_col=0, header=[0, 1])

    # Get genre information
    genre_col = ('track', 'genre_top')
    tracks_with_genre = tracks_df[tracks_df[genre_col].notna()].copy()
    tracks_with_genre['genre'] = tracks_with_genre[genre_col]

    print(f"   Found {len(tracks_with_genre)} tracks with genre labels")
    print(f"   Genres: {tracks_with_genre['genre'].nunique()}")

    # Sample tracks for processing (use subset for faster processing)
    # Take 100 tracks per genre for balanced dataset
    print("\n2. Sampling tracks (100 per genre for speed)...")

    sampled_tracks = []
    for genre in tracks_with_genre['genre'].unique():
        genre_tracks = tracks_with_genre[tracks_with_genre['genre'] == genre]
        n_samples = min(100, len(genre_tracks))
        sampled = genre_tracks.sample(n=n_samples, random_state=42)
        sampled_tracks.append(sampled)

    tracks_to_process = pd.concat(sampled_tracks)
    print(f"   Selected {len(tracks_to_process)} tracks for processing")

    # Extract features
    print("\n3. Extracting audio features...")
    SPECTROGRAMS_DIR.mkdir(parents=True, exist_ok=True)

    features_list = []
    spectrograms_saved = {}

    for idx, (track_id, row) in enumerate(tqdm(tracks_to_process.iterrows(),
                                                 total=len(tracks_to_process),
                                                 desc="Processing audio")):
        audio_path = get_audio_path(track_id)

        if not audio_path.exists():
            continue

        features, mel_spec, y = extract_features_from_audio(audio_path)

        if features is None:
            continue

        # Add metadata
        features['track_id'] = track_id

        # Extract genre (handle both string and Series)
        genre = row['genre']
        if isinstance(genre, pd.Series):
            genre_value = str(genre.iloc[0])
        elif isinstance(genre, str):
            genre_value = genre
        else:
            genre_value = str(genre)

        features['genre'] = genre_value

        features_list.append(features)

        # Save spectrogram examples (5 per genre)
        if genre_value not in spectrograms_saved:
            spectrograms_saved[genre_value] = 0

        if spectrograms_saved[genre_value] < 5 and mel_spec is not None:
            save_spectrogram_example(track_id, mel_spec, genre_value, SPECTROGRAMS_DIR)
            spectrograms_saved[genre_value] += 1

    # Create DataFrame
    print(f"\n4. Creating feature dataframe...")
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

    print("\nSUCCESS!")
    print(f"  Features: {OUTPUT_PKL.relative_to(PROJECT_ROOT)}")
    print(f"  Size: {OUTPUT_PKL.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"  Spectrograms: {SPECTROGRAMS_DIR.relative_to(PROJECT_ROOT)}/")
    print(f"  Count: {sum(spectrograms_saved.values())} example spectrograms saved")
    print("=" * 70)

    print("\nNext steps:")
    print("  - Run EDA notebook: jupyter notebook notebooks/01_EDA.ipynb")
    print("  - View spectrograms in:", SPECTROGRAMS_DIR.relative_to(PROJECT_ROOT))

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
