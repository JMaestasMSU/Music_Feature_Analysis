#!/usr/bin/env python3
"""
Prepare full mel spectrograms for CNN training.
Extracts and saves complete 2D spectrograms (not just summary statistics).
"""

import pickle
import json
import numpy as np
import librosa
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse
from multiprocessing import Pool, cpu_count


# Audio processing parameters
SAMPLE_RATE = 22050
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
DURATION = 30  # seconds (trim/pad to this length)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRACKS_CSV = PROJECT_ROOT / "data" / "metadata" / "tracks.csv"
RAW_AUDIO_DIR = PROJECT_ROOT / "data" / "raw"


def get_audio_path(track_id):
    """Convert track ID to file path"""
    tid_str = f'{track_id:06d}'
    # Try multiple possible locations
    possible_paths = [
        RAW_AUDIO_DIR / tid_str[:3] / f'{tid_str}.mp3',  # Original FMA small structure
        RAW_AUDIO_DIR / 'fma_large' / tid_str[:3] / f'{tid_str}.mp3',  # FMA large structure
        RAW_AUDIO_DIR / 'fma_medium' / tid_str[:3] / f'{tid_str}.mp3',  # FMA medium structure
    ]
    for path in possible_paths:
        if path.exists():
            return path
    # Return first path as default (for error messages)
    return possible_paths[0]


def extract_mel_spectrogram(audio_path, target_length=None):
    """
    Extract mel spectrogram from audio file.
    
    Args:
        audio_path: Path to audio file
        target_length: Target number of time frames (for padding/trimming)
    
    Returns:
        mel_spec_db: Mel spectrogram in dB (n_mels, time_frames)
    """
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, duration=DURATION)
        
        # Pad if too short
        if len(y) < SAMPLE_RATE * DURATION:
            y = np.pad(y, (0, SAMPLE_RATE * DURATION - len(y)), mode='constant')
        
        # Extract mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
        )
        
        # Convert to dB scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Pad or trim to target length if specified
        if target_length is not None:
            current_length = mel_spec_db.shape[1]
            if current_length < target_length:
                # Pad with minimum value
                pad_width = target_length - current_length
                mel_spec_db = np.pad(
                    mel_spec_db, 
                    ((0, 0), (0, pad_width)), 
                    mode='constant', 
                    constant_values=mel_spec_db.min()
                )
            elif current_length > target_length:
                # Trim
                mel_spec_db = mel_spec_db[:, :target_length]
        
        return mel_spec_db
        
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None


def process_single_track(args_tuple):
    """Process a single track (for multiprocessing)."""
    track_id, genre, expected_frames = args_tuple
    
    audio_path = get_audio_path(track_id)
    if not audio_path.exists():
        return None
    
    mel_spec = extract_mel_spectrogram(audio_path, target_length=expected_frames)
    if mel_spec is None:
        return None
    
    return (track_id, mel_spec, genre)


def main():
    parser = argparse.ArgumentParser(description='Prepare CNN spectrograms from audio files')
    parser.add_argument('--output-dir', type=str, default='data/processed',
                        help='Output directory')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum number of samples to process (for testing)')
    parser.add_argument('--multi-label', action='store_true',
                        help='Create multi-label format (one-hot encoding)')
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Number of parallel workers (default: CPU count)')
    parser.add_argument('--use-subgenres', action='store_true',
                        help='Use detailed subgenres instead of top-level genres (gives 50+ genres)')
    parser.add_argument('--min-samples-per-genre', type=int, default=10,
                        help='Minimum number of samples per genre to include')
    parser.add_argument('--chunk-size', type=int, default=5000,
                        help='Process and save in chunks to avoid memory issues (default: 5000)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("PREPARING CNN SPECTROGRAMS")
    print("=" * 70)
    
    # Load track metadata
    print(f"\n1. Loading track metadata from: {TRACKS_CSV.relative_to(PROJECT_ROOT)}")
    tracks_df = pd.read_csv(TRACKS_CSV, index_col=0, header=[0, 1])
    
    # Get tracks with genre labels
    if args.use_subgenres:
        print("   Using subgenres (detailed genre classification)")
        # Use the 'genres' column which contains lists of genre IDs
        # First load the genres mapping
        genres_csv = PROJECT_ROOT / "data" / "metadata" / "genres.csv"
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
    
    if args.max_samples:
        tracks_with_genre = tracks_with_genre.head(args.max_samples)
    
    print(f"   Found {len(tracks_with_genre)} tracks with genre labels")
    print(f"   Genres: {tracks_with_genre['genre'].nunique()}")
    
    # Filter out genres with too few samples
    if args.min_samples_per_genre > 0:
        genre_counts = tracks_with_genre['genre'].value_counts()
        valid_genres = genre_counts[genre_counts >= args.min_samples_per_genre].index
        tracks_with_genre = tracks_with_genre[tracks_with_genre['genre'].isin(valid_genres)]
        print(f"   After filtering (min {args.min_samples_per_genre} samples/genre): {len(tracks_with_genre)} tracks, {len(valid_genres)} genres")
    
    # Create genre mapping
    unique_genres = sorted(tracks_with_genre['genre'].unique())
    genre_to_idx = {genre: idx for idx, genre in enumerate(unique_genres)}
    num_genres = len(unique_genres)
    
    print(f"\n2. Genre mapping ({num_genres} genres):")
    for genre, idx in sorted(genre_to_idx.items(), key=lambda x: x[1]):
        count = (tracks_with_genre['genre'] == genre).sum()
        print(f"     {idx:2d}. {genre:20s}: {count:4d} tracks")
    
    # Extract spectrograms
    print(f"\n3. Extracting mel spectrograms...")
    print(f"   Parameters:")
    print(f"     Sample rate: {SAMPLE_RATE} Hz")
    print(f"     N_mels: {N_MELS}")
    print(f"     N_FFT: {N_FFT}")
    print(f"     Hop length: {HOP_LENGTH}")
    print(f"     Duration: {DURATION}s")
    
    spectrograms = []
    labels = []
    valid_track_ids = []
    
    # Calculate expected spectrogram shape
    expected_frames = int(np.ceil(SAMPLE_RATE * DURATION / HOP_LENGTH))
    print(f"     Expected shape per spectrogram: ({N_MELS}, {expected_frames})")
    
    # Prepare processing arguments
    num_workers = args.num_workers if args.num_workers else cpu_count()
    print(f"     Using {num_workers} parallel workers")
    
    # Prepare track data for multiprocessing
    track_data = []
    for track_id in tracks_with_genre.index:
        genre_value = tracks_with_genre.loc[track_id, 'genre']
        if hasattr(genre_value, 'iloc'):
            genre = str(genre_value.iloc[0]).strip()
        else:
            genre = str(genre_value).strip()
        track_data.append((track_id, genre, expected_frames))
    
    total_tracks = len(track_data)
    chunk_size = args.chunk_size
    
    print(f"\n   Processing {total_tracks} tracks in chunks of {chunk_size}...")
    print(f"   This will save incrementally to avoid memory issues.")
    
    # Process in chunks
    all_spectrograms = []
    all_labels = []
    chunk_num = 0
    
    for chunk_start in range(0, total_tracks, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_tracks)
        chunk_data = track_data[chunk_start:chunk_end]
        chunk_num += 1
        
        print(f"\n   Chunk {chunk_num}/{(total_tracks + chunk_size - 1) // chunk_size}: Processing tracks {chunk_start} to {chunk_end}")
        
        with Pool(num_workers) as pool:
            results = list(tqdm(
                pool.imap(process_single_track, chunk_data),
                total=len(chunk_data),
                desc=f"Chunk {chunk_num}"
            ))
        
        # Collect valid results from this chunk
        for result in results:
            if result is None:
                continue
            
            track_id, mel_spec, genre = result
            all_spectrograms.append(mel_spec)
            valid_track_ids.append(track_id)
            
            if args.multi_label:
                # One-hot encoding
                label = np.zeros(num_genres, dtype=np.float32)
                label[genre_to_idx[genre]] = 1.0
                all_labels.append(label)
            else:
                # Single label (class index)
                all_labels.append(genre_to_idx[genre])
    
    num_valid = len(all_spectrograms)
    if num_valid == 0:
        print("\n ERROR: No valid spectrograms were processed!")
        return
    
    print(f"\n4. Successfully processed {num_valid} tracks")
    print(f"   Creating final arrays...")
    
    # Convert to numpy arrays
    spectrograms = np.array(all_spectrograms, dtype=np.float32)
    labels = np.array(all_labels)
    
    print(f"   Spectrograms shape: {spectrograms.shape}")
    print(f"   Labels shape: {labels.shape}")
    print(f"   Memory usage: ~{spectrograms.nbytes / (1024**3):.2f} GB")
    
    # Save data
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n5. Saving data to: {output_dir}")
    
    # Save spectrograms
    spec_path = output_dir / 'spectrograms.npy'
    np.save(spec_path, spectrograms)
    print(f"   Saved spectrograms: {spec_path}")
    print(f"     Size: {spec_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Save labels
    if args.multi_label:
        labels_path = output_dir / 'labels_multilabel.npy'
    else:
        labels_path = output_dir / 'labels.npy'
    np.save(labels_path, labels)
    print(f"   Saved labels: {labels_path}")
    
    # Save genre names
    genre_names_path = output_dir / f'genre_names_{num_genres}.json'
    with open(genre_names_path, 'w') as f:
        json.dump(unique_genres, f, indent=2)
    print(f"   Saved genre names: {genre_names_path}")
    
    # Save track IDs for reference
    track_ids_path = output_dir / 'track_ids.json'
    with open(track_ids_path, 'w') as f:
        json.dump([int(tid) for tid in valid_track_ids], f, indent=2)
    print(f"   Saved track IDs: {track_ids_path}")
    
    print("\n" + "=" * 70)
    print("SUCCESS!")
    print("=" * 70)
    print(f"\nDataset summary:")
    print(f"  Samples: {len(spectrograms)}")
    print(f"  Spectrogram shape: {spectrograms.shape[1:]}")
    print(f"  Number of genres: {num_genres}")
    print(f"  Label format: {'Multi-label' if args.multi_label else 'Single-label'}")
    
    print(f"\nNext steps:")
    if args.multi_label:
        print(f"  python scripts/train_multilabel_cnn.py --num-genres {num_genres} --labels-file labels_multilabel.npy --genre-names-file genre_names_{num_genres}.json")
    else:
        print(f"  python scripts/train_multilabel_cnn.py --num-genres {num_genres} --multi-label false")


if __name__ == '__main__':
    main()
