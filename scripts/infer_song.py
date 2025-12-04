"""
Quick inference script to predict genres for a single audio file.
Usage: python infer_song.py path/to/song.mp3
"""

import sys
import torch
import numpy as np
import librosa
import json
import yaml
from pathlib import Path

# Check arguments
if len(sys.argv) < 2:
    print("Usage: python infer_song.py path/to/song.mp3")
    print("\nExample:")
    print("  python infer_song.py data/raw/fma_large/000/000002.mp3")
    sys.exit(1)

audio_path = sys.argv[1]
if not Path(audio_path).exists():
    print(f"Error: File not found: {audio_path}")
    sys.exit(1)

# Setup
PROJECT_ROOT = Path(__file__).parent
MODEL_DIR = PROJECT_ROOT / "models" / "trained_models" / "multilabel_cnn_filtered_improved"

print("="*80)
print("GENRE PREDICTION")
print("="*80)
print(f"\nAudio file: {audio_path}")

# Load config and genre names
with open(MODEL_DIR / "config.yaml", 'r') as f:
    config = yaml.safe_load(f)

with open(MODEL_DIR / "genre_names.json", 'r') as f:
    genre_names = json.load(f)

print(f"Model: {config['experiment_name']}")
print(f"Genres: {len(genre_names)}")

# Load audio and create spectrogram
print("\nProcessing audio...")
SAMPLE_RATE = 22050
N_MELS = 128
DURATION = 30

try:
    # Load audio
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, duration=DURATION)

    # Pad or trim to 30 seconds
    target_length = SAMPLE_RATE * DURATION
    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)))
    else:
        y = y[:target_length]

    # Create mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=SAMPLE_RATE,
        n_mels=N_MELS,
        n_fft=2048,
        hop_length=512
    )

    # Convert to log scale (dB)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Resize to 128x128
    if mel_spec_db.shape[1] != 128:
        from scipy.ndimage import zoom
        zoom_factor = 128 / mel_spec_db.shape[1]
        mel_spec_db = zoom(mel_spec_db, (1, zoom_factor))

    print(f"Spectrogram shape: {mel_spec_db.shape}")

except Exception as e:
    print(f"Error processing audio: {e}")
    sys.exit(1)

# Load model
print("\nLoading model...")
sys.path.append(str(PROJECT_ROOT / "models"))
from cnn_model import MultiLabelAudioCNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

model = MultiLabelAudioCNN(
    num_genres=len(genre_names),
    base_channels=config['base_channels'],
    use_attention=config['use_attention']
)

# Load checkpoint
checkpoint_path = MODEL_DIR / "best_model.pt"
checkpoint = torch.load(checkpoint_path, map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
model = model.to(device)

print(f"Model loaded (epoch {checkpoint['epoch']}, val_F1={checkpoint['val_f1']:.3f})")

# Predict
print("\nPredicting genres...")
with torch.no_grad():
    # Convert to tensor [1, 1, 128, 128]
    spec_tensor = torch.FloatTensor(mel_spec_db).unsqueeze(0).unsqueeze(0).to(device)

    # Forward pass
    outputs = model(spec_tensor)
    probs = torch.sigmoid(outputs).cpu().numpy()[0]

# Get predictions
threshold = config.get('prediction_threshold', 0.5)
top_k = config.get('top_k', 5)

# Sort by probability
sorted_indices = np.argsort(probs)[::-1]

print("\n" + "="*80)
print("PREDICTIONS")
print("="*80)

print(f"\nTop {top_k} genres (by confidence):")
for i, idx in enumerate(sorted_indices[:top_k], 1):
    genre = genre_names[idx]
    confidence = probs[idx] * 100
    bar = "█" * int(confidence / 5)
    print(f"{i}. {genre:30s} {confidence:5.1f}% {bar}")

print(f"\nGenres above threshold ({threshold}):")
above_threshold = [(genre_names[i], probs[i]) for i in range(len(probs)) if probs[i] >= threshold]
if above_threshold:
    for genre, prob in sorted(above_threshold, key=lambda x: x[1], reverse=True):
        print(f"  - {genre:30s} {prob*100:5.1f}%")
else:
    print(f"  (None above {threshold})")

print("\n" + "="*80)
