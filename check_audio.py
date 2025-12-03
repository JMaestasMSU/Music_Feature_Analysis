import pandas as pd
from pathlib import Path

# Load tracks
tracks = pd.read_csv('data/metadata/tracks.csv', index_col=0, header=[0,1])
genre_col = ('track', 'genre_top')
sample_tracks = tracks[tracks[genre_col].notna()].head(20)

def get_audio_path(track_id):
    tid_str = f'{track_id:06d}'
    return Path('data/raw') / tid_str[:3] / f'{tid_str}.mp3'

print("Checking if audio files exist for first 20 tracks:")
for tid in sample_tracks.index:
    audio_path = get_audio_path(tid)
    exists = audio_path.exists()
    print(f'Track {tid:6d}: {exists:5} - {audio_path}')

# Count total
print(f"\nTotal tracks with genre in metadata: {len(tracks[tracks[genre_col].notna()])}")
print(f"Total MP3 files in data/raw: {len(list(Path('data/raw').rglob('*.mp3')))}")
