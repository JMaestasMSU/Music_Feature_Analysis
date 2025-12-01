"""
ONE SCRIPT TO CREATE ALL FEATURES
Single source of truth for features.pkl

Usage:
    python scripts/process_audio_files.py
    
Output:
    data/processed/features.pkl
"""

import pandas as pd
import pickle
from pathlib import Path
import sys

# Project paths (works from any directory)
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / 'data'
FEATURES_CSV = DATA_DIR / 'metadata' / 'features.csv'
TRACKS_CSV = DATA_DIR / 'metadata' / 'tracks.csv'
OUTPUT_PKL = DATA_DIR / 'processed' / 'features.pkl'  # Changed to features.pkl

def main():
    print("=" * 70)
    print("CREATING features.pkl - Single Source of Truth")
    print("=" * 70)
    
    try:
        # Validate input files exist
        if not FEATURES_CSV.exists():
            raise FileNotFoundError(
                f"Features file not found: {FEATURES_CSV}\n\n"
                f"Download FMA metadata:\n"
                f"  cd {DATA_DIR / 'metadata'}\n"
                f"  curl -L -o fma_metadata.zip https://os.unil.cloud.switch.ch/fma/fma_metadata.zip\n"
                f"  unzip fma_metadata.zip"
            )
        
        if not TRACKS_CSV.exists():
            raise FileNotFoundError(f"Tracks file not found: {TRACKS_CSV}")
        
        # Load FMA metadata
        print(f"\n1. Loading features from: {FEATURES_CSV.relative_to(PROJECT_ROOT)}")
        features_df = pd.read_csv(FEATURES_CSV, index_col=0, header=[0, 1, 2])
        
        print(f"2. Loading tracks from: {TRACKS_CSV.relative_to(PROJECT_ROOT)}")
        tracks_df = pd.read_csv(TRACKS_CSV, index_col=0, header=[0, 1])
        
        # Select key feature groups
        feature_groups = ['mfcc', 'spectral_centroid', 'spectral_rolloff', 
                          'zcr', 'rmse', 'chroma_cens']
        
        print(f"\n3. Selecting features:")
        selected = []
        for group in feature_groups:
            if group in features_df.columns.levels[0]:
                selected.append(features_df[group])
                print(f"   ✓ {group}: {features_df[group].shape[1]} features")
        
        # Combine features
        features_combined = pd.concat(selected, axis=1)
        
        # Add genre labels
        genre_col = ('track', 'genre_top')
        features_combined['genre'] = tracks_df[genre_col]
        
        # Clean
        features_combined = features_combined.dropna()
        
        print(f"\n4. Final dataset:")
        print(f"   Shape: {features_combined.shape}")
        print(f"   Tracks: {len(features_combined):,}")
        print(f"   Features: {len(features_combined.columns) - 1}")
        print(f"   Genres: {features_combined['genre'].nunique()}")
        print(f"\n   Genre distribution:")
        for genre, count in features_combined['genre'].value_counts().head(10).items():
            print(f"     {genre:15s}: {count:4d} tracks")
        
        # Save
        OUTPUT_PKL.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_PKL, 'wb') as f:
            pickle.dump(features_combined, f)
        
        print(f"\n✓ SUCCESS!")
        print(f"  File: {OUTPUT_PKL.relative_to(PROJECT_ROOT)}")
        print(f"  Size: {OUTPUT_PKL.stat().st_size / 1024 / 1024:.2f} MB")
        print("=" * 70)
        print("\nNow you can:")
        print("  • Open notebooks: jupyter notebook notebooks/01_EDA.ipynb")
        print("  • Start backend: python backend/app.py")
        print("  • Run tests: bash tests/run_all_tests.sh")
        
        return 0
    
    except FileNotFoundError as e:
        print(f"\n✗ ERROR: {e}")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    sys.exit(main())
