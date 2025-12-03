#!/usr/bin/env python3
"""
FMA Dataset Setup Script

Downloads, extracts, and organizes the FMA Medium dataset for local training.
Ensures proper directory structure and validates data integrity.

Usage:
    python scripts/setup_fma_dataset.py [--skip-download]

The FMA Medium dataset is ~25GB and contains 25,000 tracks.
"""

import os
import sys
import argparse
import zipfile
import hashlib
import requests
from pathlib import Path
from tqdm import tqdm
import shutil


# FMA Medium dataset information
FMA_MEDIUM_URL = "https://os.unil.cloud.switch.ch/fma/fma_medium.zip"
FMA_METADATA_URL = "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip"

# Expected file sizes (approximate)
FMA_MEDIUM_SIZE = 25 * 1024 * 1024 * 1024  # ~25GB
FMA_METADATA_SIZE = 342 * 1024 * 1024  # ~342MB

# Directory structure (matches existing scripts)
SCRIPT_DIR = Path(__file__).parent.resolve()
BASE_DIR = SCRIPT_DIR.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
METADATA_DIR = DATA_DIR / "metadata"


def create_directory_structure():
    """Create required directory structure."""
    print("\n" + "="*70)
    print("CREATING DIRECTORY STRUCTURE")
    print("="*70)

    directories = [
        DATA_DIR,
        RAW_DIR,
        PROCESSED_DIR,
        METADATA_DIR,
        BASE_DIR / "models" / "trained_models",
        BASE_DIR / "outputs",
        BASE_DIR / "logs"
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f" {directory.relative_to(BASE_DIR)}")

    print("\nDirectory structure created successfully!")


def download_file(url: str, destination: Path, expected_size: int = None):
    """Download file with progress bar."""
    if destination.exists():
        print(f"\n{destination.name} already exists. Skipping download.")
        return True

    print(f"\nDownloading {destination.name}...")
    print(f"Source: {url}")
    print(f"Destination: {destination}")

    try:
        # Stream download with progress bar
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        if expected_size and total_size > 0:
            size_gb = total_size / (1024**3)
            expected_gb = expected_size / (1024**3)
            print(f"Size: {size_gb:.2f} GB (expected ~{expected_gb:.1f} GB)")

        # Create temporary file
        temp_file = destination.with_suffix('.tmp')

        with open(temp_file, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, unit_divisor=1024) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        # Rename to final destination
        temp_file.rename(destination)
        print(f" Download complete: {destination.name}")
        return True

    except Exception as e:
        print(f" Download failed: {e}")
        if temp_file.exists():
            temp_file.unlink()
        return False


def extract_zip(zip_path: Path, extract_to: Path, strip_components: int = 0):
    """
    Extract zip file with progress bar.

    Args:
        zip_path: Path to zip file
        extract_to: Directory to extract to
        strip_components: Number of leading path components to strip (like tar --strip-components)
    """
    print(f"\nExtracting {zip_path.name}...")

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            members = zip_ref.namelist()

            with tqdm(total=len(members), unit='file') as pbar:
                for member in members:
                    # Skip directories
                    if member.endswith('/'):
                        pbar.update(1)
                        continue

                    # Strip leading components if requested
                    if strip_components > 0:
                        parts = Path(member).parts
                        if len(parts) <= strip_components:
                            pbar.update(1)
                            continue
                        new_path = Path(*parts[strip_components:])
                    else:
                        new_path = Path(member)

                    # Create target path
                    target_path = extract_to / new_path
                    target_path.parent.mkdir(parents=True, exist_ok=True)

                    # Extract file
                    with zip_ref.open(member) as source, open(target_path, 'wb') as target:
                        target.write(source.read())

                    pbar.update(1)

        print(f" Extraction complete")
        return True

    except Exception as e:
        print(f" Extraction failed: {e}")
        return False


def validate_fma_structure():
    """Validate FMA dataset structure."""
    print("\n" + "="*70)
    print("VALIDATING FMA DATASET")
    print("="*70)

    # Check if raw directory exists with expected structure (000/, 001/, etc.)
    if not RAW_DIR.exists():
        print(" Raw directory not found")
        return False

    # Count audio files (should be in numbered subdirectories)
    audio_files = list(RAW_DIR.rglob("*.mp3"))

    if len(audio_files) == 0:
        print(" No audio files found")
        return False

    print(f" Found {len(audio_files)} audio files")

    # Check metadata
    metadata_files = list(METADATA_DIR.glob("*.csv"))
    required_metadata = ['tracks.csv', 'genres.csv']
    found_required = [mf.name for mf in metadata_files if mf.name in required_metadata]

    if len(found_required) < len(required_metadata):
        print(f" Missing metadata files: {set(required_metadata) - set(found_required)}")
        print("  You may need to extract fma_metadata.zip")
    else:
        print(f" Found {len(metadata_files)} metadata files")
        for mf in metadata_files:
            print(f"  - {mf.name}")

    # Check directory structure (should be 000/, 001/, 002/, etc.)
    subdirs = [d for d in RAW_DIR.iterdir() if d.is_dir() and d.name.isdigit()]
    print(f" Found {len(subdirs)} numbered audio subdirectories")

    return True


def setup_fma_dataset(skip_download: bool = False):
    """Main setup function."""
    print("\n" + "="*70)
    print("FMA MEDIUM DATASET SETUP")
    print("="*70)
    print("\nThis script will:")
    print("  1. Create directory structure")
    print("  2. Download FMA Medium dataset (~25GB)")
    print("  3. Download FMA metadata (~342MB)")
    print("  4. Extract and organize files")
    print("  5. Validate dataset structure")
    print("\n" + "="*70)

    # Create directories
    create_directory_structure()

    if not skip_download:
        # Download FMA Medium dataset
        fma_zip = RAW_DIR / "fma_medium.zip"

        print("\n" + "="*70)
        print("DOWNLOADING FMA MEDIUM DATASET")
        print("="*70)
        print("\n WARNING: This is a large download (~25GB)")
        print("Make sure you have:")
        print("  - Stable internet connection")
        print("  - At least 50GB free disk space")
        print("  - Sufficient time (may take 1-2 hours)")

        response = input("\nContinue with download? (yes/no): ")

        if response.lower() not in ['yes', 'y']:
            print("\nDownload cancelled. Run with --skip-download if you already have the files.")
            return False

        if not download_file(FMA_MEDIUM_URL, fma_zip, FMA_MEDIUM_SIZE):
            print("\n Failed to download FMA Medium dataset")
            return False

        # Extract FMA Medium (strip the "fma_medium/" prefix to flatten structure)
        print("\n" + "="*70)
        print("EXTRACTING FMA MEDIUM")
        print("="*70)
        print("Note: Extracting directly to data/raw/ to match expected structure")

        if not extract_zip(fma_zip, RAW_DIR, strip_components=1):
            print("\n Failed to extract FMA Medium dataset")
            return False

        # Download FMA Metadata
        print("\n" + "="*70)
        print("DOWNLOADING FMA METADATA")
        print("="*70)

        metadata_zip = RAW_DIR / "fma_metadata.zip"

        if not download_file(FMA_METADATA_URL, metadata_zip, FMA_METADATA_SIZE):
            print("\n Failed to download FMA metadata")
            print("  You can manually download from:")
            print(f"  {FMA_METADATA_URL}")
        else:
            # Extract metadata
            if not extract_zip(metadata_zip, METADATA_DIR):
                print("\n Failed to extract metadata")

    # Validate
    if not validate_fma_structure():
        print("\n Dataset validation failed")
        return False

    # Create .gitignore in data directory
    gitignore_path = DATA_DIR / ".gitignore"
    with open(gitignore_path, 'w') as f:
        f.write("# Ignore large dataset files\n")
        f.write("raw/fma_medium/\n")
        f.write("raw/*.zip\n")
        f.write("processed/*.pkl\n")
        f.write("processed/*.npy\n")

    print("\n" + "="*70)
    print("SETUP COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Run feature extraction:")
    print("     python scripts/extract_audio_features.py")
    print("\n  2. Train the model:")
    print("     python scripts/train_multilabel_cnn.py")
    print("\n  3. Start the API:")
    print("     python backend/app.py")
    print("\n" + "="*70)

    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Setup FMA Medium dataset for music genre classification"
    )
    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Skip downloading files (use if already downloaded)'
    )

    args = parser.parse_args()

    try:
        success = setup_fma_dataset(skip_download=args.skip_download)
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nSetup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n Setup failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
