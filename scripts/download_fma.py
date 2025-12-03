#!/usr/bin/env python3
"""
FMA Dataset Setup Script

Downloads, extracts, and organizes the FMA dataset for local training.
Supports multiple dataset sizes: small, medium, large, and full.
Ensures proper directory structure and validates data integrity.

Usage:
    python scripts/download_fma.py                    # Downloads small by default
    python scripts/download_fma.py --size small       # 8GB, 8,000 tracks, 8 genres
    python scripts/download_fma.py --size medium      # 25GB, 25,000 tracks, 16 genres
    python scripts/download_fma.py --size large       # 93GB, 106,574 tracks, 161 genres
    python scripts/download_fma.py --size full        # 879GB, all tracks (rarely needed)
    python scripts/download_fma.py --skip-download    # Skip download if already present
"""

import os
import sys
import argparse
import zipfile
import urllib.request
from pathlib import Path
import shutil

# Dataset configurations
DATASETS = {
    'small': {
        'url': 'https://os.unil.cloud.switch.ch/fma/fma_small.zip',
        'size_gb': 8,
        'tracks': 8000,
        'genres': '8 balanced genres',
        'description': 'Small dataset for quick testing and development'
    },
    'medium': {
        'url': 'https://os.unil.cloud.switch.ch/fma/fma_medium.zip',
        'size_gb': 25,
        'tracks': 25000,
        'genres': '16 unbalanced genres',
        'description': 'Medium dataset for standard training'
    },
    'large': {
        'url': 'https://os.unil.cloud.switch.ch/fma/fma_large.zip',
        'size_gb': 93,
        'tracks': 106574,
        'genres': '161 genres',
        'description': 'Large dataset for production-grade models'
    },
    'full': {
        'url': 'https://os.unil.cloud.switch.ch/fma/fma_full.zip',
        'size_gb': 879,
        'tracks': 106574,
        'genres': '161 genres with full audio',
        'description': 'Full dataset with complete audio (30s clips in other sets)'
    }
}

FMA_METADATA_URL = "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip"
FMA_METADATA_SIZE = 342 * 1024 * 1024  # ~342MB

# Project paths - matches train_multilabel_cnn.py structure
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DIR = DATA_DIR / 'raw'
PROCESSED_DIR = DATA_DIR / 'processed'
METADATA_DIR = DATA_DIR / 'metadata'


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
        PROJECT_ROOT / "models" / "trained_models",
        PROJECT_ROOT / "outputs",
        PROJECT_ROOT / "logs"
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"  {directory.relative_to(PROJECT_ROOT)}")

    print("\nDirectory structure created successfully!")


def check_raw_data_exists():
    """Check if raw audio data already extracted."""
    if RAW_DIR.exists():
        subdirs = [d for d in RAW_DIR.iterdir() if d.is_dir() and d.name.isdigit()]
        if subdirs:
            audio_files = list(RAW_DIR.rglob("*.mp3"))
            if audio_files:
                print(f"  Found {len(audio_files)} audio files in {len(subdirs)} directories")
                return True, len(audio_files)
    return False, 0


def check_metadata_exists():
    """Check if metadata already extracted."""
    required_files = ['tracks.csv', 'genres.csv']
    existing = [f for f in required_files if (METADATA_DIR / f).exists()]

    if len(existing) == len(required_files):
        print(f"  All metadata files found in {METADATA_DIR}")
        return True
    elif existing:
        print(f"  Partial metadata found: {existing}")
        return False
    return False


def download_file(url: str, destination: Path, expected_size: int = None):
    """Download file with progress bar."""
    if destination.exists():
        print(f"\n  {destination.name} already exists. Skipping download.")
        return True

    print(f"\n  Downloading {destination.name}...")
    print(f"  Source: {url}")
    print(f"  Destination: {destination}")

    try:
        def progress_hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(downloaded * 100 / total_size, 100)
                mb_downloaded = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                print(f"\r  Progress: {percent:5.1f}% ({mb_downloaded:6.1f} / {mb_total:6.1f} MB)", end='')

        urllib.request.urlretrieve(url, destination, reporthook=progress_hook)
        print()  # New line after progress
        return True

    except Exception as e:
        print(f"\n  Download failed: {e}")
        if destination.exists():
            destination.unlink()
        return False


def extract_zip_flat(zip_path: Path, dest_dir: Path, skip_root: bool = True):
    """
    Extract zip file, optionally skipping the root directory.

    Args:
        zip_path: Path to zip file
        dest_dir: Destination directory
        skip_root: If True, extracts contents directly to dest_dir
    """
    print(f"\n  Extracting {zip_path.name}...")

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            members = zip_ref.namelist()

            if skip_root:
                # Find the common root directory
                root_dir = None
                for member in members:
                    parts = Path(member).parts
                    if len(parts) > 0:
                        if root_dir is None:
                            root_dir = parts[0]
                        elif root_dir != parts[0]:
                            skip_root = False
                            break

                if skip_root and root_dir:
                    print(f"  Skipping root directory: {root_dir}")
                    extracted_count = 0
                    for member in members:
                        if member == root_dir or member == f"{root_dir}/":
                            continue

                        target_path = Path(member)
                        if target_path.parts[0] == root_dir:
                            target_path = Path(*target_path.parts[1:])

                        dest_path = dest_dir / target_path

                        if member.endswith('/'):
                            dest_path.mkdir(parents=True, exist_ok=True)
                        else:
                            dest_path.parent.mkdir(parents=True, exist_ok=True)
                            with zip_ref.open(member) as source:
                                with open(dest_path, 'wb') as target:
                                    shutil.copyfileobj(source, target)
                            extracted_count += 1

                            if extracted_count % 1000 == 0:
                                print(f"\r  Extracted {extracted_count} files...", end='')

                    print(f"\r  Extracted {extracted_count} files")
                    return True

            # Standard extraction (preserves structure)
            zip_ref.extractall(dest_dir)
            print(f"  Extracted to {dest_dir}")
            return True

    except Exception as e:
        print(f"\n  Extraction failed: {e}")
        return False


def setup_fma_dataset(dataset_size: str = 'small', skip_download: bool = False):
    """Main setup function."""
    dataset_info = DATASETS[dataset_size]

    print("\n" + "="*70)
    print(f"FMA {dataset_size.upper()} DATASET SETUP")
    print("="*70)
    print(f"\nDataset: {dataset_size}")
    print(f"  Tracks: {dataset_info['tracks']:,}")
    print(f"  Genres: {dataset_info['genres']}")
    print(f"  Size: ~{dataset_info['size_gb']}GB")
    print(f"  Description: {dataset_info['description']}")
    print("\nThis script will:")
    print("  1. Create directory structure")
    print(f"  2. Download FMA {dataset_size} dataset (~{dataset_info['size_gb']}GB)")
    print("  3. Download FMA metadata (~342MB)")
    print("  4. Extract and organize files")
    print("  5. Validate dataset structure")
    print("\n" + "="*70)

    # Create directories
    create_directory_structure()

    # Check if data already exists
    data_exists, file_count = check_raw_data_exists()
    if data_exists and not skip_download:
        print(f"\n  WARNING: Found {file_count} existing audio files in {RAW_DIR}")
        response = input("  Do you want to skip download and use existing files? (yes/no): ")
        if response.lower() in ['yes', 'y']:
            skip_download = True

    if not skip_download:
        # Download FMA dataset
        fma_zip = RAW_DIR / f"fma_{dataset_size}.zip"

        print("\n" + "="*70)
        print(f"DOWNLOADING FMA {dataset_size.upper()} DATASET")
        print("="*70)
        print(f"\n  WARNING: This is a {'VERY ' if dataset_size in ['large', 'full'] else ''}large download (~{dataset_info['size_gb']}GB)")
        print("  Make sure you have:")
        print("    - Stable internet connection")
        print(f"    - At least {dataset_info['size_gb'] * 2 + 10}GB free disk space")

        if dataset_size == 'small':
            print("    - Sufficient time (may take 30-60 minutes)")
        elif dataset_size == 'medium':
            print("    - Sufficient time (may take 1-2 hours)")
        elif dataset_size == 'large':
            print("    - Sufficient time (may take 3-6 hours)")
        else:  # full
            print("    - Sufficient time (may take 12-24+ hours)")

        response = input("\n  Continue with download? (yes/no): ")

        if response.lower() not in ['yes', 'y']:
            print("\n  Download cancelled. Run with --skip-download if you already have the files.")
            return False

        if not download_file(dataset_info['url'], fma_zip):
            print(f"\n  Failed to download FMA {dataset_size} dataset")
            return False

        # Extract FMA dataset
        print("\n" + "="*70)
        print(f"EXTRACTING FMA {dataset_size.upper()}")
        print("="*70)
        print(f"  Note: Extracting directly to data/raw/ to match expected structure")

        if not extract_zip_flat(fma_zip, RAW_DIR, skip_root=True):
            print(f"\n  Failed to extract FMA {dataset_size} dataset")
            return False

        # Clean up zip file
        print(f"\n  Cleaning up {fma_zip.name}...")
        fma_zip.unlink()
        print(f"  Removed {fma_zip.name}")

        # Download FMA Metadata (if not already present)
        if not check_metadata_exists():
            print("\n" + "="*70)
            print("DOWNLOADING FMA METADATA")
            print("="*70)

            metadata_zip = RAW_DIR / "fma_metadata.zip"

            if not download_file(FMA_METADATA_URL, metadata_zip):
                print("\n  Failed to download FMA metadata")
                print("  You can manually download from:")
                print(f"  {FMA_METADATA_URL}")
            else:
                # Extract metadata
                if not extract_zip_flat(metadata_zip, METADATA_DIR, skip_root=True):
                    print("\n  Failed to extract metadata")
                else:
                    # Clean up zip file
                    print(f"\n  Cleaning up {metadata_zip.name}...")
                    metadata_zip.unlink()
                    print(f"  Removed {metadata_zip.name}")

    # Validate
    print("\n" + "="*70)
    print("VALIDATING DATASET")
    print("="*70)

    data_exists, file_count = check_raw_data_exists()
    if not data_exists:
        print("\n  Dataset validation failed: No audio files found")
        return False

    print(f"  Audio files: {file_count:,}")

    if not check_metadata_exists():
        print("\n  Warning: Metadata files not found or incomplete")
        print("  You may need to manually extract fma_metadata.zip")
    else:
        print("  Metadata: OK")

    # Create .gitignore in data directory
    gitignore_path = DATA_DIR / ".gitignore"
    if not gitignore_path.exists():
        with open(gitignore_path, 'w') as f:
            f.write("# Ignore large dataset files\n")
            f.write("raw/*.mp3\n")
            f.write("raw/*.zip\n")
            f.write("processed/*.pkl\n")
            f.write("processed/*.npy\n")

    print("\n" + "="*70)
    print("SETUP COMPLETE!")
    print("="*70)
    print(f"\nDataset: FMA {dataset_size} ({file_count:,} tracks)")
    print("\nNext steps:")
    print("  1. Prepare spectrograms for CNN:")
    print("     python scripts/prepare_cnn_spectrograms.py")
    print("\n  2. Train the model:")
    print("     python scripts/train_multilabel_cnn.py")
    print("\n  3. Start the API:")
    print("     python backend/app.py")
    print("\n" + "="*70)

    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Setup FMA dataset for music genre classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Dataset sizes:
  small   - 8GB, 8,000 tracks, 8 balanced genres (recommended for testing)
  medium  - 25GB, 25,000 tracks, 16 unbalanced genres
  large   - 93GB, 106,574 tracks, 161 genres (recommended for production)
  full    - 879GB, 106,574 tracks, 161 genres with full audio

Examples:
  python scripts/download_fma.py                    # Downloads small (default)
  python scripts/download_fma.py --size medium      # Downloads medium
  python scripts/download_fma.py --size large       # Downloads large
  python scripts/download_fma.py --skip-download    # Use existing files
        """
    )
    parser.add_argument(
        '--size',
        type=str,
        default='small',
        choices=['small', 'medium', 'large', 'full'],
        help='Dataset size to download (default: small)'
    )
    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Skip downloading files (use if already downloaded)'
    )

    args = parser.parse_args()

    # Show available datasets
    if not args.skip_download:
        print("\n" + "="*70)
        print("AVAILABLE FMA DATASETS")
        print("="*70)
        for size, info in DATASETS.items():
            marker = " (selected)" if size == args.size else ""
            print(f"\n  {size.upper()}{marker}")
            print(f"    Tracks: {info['tracks']:,}")
            print(f"    Genres: {info['genres']}")
            print(f"    Size: ~{info['size_gb']}GB")
            print(f"    {info['description']}")

    try:
        success = setup_fma_dataset(dataset_size=args.size, skip_download=args.skip_download)
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nSetup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n  Setup failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
