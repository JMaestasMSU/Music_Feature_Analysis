"""
Extract FMA Dataset
Downloads and extracts FMA dataset to proper locations.
Checks for existing data before extracting.
"""

import os
import sys
import zipfile
import urllib.request
from pathlib import Path
import shutil

# Project paths
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DIR = DATA_DIR / 'raw'
METADATA_DIR = DATA_DIR / 'metadata'

# FMA URLs
FMA_SMALL_URL = "https://os.unil.cloud.switch.ch/fma/fma_small.zip"
FMA_METADATA_URL = "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip"

def ensure_directories():
    """Create necessary directories if they don't exist."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Directories ready: {DATA_DIR}")

def check_raw_data_exists():
    """Check if raw audio data already extracted."""
    # Look for audio files in subdirectories (000, 001, etc.)
    if RAW_DIR.exists():
        subdirs = [d for d in RAW_DIR.iterdir() if d.is_dir() and d.name.isdigit()]
        if subdirs:
            # Count audio files
            audio_files = list(RAW_DIR.rglob("*.mp3"))
            if audio_files:
                print(f"✓ Found {len(audio_files)} audio files in {RAW_DIR}")
                return True
    return False

def check_metadata_exists():
    """Check if metadata already extracted."""
    required_files = ['tracks.csv', 'genres.csv', 'features.csv']
    existing = [f for f in required_files if (METADATA_DIR / f).exists()]
    
    if len(existing) == len(required_files):
        print(f"✓ All metadata files found in {METADATA_DIR}")
        return True
    elif existing:
        print(f"⚠️  Partial metadata found: {existing}")
        return False
    return False

def download_file(url, dest_path):
    """Download file with progress."""
    print(f"\nDownloading: {url}")
    print(f"Destination: {dest_path}")
    
    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(downloaded * 100 / total_size, 100)
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            print(f"\rProgress: {percent:5.1f}% ({mb_downloaded:6.1f} / {mb_total:6.1f} MB)", end='')
    
    try:
        urllib.request.urlretrieve(url, dest_path, reporthook=progress_hook)
        print()  # New line after progress
        return True
    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        return False

def extract_zip_flat(zip_path, dest_dir, skip_root=True):
    """
    Extract zip file, optionally skipping the root directory.
    
    Args:
        zip_path: Path to zip file
        dest_dir: Destination directory
        skip_root: If True, extracts contents directly to dest_dir
                   If False, preserves zip structure
    """
    print(f"\nExtracting: {zip_path.name}")
    
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
                        # Multiple roots, don't skip
                        skip_root = False
                        break
            
            if skip_root and root_dir:
                print(f"  Skipping root directory: {root_dir}")
                # Extract without root directory
                for member in members:
                    # Skip the root directory itself
                    if member == root_dir or member == f"{root_dir}/":
                        continue
                    
                    # Remove root from path
                    target_path = Path(member)
                    if target_path.parts[0] == root_dir:
                        target_path = Path(*target_path.parts[1:])
                    
                    # Full destination path
                    dest_path = dest_dir / target_path
                    
                    # Extract file or create directory
                    if member.endswith('/'):
                        dest_path.mkdir(parents=True, exist_ok=True)
                    else:
                        dest_path.parent.mkdir(parents=True, exist_ok=True)
                        with zip_ref.open(member) as source:
                            with open(dest_path, 'wb') as target:
                                shutil.copyfileobj(source, target)
                
                print(f"  ✓ Extracted {len(members)} items to {dest_dir}")
                return True
        
        # Standard extraction (preserves structure)
        zip_ref.extractall(dest_dir)
        print(f"  ✓ Extracted to {dest_dir}")
        return True

def extract_raw_audio():
    """Download and extract raw audio data."""
    print("\n" + "=" * 70)
    print("RAW AUDIO DATA (fma_small.zip)")
    print("=" * 70)
    
    # Check if already exists
    if check_raw_data_exists():
        print("⏩ Skipping download - audio files already present")
        return True
    
    # Download
    zip_path = RAW_DIR / 'fma_small.zip'
    if not zip_path.exists():
        print(f"\n📥 Downloading fma_small.zip (~8 GB)...")
        if not download_file(FMA_SMALL_URL, zip_path):
            return False
    else:
        print(f"✓ Using existing download: {zip_path}")
    
    # Extract directly to raw/ (skip fma_small/ directory)
    print(f"\n📦 Extracting to {RAW_DIR}...")
    success = extract_zip_flat(zip_path, RAW_DIR, skip_root=True)
    
    if success:
        # Clean up zip file
        print(f"\n🧹 Cleaning up {zip_path.name}...")
        zip_path.unlink()
        print(f"✓ Removed {zip_path.name}")
        
        # Verify extraction
        audio_files = list(RAW_DIR.rglob("*.mp3"))
        print(f"\n✓ Extraction complete: {len(audio_files)} audio files")
        return True
    
    return False

def extract_metadata():
    """Download and extract metadata."""
    print("\n" + "=" * 70)
    print("METADATA (fma_metadata.zip)")
    print("=" * 70)
    
    # Check if already exists
    if check_metadata_exists():
        print("⏩ Skipping download - metadata already present")
        return True
    
    # Download
    zip_path = METADATA_DIR / 'fma_metadata.zip'
    if not zip_path.exists():
        print(f"\n📥 Downloading fma_metadata.zip (~5 MB)...")
        if not download_file(FMA_METADATA_URL, zip_path):
            return False
    else:
        print(f"✓ Using existing download: {zip_path}")
    
    # Extract directly to metadata/ (skip fma_metadata/ directory)
    print(f"\n📦 Extracting to {METADATA_DIR}...")
    success = extract_zip_flat(zip_path, METADATA_DIR, skip_root=True)
    
    if success:
        # Clean up zip file
        print(f"\n🧹 Cleaning up {zip_path.name}...")
        zip_path.unlink()
        print(f"✓ Removed {zip_path.name}")
        
        # Verify extraction
        csv_files = list(METADATA_DIR.glob("*.csv"))
        print(f"\n✓ Extraction complete: {len(csv_files)} CSV files")
        return True
    
    return False

def main():
    """Main extraction workflow."""
    print("=" * 70)
    print("FMA DATASET EXTRACTION")
    print("=" * 70)
    print(f"Project: {PROJECT_ROOT}")
    print(f"Data directory: {DATA_DIR}")
    
    # Create directories
    ensure_directories()
    
    # Extract metadata (smaller, faster)
    print("\n[1/2] Processing metadata...")
    metadata_success = extract_metadata()
    
    # Extract raw audio (larger, slower)
    print("\n[2/2] Processing raw audio...")
    audio_success = extract_raw_audio()
    
    # Summary
    print("\n" + "=" * 70)
    print("EXTRACTION SUMMARY")
    print("=" * 70)
    print(f"Metadata: {'✓ SUCCESS' if metadata_success else '✗ FAILED'}")
    print(f"Raw audio: {'✓ SUCCESS' if audio_success else '✗ FAILED'}")
    
    if metadata_success and audio_success:
        print("\n✓ All data extracted successfully!")
        print("\nNext steps:")
        print("  python scripts/process_audio_files.py")
        return 0
    else:
        print("\n✗ Some extractions failed - check errors above")
        return 1

if __name__ == '__main__':
    sys.exit(main())
