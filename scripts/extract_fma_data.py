"""
Extract FMA dataset ZIP files using Python (handles newer ZIP formats).
Works on macOS, Windows, and Linux.
"""

import zipfile
from pathlib import Path
import sys

def extract_zip(zip_path, extract_to):
    """Extract ZIP file using Python's zipfile module."""
    print(f"Extracting {zip_path} to {extract_to}...")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get total number of files
            total_files = len(zip_ref.namelist())
            print(f"Found {total_files} files in archive")
            
            # Extract with progress
            for i, member in enumerate(zip_ref.namelist(), 1):
                if i % 100 == 0:
                    print(f"  Progress: {i}/{total_files} files ({i*100//total_files}%)")
                zip_ref.extract(member, extract_to)
            
            print(f"✓ Extraction complete!")
            return True
            
    except zipfile.BadZipFile:
        print(f"✗ Error: {zip_path} is not a valid ZIP file")
        return False
    except Exception as e:
        print(f"✗ Error extracting: {e}")
        return False

if __name__ == '__main__':
    # Determine paths
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / 'data'
    
    # Extract FMA small dataset
    fma_small_zip = data_dir / 'raw' / 'fma_small.zip'
    fma_small_extract = data_dir / 'raw'
    
    if fma_small_zip.exists():
        print("=" * 60)
        print("Extracting FMA Small Dataset")
        print("=" * 60)
        extract_zip(fma_small_zip, fma_small_extract)
    else:
        print(f"✗ File not found: {fma_small_zip}")
        print("  Please download first:")
        print("  cd data/raw/")
        print("  curl -L -o fma_small.zip https://os.unil.cloud.switch.ch/fma/fma_small.zip")
    
    # Extract metadata
    metadata_zip = data_dir / 'metadata' / 'fma_metadata.zip'
    metadata_extract = data_dir / 'metadata'
    
    if metadata_zip.exists():
        print("\n" + "=" * 60)
        print("Extracting FMA Metadata")
        print("=" * 60)
        extract_zip(metadata_zip, metadata_extract)
    else:
        print(f"✗ File not found: {metadata_zip}")
    
    print("\n" + "=" * 60)
    print("Extraction Complete")
    print("=" * 60)
