"""Small command-line utilities for the project."""
from preprocessing.feature_extraction import extract_features


def print_features(path: str) -> None:
    feats = extract_features(path)
    print("Extracted features keys:", list(feats.keys()))


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python cli.py path/to/audio")
    else:
        print_features(sys.argv[1])
