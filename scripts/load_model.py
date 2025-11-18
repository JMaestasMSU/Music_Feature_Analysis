"""Load a serialized sklearn or torch model and print a short summary."""
import argparse
import os


def main(path: str):
    if not os.path.exists(path):
        print("Model not found:", path)
        return
    # naive inspection
    print("Found model file:", path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("path")
    args = p.parse_args()
    main(args.path)
