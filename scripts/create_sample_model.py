"""Create and serialize a small baseline sklearn model for CI and local testing.

This script trains a tiny RandomForestClassifier on synthetic data and writes
the model to the given output path (default: models/sample_model.joblib).
"""
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
import joblib
import argparse
import os


def main(out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=6,
        n_classes=3,
        random_state=42
        )
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    joblib.dump(model, out_path)
    print(f"Wrote sample model to: {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", dest="out", default="models/sample_model.joblib")
    args = p.parse_args()
    main(args.out)
