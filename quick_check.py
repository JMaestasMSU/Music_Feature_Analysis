import torch
from pathlib import Path

MODEL_DIR = Path("models/trained_models/multilabel_cnn_filtered_improved")
checkpoint_path = MODEL_DIR / "best_model.pt"

if checkpoint_path.exists():
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    print("\nCheckpoint Info:")
    print(f"  Epoch: {checkpoint.get('epoch', 'Unknown')}")
    print(f"  Val F1: {checkpoint.get('val_f1', 'Unknown')}")
    print(f"  Val Loss: {checkpoint.get('val_loss', 'Unknown')}")
    print(f"  Keys: {list(checkpoint.keys())}")

    # Check for training history
    history_path = MODEL_DIR / "training_history.json"
    test_path = MODEL_DIR / "test_results.json"

    print("\nFiles:")
    print(f"  Training history: {history_path.exists()}")
    print(f"  Test results: {test_path.exists()}")
else:
    print("Checkpoint not found")
