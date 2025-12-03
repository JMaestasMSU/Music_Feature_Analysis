#!/usr/bin/env python3
"""Quick script to check improved model training status."""

import torch
from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).parent
MODEL_DIR = PROJECT_ROOT / "models" / "trained_models" / "multilabel_cnn_filtered_improved"

def main():
    print("="*80)
    print("TRAINING STATUS CHECK: multilabel_cnn_filtered_improved")
    print("="*80)

    # Check files
    print("\nFiles present:")
    files_found = list(MODEL_DIR.glob("*"))
    if not files_found:
        print("  Model directory is empty or doesn't exist")
        return

    for file in sorted(files_found):
        size_mb = file.stat().st_size / 1024 / 1024
        print(f"   {file.name} ({size_mb:.1f} MB)")

    # Load checkpoint
    checkpoint_path = MODEL_DIR / "best_model.pt"
    if not checkpoint_path.exists():
        print("\nNo checkpoint found (best_model.pt missing)")
        return

    print("\nLoading checkpoint...")
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    print(f"\nCheckpoint Details:")
    epoch = checkpoint.get('epoch', 'Unknown')
    val_loss = checkpoint.get('val_loss', None)
    val_f1 = checkpoint.get('val_f1', None)

    print(f"  Epoch: {epoch}")
    if val_loss is not None:
        print(f"  Val Loss: {val_loss:.4f}")
    else:
        print(f"  Val Loss: Unknown")

    if val_f1 is not None:
        print(f"  Val F1: {val_f1:.4f}")
    else:
        print(f"  Val F1: Unknown")

    print(f"\n  Available keys: {list(checkpoint.keys())}")

    # Check if training finished
    training_history_path = MODEL_DIR / "training_history.json"
    test_results_path = MODEL_DIR / "test_results.json"

    print("\nTraining Status:")
    if training_history_path.exists():
        print("   Training history saved")
        try:
            with open(training_history_path) as f:
                history = json.load(f)
            total_epochs = len(history.get('train_loss', []))
            print(f"    Total epochs trained: {total_epochs}")

            if 'val_loss' in history and history['val_loss']:
                final_val_loss = history['val_loss'][-1]
                print(f"    Final val loss: {final_val_loss:.4f}")

            if 'val_f1' in history and history['val_f1']:
                best_val_f1 = max(history['val_f1'])
                print(f"    Best val F1: {best_val_f1:.4f}")
        except Exception as e:
            print(f"    Error reading history: {e}")
    else:
        print("  Training history not saved")
        print("    → Training may still be in progress")
        print("    → Or training script didn't complete properly")

    if test_results_path.exists():
        print("\n   Test evaluation completed")
        try:
            with open(test_results_path) as f:
                results = json.load(f)

            test_f1 = results.get('f1_macro', None)
            test_prec = results.get('precision_macro', None)
            test_recall = results.get('recall_macro', None)

            if test_f1 is not None:
                print(f"    Test F1: {test_f1:.4f} ({test_f1*100:.1f}%)")
            if test_prec is not None:
                print(f"    Test Precision: {test_prec:.4f} ({test_prec*100:.1f}%)")
            if test_recall is not None:
                print(f"    Test Recall: {test_recall:.4f} ({test_recall*100:.1f}%)")

            # Compare to original model
            print("\n  Comparison to Original Model (70 genres):")
            print("     Original: F1=13.4%, Precision=7.8%, Recall=77.0%")
            if test_f1 is not None and test_prec is not None:
                f1_improvement = (test_f1 / 0.134) if test_f1 > 0 else 0
                prec_improvement = (test_prec / 0.078) if test_prec > 0 else 0
                print(f"     Improved: F1={test_f1*100:.1f}% ({f1_improvement:.1f}x), Precision={test_prec*100:.1f}% ({prec_improvement:.1f}x)")
        except Exception as e:
            print(f"    Error reading test results: {e}")
    else:
        print("\n  Test evaluation not run yet")
        print("    → Need to run evaluation script on test set")

    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)

    if not training_history_path.exists():
        print("\n1. Check if training is still running:")
        print("   - Look for python process: tasklist | findstr python")
        print("   - Check GPU usage")
        print("\n2. If not running, consider resuming training:")
        print("   - Training may have been interrupted")
        print("   - Can resume from checkpoint if needed")
    elif not test_results_path.exists():
        print("\n1. Training appears complete - run test evaluation:")
        print("   python scripts/evaluate_model.py \\")
        print("     --model-dir models/trained_models/multilabel_cnn_filtered_improved")
    else:
        print("\n Training and evaluation complete!")
        print("   - Update presentation with final results")
        print("   - Generate comparison plots")
        print("   - Create before/after visualization")

    print("\n" + "="*80)

if __name__ == "__main__":
    main()
