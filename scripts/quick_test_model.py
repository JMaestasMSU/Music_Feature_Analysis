"""
Quick model inference and evaluation script.
Tests the current best_model.pt checkpoint on test data.

Usage:
    python scripts/quick_test_model.py --model-dir models/trained_models/multilabel_cnn_filtered_improved
    python scripts/quick_test_model.py --model-dir models/trained_models/multilabel_cnn_filtered_improved --threshold 0.7
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import torch
import json
from sklearn.metrics import (
    classification_report,
    hamming_loss,
    jaccard_score,
    precision_recall_fscore_support
)

# Add models directory to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT / "models"))
from cnn_model import MultiLabelAudioCNN


def parse_args():
    parser = argparse.ArgumentParser(description='Quick model testing')
    parser.add_argument('--model-dir', type=str, required=True,
                       help='Model directory containing best_model.pt')
    parser.add_argument('--data-dir', type=str, default='data/processed',
                       help='Data directory')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Prediction threshold')
    parser.add_argument('--top-k', type=int, default=None,
                       help='Keep only top-K predictions')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for inference')
    return parser.parse_args()


def load_model(model_dir, device):
    """Load model from checkpoint."""
    model_dir = Path(model_dir)
    checkpoint_path = model_dir / "best_model.pt"
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Try to load genre names
    genre_names_candidates = [
        model_dir / "genre_names.json",
        model_dir / "genre_names_filtered.json",
        model_dir.parent.parent.parent / "data" / "processed" / "genre_names_24.json",
    ]
    
    genre_names = None
    for path in genre_names_candidates:
        if path.exists():
            with open(path, 'r') as f:
                genre_names = json.load(f)
            print(f"Loaded genre names from: {path.name}")
            break
    
    # Fallback: reconstruct from training config
    if genre_names is None:
        config_path = model_dir / "config.yaml"
        if config_path.exists():
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Reconstruct genre filtering from training
            min_samples = config.get('min_samples_per_genre', 0)
            if min_samples > 0:
                print(f"Reconstructing genres from config (min_samples={min_samples})...")
                data_dir = Path(config.get('data_dir', 'data/processed'))
                all_genre_names = json.load(open(data_dir / "genre_names_70.json"))
                labels_all = np.load(data_dir / "labels_multilabel.npy")
                counts = labels_all.sum(axis=0).astype(int)
                
                genre_names = [all_genre_names[i] for i in range(len(all_genre_names)) 
                              if counts[i] >= min_samples]
                genre_names.sort(key=lambda g: counts[all_genre_names.index(g)], reverse=True)
                
                if config.get('max_genres'):
                    genre_names = genre_names[:config['max_genres']]
                
                print(f"Reconstructed {len(genre_names)} genre names")
    
    if genre_names is None:
        num_genres = checkpoint['num_genres']
        genre_names = [f"Genre_{i}" for i in range(num_genres)]
        print(f"Warning: Genre names not found, using generic names")
    
    num_genres = len(genre_names)
    
    # Load config to get model architecture
    config_path = model_dir / "config.yaml"
    base_channels = 128  # default
    if config_path.exists():
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        base_channels = config.get('base_channels', 128)
    
    model = MultiLabelAudioCNN(
        num_genres=num_genres,
        base_channels=base_channels,
        use_attention=True
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"\nModel loaded:")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Val F1: {checkpoint['val_f1']:.4f}")
    print(f"  Val Loss: {checkpoint['val_loss']:.4f}")
    print(f"  Genres: {num_genres}")
    print(f"  Base channels: {base_channels}")
    
    return model, genre_names, checkpoint


def load_test_data(data_dir, genre_names):
    """Load test split from processed data."""
    data_dir = Path(data_dir)
    
    # Load full dataset
    spectrograms = np.load(data_dir / "spectrograms.npy")
    labels = np.load(data_dir / "labels_multilabel.npy")
    all_genre_names = json.load(open(data_dir / "genre_names_70.json"))
    
    # Filter to match model's genres (need to do this the same way training did)
    if len(all_genre_names) != len(genre_names):
        print(f"\nFiltering data from {len(all_genre_names)} to {len(genre_names)} genres...")
        
        # Get sample counts for original genres
        genre_counts = labels.sum(axis=0).astype(int)
        
        # Find genres with >= 300 samples (matching training filter)
        viable_genres = [(all_genre_names[i], i, genre_counts[i]) 
                        for i in range(len(all_genre_names)) if genre_counts[i] >= 300]
        viable_genres.sort(key=lambda x: x[2], reverse=True)
        
        # Keep only the genres that made the cut
        genre_indices = [idx for name, idx, count in viable_genres if name in genre_names]
        
        print(f"  Found {len(genre_indices)} matching genres in original data")
        
        # Filter labels to those genres
        labels = labels[:, genre_indices]
        
        # Remove samples with no labels in the filtered set
        has_label = labels.sum(axis=1) > 0
        spectrograms = spectrograms[has_label]
        labels = labels[has_label]
        
        print(f"  Filtered to {len(spectrograms)} samples with valid labels")
    
    # Create test split (same as training: 60/20/20)
    n_samples = len(spectrograms)
    n_train = int(0.6 * n_samples)
    n_val = int(0.2 * n_samples)
    
    np.random.seed(42)
    indices = np.random.permutation(n_samples)
    test_idx = indices[n_train + n_val:]
    
    X_test = spectrograms[test_idx]
    y_test = labels[test_idx]
    
    print(f"\nTest set: {len(X_test)} samples")
    print(f"Labels per sample: {y_test.sum(axis=1).mean():.2f} avg")
    
    return X_test, y_test


def predict_batched(model, X_test, device, batch_size, threshold):
    """Make predictions in batches."""
    model.eval()
    all_probs = []
    
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            batch = X_test[i:i+batch_size]
            batch_tensor = torch.FloatTensor(batch).unsqueeze(1).to(device)
            outputs = model(batch_tensor)
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_probs.append(probs)
    
    y_probs = np.vstack(all_probs)
    y_pred = (y_probs >= threshold).astype(int)
    
    return y_pred, y_probs


def apply_topk(y_pred, y_probs, k):
    """Apply top-K filtering."""
    filtered = np.zeros_like(y_pred)
    
    for i in range(len(y_probs)):
        pred_indices = np.where(y_pred[i] == 1)[0]
        
        if len(pred_indices) <= k:
            filtered[i] = y_pred[i]
        else:
            pred_probs = y_probs[i, pred_indices]
            top_k_local = np.argsort(pred_probs)[-k:]
            top_k_global = pred_indices[top_k_local]
            filtered[i, top_k_global] = 1
    
    return filtered


def main():
    args = parse_args()
    
    print("="*80)
    print("QUICK MODEL TEST")
    print("="*80)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model, genre_names, checkpoint = load_model(args.model_dir, device)
    
    # Load test data
    X_test, y_test = load_test_data(args.data_dir, genre_names)
    
    # Make predictions
    print(f"\nGenerating predictions (threshold={args.threshold})...")
    y_pred, y_probs = predict_batched(model, X_test, device, args.batch_size, args.threshold)
    
    # Apply top-K if specified
    if args.top_k is not None:
        print(f"Applying top-{args.top_k} filtering...")
        y_pred = apply_topk(y_pred, y_probs, args.top_k)
    
    # Calculate metrics
    print("\n" + "="*80)
    print("TEST SET RESULTS")
    print("="*80)
    
    # Overall metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='macro', zero_division=0
    )
    hamming = hamming_loss(y_test, y_pred)
    jaccard = jaccard_score(y_test, y_pred, average='samples')
    
    print(f"\nOverall Metrics:")
    print(f"  F1 Score:       {f1:.4f} ({f1*100:.2f}%)")
    print(f"  Precision:      {precision:.4f} ({precision*100:.2f}%)")
    print(f"  Recall:         {recall:.4f} ({recall*100:.2f}%)")
    print(f"  Hamming Loss:   {hamming:.4f}")
    print(f"  Jaccard Score:  {jaccard:.4f}")
    
    # Predictions per sample
    avg_preds = y_pred.sum(axis=1).mean()
    avg_true = y_test.sum(axis=1).mean()
    print(f"\nPredictions per sample:")
    print(f"  Average predicted: {avg_preds:.2f} genres")
    print(f"  Average actual:    {avg_true:.2f} genres")
    
    # Per-genre performance
    print(f"\n{'Genre':<30s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s} {'Support':>8s}")
    print("-"*80)
    
    precisions, recalls, f1s, supports = precision_recall_fscore_support(
        y_test, y_pred, average=None, zero_division=0
    )
    
    # Sort by F1 descending
    indices = np.argsort(f1s)[::-1]
    
    for idx in indices[:10]:  # Show top 10
        genre = genre_names[idx]
        print(f"{genre:<30s} {precisions[idx]:>10.4f} {recalls[idx]:>10.4f} {f1s[idx]:>10.4f} {int(supports[idx]):>8d}")
    
    print(f"\n... ({len(genre_names)-10} more genres)")
    
    # Show worst performers
    print(f"\nWorst performing genres:")
    for idx in indices[-5:]:
        genre = genre_names[idx]
        print(f"{genre:<30s} {precisions[idx]:>10.4f} {recalls[idx]:>10.4f} {f1s[idx]:>10.4f} {int(supports[idx]):>8d}")
    
    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
