#!/usr/bin/env python3
"""
Multi-label CNN Training Script

Flexible training pipeline that works with any genre dataset.
Supports unlimited genres, multi-label classification, and custom architectures.

Usage:
    python train_multilabel_cnn.py --data-dir /path/to/data --num-genres 50 --epochs 100
    python train_multilabel_cnn.py --config configs/resnet_50genres.yaml
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import json
import yaml
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

from models.cnn_model import MultiLabelAudioCNN, MultiLabelTrainer
from models.audio_augmentation import create_dataloaders


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train multi-label CNN for genre classification')

    # Data arguments
    parser.add_argument('--data-dir', type=str, default='data/processed',
                       help='Directory containing processed spectrograms')
    parser.add_argument('--spectrogram-file', type=str, default='spectrograms.npy',
                       help='Numpy file with spectrograms')
    parser.add_argument('--labels-file', type=str, default='labels.npy',
                       help='Numpy file with labels')
    parser.add_argument('--genre-names-file', type=str, default='genre_names.json',
                       help='JSON file with genre names')
    parser.add_argument('--min-samples-per-genre', type=int, default=0,
                       help='Filter out genres with fewer than N samples (0 = no filter)')
    parser.add_argument('--max-genres', type=int, default=None,
                       help='Keep only top N most common genres (None = keep all)')

    # Model arguments
    parser.add_argument('--num-genres', type=int, default=50,
                       help='Number of genre classes')
    parser.add_argument('--base-channels', type=int, default=64,
                       help='Base number of CNN channels')
    parser.add_argument('--use-attention', type=lambda x: str(x).lower() == 'true', default=True,
                       help='Use channel attention (true/false)')
    parser.add_argument('--multi-label', type=lambda x: str(x).lower() == 'true', default=True,
                       help='Multi-label classification (true/false)')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                       help='Maximum number of epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                       help='Weight decay (L2 regularization)')
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience')

    # Data split arguments
    parser.add_argument('--train-split', type=float, default=0.6,
                       help='Training set proportion')
    parser.add_argument('--val-split', type=float, default=0.2,
                       help='Validation set proportion')
    parser.add_argument('--test-split', type=float, default=0.2,
                       help='Test set proportion')

    # Hardware arguments
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda, mps)')
    parser.add_argument('--num-workers', type=int, default=0,
                       help='Number of data loading workers (use 0 on Windows)')

    # Output arguments
    parser.add_argument('--output-dir', type=str, default='models/trained_models',
                       help='Directory to save trained models')
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='Experiment name (auto-generated if not provided)')

    # Threshold tuning arguments
    parser.add_argument('--load-tuning-results', type=str, default=None,
                       help='Path to threshold_tuning_results.json to load optimal settings')
    parser.add_argument('--prediction-threshold', type=float, default=0.5,
                       help='Prediction threshold for binary classification')
    parser.add_argument('--top-k', type=int, default=None,
                       help='Keep only top-K predictions per sample (None = no limit)')
    
    # Class weighting arguments
    parser.add_argument('--pos-weight-cap', type=float, default=None,
                       help='Cap maximum pos_weight value to prevent extreme weights')
    parser.add_argument('--pos-weight-power', type=float, default=1.0,
                       help='Apply power to pos_weight (e.g., 0.5 for sqrt scaling)')
    
    # Debugging arguments
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode with extra logging')

    # Config file (overrides all other arguments)
    parser.add_argument('--config', type=str, default=None,
                       help='YAML config file (overrides CLI args)')

    args = parser.parse_args()

    # Load config file if provided
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        for key, value in config.items():
            setattr(args, key, value)

    return args


def get_device(device_str: str) -> torch.device:
    """Get PyTorch device."""
    if device_str == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
            print("Using Apple Silicon GPU (MPS)")
        else:
            device = torch.device('cpu')
            print("Using CPU")
    else:
        device = torch.device(device_str)
        print(f"Using device: {device}")

    return device


def load_tuning_results(tuning_file):
    """Load threshold tuning results and return optimal settings."""
    tuning_path = Path(tuning_file)
    if not tuning_path.exists():
        raise FileNotFoundError(f"Tuning results not found: {tuning_path}")
    
    with open(tuning_path, 'r') as f:
        results = json.load(f)
    
    best_strategy = results.get('best_strategy', {})
    metrics = best_strategy.get('metrics', {})
    
    # Extract threshold and top-k if available
    threshold = metrics.get('threshold', 0.5)
    top_k = metrics.get('k', None)
    
    print(f"\nLoaded optimal tuning settings from {tuning_path.name}:")
    print(f"  Strategy: {best_strategy.get('name', 'Unknown')}")
    print(f"  Threshold: {threshold:.3f}")
    if top_k:
        print(f"  Top-K: {top_k}")
    print(f"  Expected F1: {metrics.get('f1_macro', 0):.4f}")
    print(f"  Expected Precision: {metrics.get('precision_macro', 0):.4f}")
    print(f"  Expected Recall: {metrics.get('recall_macro', 0):.4f}")
    
    return threshold, top_k


def apply_topk(probabilities, predictions, k):
    """
    Apply top-K filtering: keep only the K highest probability predictions per sample.
    
    Args:
        probabilities: Probability scores (n_samples, n_classes)
        predictions: Binary predictions (n_samples, n_classes)
        k: Number of top predictions to keep
    
    Returns:
        filtered_predictions: Binary predictions with top-K filtering applied
    """
    filtered_predictions = np.zeros_like(predictions)
    
    for i in range(len(probabilities)):
        # Get indices of predictions above threshold
        pred_indices = np.where(predictions[i] == 1)[0]
        
        if len(pred_indices) <= k:
            # Keep all if fewer than K predictions
            filtered_predictions[i] = predictions[i]
        else:
            # Keep only top K by probability
            pred_probs = probabilities[i, pred_indices]
            top_k_local = np.argsort(pred_probs)[-k:]
            top_k_global = pred_indices[top_k_local]
            filtered_predictions[i, top_k_global] = 1
    
    return filtered_predictions


def load_data(args):
    """Load spectrograms and labels from disk."""
    data_dir = Path(args.data_dir)

    print(f"\nLoading data from {data_dir}")

    # Load spectrograms
    spec_path = data_dir / args.spectrogram_file
    if not spec_path.exists():
        raise FileNotFoundError(f"Spectrograms not found: {spec_path}")
    spectrograms = np.load(spec_path)
    print(f"  Loaded spectrograms: {spectrograms.shape}")

    # Load labels
    labels_path = data_dir / args.labels_file
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels not found: {labels_path}")
    labels = np.load(labels_path)
    print(f"  Loaded labels: {labels.shape}")

    # Load genre names if available
    genre_names = None
    genre_names_path = data_dir / args.genre_names_file
    if genre_names_path.exists():
        with open(genre_names_path, 'r') as f:
            genre_names = json.load(f)
        print(f"  Loaded {len(genre_names)} genre names")
    else:
        print(f"  Warning: Genre names file not found: {genre_names_path}")
        genre_names = [f"Genre_{i}" for i in range(labels.shape[1])]

    # Filter genres by sample count
    if args.min_samples_per_genre > 0 or args.max_genres is not None:
        spectrograms, labels, genre_names = filter_genres(
            spectrograms, labels, genre_names,
            min_samples=args.min_samples_per_genre,
            max_genres=args.max_genres,
            debug=args.debug
        )
        # Update num_genres after filtering
        args.num_genres = len(genre_names)

    # Validate data
    assert len(spectrograms) == len(labels), "Mismatch between spectrograms and labels"

    if args.multi_label:
        assert labels.ndim == 2, "Multi-label requires 2D labels (samples, classes)"
        assert labels.shape[1] == args.num_genres, f"Label shape {labels.shape[1]} != num_genres {args.num_genres}"
    else:
        assert labels.ndim == 1, "Single-label requires 1D labels"

    return spectrograms, labels, genre_names


def filter_genres(spectrograms, labels, genre_names, min_samples=0, max_genres=None, debug=False):
    """
    Filter dataset to include only genres with sufficient samples.
    
    Args:
        spectrograms: Spectrogram array
        labels: Multi-label array (samples, genres)
        genre_names: List of genre names
        min_samples: Minimum samples per genre
        max_genres: Maximum number of genres to keep (keeps most common)
        debug: Print detailed filtering info
    
    Returns:
        Filtered spectrograms, labels, and genre_names
    """
    print(f"\nFiltering genres:")
    print(f"  Original: {len(genre_names)} genres, {len(spectrograms)} samples")
    
    # Count samples per genre
    genre_counts = labels.sum(axis=0).astype(int)
    
    # Determine which genres to keep
    keep_genres = np.ones(len(genre_names), dtype=bool)
    
    if min_samples > 0:
        keep_genres &= (genre_counts >= min_samples)
        print(f"  After min_samples={min_samples}: {keep_genres.sum()} genres remain")
    
    if max_genres is not None:
        # Sort by count, keep top N
        top_genre_indices = np.argsort(genre_counts)[::-1][:max_genres]
        keep_mask = np.zeros(len(genre_names), dtype=bool)
        keep_mask[top_genre_indices] = True
        keep_genres &= keep_mask
        print(f"  After max_genres={max_genres}: {keep_genres.sum()} genres remain")
    
    # Filter genres
    genre_indices = np.where(keep_genres)[0]
    filtered_labels = labels[:, genre_indices]
    filtered_genre_names = [genre_names[i] for i in genre_indices]
    
    # Remove samples with no remaining labels
    sample_has_label = filtered_labels.sum(axis=1) > 0
    filtered_spectrograms = spectrograms[sample_has_label]
    filtered_labels = filtered_labels[sample_has_label]
    
    print(f"  Final: {len(filtered_genre_names)} genres, {len(filtered_spectrograms)} samples")
    
    if debug:
        print(f"\\n  Kept genres and their counts:")
        for i, idx in enumerate(genre_indices):
            print(f"    {filtered_genre_names[i]:30s}: {genre_counts[idx]:4d} samples")
    
    # Show genre distribution summary
    new_counts = filtered_labels.sum(axis=0)
    print(f"  Imbalance ratio: {new_counts.max()}/{new_counts.min()} = {new_counts.max()/new_counts.min():.1f}x")
    
    return filtered_spectrograms, filtered_labels, filtered_genre_names


def create_splits(spectrograms, labels, args):
    """Create train/val/test splits."""
    n_samples = len(spectrograms)
    indices = np.arange(n_samples)

    # Calculate split sizes
    train_size = args.train_split
    val_size = args.val_split
    test_size = args.test_split

    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, "Splits must sum to 1.0"

    print(f"\nCreating splits: {train_size:.0%} train, {val_size:.0%} val, {test_size:.0%} test")

    # For multi-label, we can't use stratify directly
    # First split: train vs (val + test)
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=(val_size + test_size),
        random_state=42
    )

    # Second split: val vs test
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=(test_size / (val_size + test_size)),
        random_state=42
    )

    print(f"  Train: {len(train_idx)} samples")
    print(f"  Val:   {len(val_idx)} samples")
    print(f"  Test:  {len(test_idx)} samples")

    return train_idx, val_idx, test_idx


def calculate_pos_weight(labels, train_idx, power=1.0, cap=None, debug=False):
    """
    Calculate positive class weights for imbalanced multi-label datasets.

    Args:
        labels: All labels array
        train_idx: Training indices
        power: Apply power transformation (0.5 = sqrt, reduces extreme weights)
        cap: Maximum weight value (None = no cap)
        debug: Print detailed statistics

    Returns tensor of weights for BCEWithLogitsLoss.
    """
    train_labels = labels[train_idx]

    # Count positive samples per class
    pos_counts = train_labels.sum(axis=0)
    neg_counts = len(train_labels) - pos_counts

    # Avoid division by zero
    pos_counts = np.maximum(pos_counts, 1)

    # Weight = neg_count / pos_count
    pos_weight = neg_counts / pos_counts
    
    # Apply power transformation to reduce extreme weights
    if power != 1.0:
        pos_weight = np.power(pos_weight, power)
        print(f"  Applied power transformation: {power}")
    
    # Cap maximum weight
    if cap is not None:
        pos_weight = np.minimum(pos_weight, cap)
        print(f"  Capped weights at: {cap}")

    print("\nClass imbalance statistics:")
    print(f"  Most imbalanced: {pos_weight.max():.2f}x")
    print(f"  Least imbalanced: {pos_weight.min():.2f}x")
    print(f"  Mean: {pos_weight.mean():.2f}x")
    print(f"  Median: {np.median(pos_weight):.2f}x")
    
    if debug:
        print("\n  Top 10 most weighted classes:")
        top_indices = np.argsort(pos_weight)[-10:][::-1]
        for idx in top_indices:
            print(f"    Class {idx}: weight={pos_weight[idx]:.2f}x, pos_count={int(pos_counts[idx])}")

    return torch.FloatTensor(pos_weight)


def main():
    """Main training function."""
    args = parse_args()

    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate experiment name
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"multilabel_cnn_{args.num_genres}genres_{timestamp}"

    experiment_dir = output_dir / args.experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print(f"MULTI-LABEL CNN TRAINING")
    print("="*80)
    print(f"Experiment: {args.experiment_name}")
    print(f"Output dir: {experiment_dir}")

    # Load tuning results if provided
    if args.load_tuning_results:
        threshold, top_k = load_tuning_results(args.load_tuning_results)
        args.prediction_threshold = threshold
        if top_k is not None:
            args.top_k = top_k
    else:
        print(f"\nUsing default prediction settings:")
        print(f"  Threshold: {args.prediction_threshold}")
        if args.top_k:
            print(f"  Top-K: {args.top_k}")

    # Save configuration
    config_path = experiment_dir / "config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(vars(args), f, default_flow_style=False)
    print(f"Config saved to: {config_path}")

    # Get device
    device = get_device(args.device)

    # Load data
    spectrograms, labels, genre_names = load_data(args)
    
    # Save genre names for later inference
    genre_names_path = experiment_dir / "genre_names.json"
    with open(genre_names_path, 'w') as f:
        json.dump(genre_names, f, indent=2)
    print(f"Genre names saved to: {genre_names_path}")

    # Create splits
    train_idx, val_idx, test_idx = create_splits(spectrograms, labels, args)

    # Calculate class weights for imbalanced data
    pos_weight = None
    if args.multi_label:
        pos_weight = calculate_pos_weight(
            labels, train_idx,
            power=args.pos_weight_power,
            cap=args.pos_weight_cap,
            debug=args.debug
        )

    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        spectrograms=spectrograms,
        labels=labels,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        batch_size=args.batch_size,
        multi_label=args.multi_label,
        num_workers=args.num_workers
    )
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    print(f"  Test batches:  {len(test_loader)}")

    # Create model
    print("\nCreating model...")
    model = MultiLabelAudioCNN(
        num_genres=args.num_genres,
        input_channels=1,
        base_channels=args.base_channels,
        use_attention=args.use_attention
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: ~{total_params * 4 / 1024 / 1024:.2f} MB")

    # Create trainer
    print("\nInitializing trainer...")
    trainer = MultiLabelTrainer(
        model=model,
        device=device,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        pos_weight=pos_weight
    )

    # Train model
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80)

    model_path = experiment_dir / "best_model.pt"
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        patience=args.patience,
        save_path=str(model_path)
    )

    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)

    # Save training history
    history_path = experiment_dir / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to: {history_path}")

    # Plot training curves
    plot_path = experiment_dir / "training_curves.png"
    trainer.plot_history(save_path=str(plot_path))
    print(f"Training curves saved to: {plot_path}")

    # Evaluate on test set
    print("\n" + "="*80)
    print("EVALUATING ON TEST SET")
    print("="*80)

    predictions, probabilities = trainer.predict(test_loader, threshold=args.prediction_threshold)

    # Apply top-K if specified
    if args.top_k is not None:
        print(f"Applying top-{args.top_k} filtering...")
        predictions = apply_topk(probabilities, predictions, args.top_k)

    # Calculate test metrics
    test_labels = labels[test_idx]
    from sklearn.metrics import classification_report, hamming_loss, jaccard_score

    print(classification_report(
        test_labels,
        predictions,
        target_names=genre_names,
        zero_division=0
    ))

    # Multi-label specific metrics
    hamming = hamming_loss(test_labels, predictions)
    jaccard = jaccard_score(test_labels, predictions, average='samples')

    print(f"\nMulti-label Metrics:")
    print(f"  Hamming Loss: {hamming:.4f}")
    print(f"  Jaccard Score (Accuracy): {jaccard:.4f}")

    # Save test predictions
    results = {
        'predictions': predictions.tolist(),
        'probabilities': probabilities.tolist(),
        'genre_names': genre_names,
        'metrics': {
            'hamming_loss': float(hamming),
            'jaccard_score': float(jaccard)
        }
    }

    results_path = experiment_dir / "test_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nTest results saved to: {results_path}")

    print("\n" + "="*80)
    print(f"EXPERIMENT COMPLETE: {args.experiment_name}")
    print(f"All outputs saved to: {experiment_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
