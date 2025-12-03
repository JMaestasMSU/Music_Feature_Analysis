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

    # Model arguments
    parser.add_argument('--num-genres', type=int, default=50,
                       help='Number of genre classes')
    parser.add_argument('--base-channels', type=int, default=64,
                       help='Base number of CNN channels')
    parser.add_argument('--use-attention', action='store_true', default=True,
                       help='Use channel attention')
    parser.add_argument('--multi-label', action='store_true', default=True,
                       help='Multi-label classification (songs can have multiple genres)')

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
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')

    # Output arguments
    parser.add_argument('--output-dir', type=str, default='models/trained_models',
                       help='Directory to save trained models')
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='Experiment name (auto-generated if not provided)')

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
        genre_names = [f"Genre_{i}" for i in range(args.num_genres)]

    # Validate data
    assert len(spectrograms) == len(labels), "Mismatch between spectrograms and labels"

    if args.multi_label:
        assert labels.ndim == 2, "Multi-label requires 2D labels (samples, classes)"
        assert labels.shape[1] == args.num_genres, f"Label shape {labels.shape[1]} != num_genres {args.num_genres}"
    else:
        assert labels.ndim == 1, "Single-label requires 1D labels"

    return spectrograms, labels, genre_names


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


def calculate_pos_weight(labels, train_idx):
    """
    Calculate positive class weights for imbalanced multi-label datasets.

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

    print("\nClass imbalance statistics:")
    print(f"  Most imbalanced: {pos_weight.max():.2f}x")
    print(f"  Least imbalanced: {pos_weight.min():.2f}x")
    print(f"  Mean: {pos_weight.mean():.2f}x")

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

    # Save configuration
    config_path = experiment_dir / "config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(vars(args), f, default_flow_style=False)
    print(f"Config saved to: {config_path}")

    # Get device
    device = get_device(args.device)

    # Load data
    spectrograms, labels, genre_names = load_data(args)

    # Create splits
    train_idx, val_idx, test_idx = create_splits(spectrograms, labels, args)

    # Calculate class weights for imbalanced data
    pos_weight = None
    if args.multi_label:
        pos_weight = calculate_pos_weight(labels, train_idx)

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

    predictions, probabilities = trainer.predict(test_loader)

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
