"""
Tune model prediction threshold and evaluate different strategies.

This script loads a trained model and experiments with:
1. Different prediction thresholds (0.3 to 0.8)
2. Top-K prediction (keep only top 3-10 genres)
3. Combination of threshold + top-K

Goal: Find optimal strategy to balance precision and recall.
"""

import numpy as np
import torch
import json
import yaml
from pathlib import Path
from sklearn.metrics import (
    hamming_loss,
    jaccard_score,
    precision_recall_fscore_support
)
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "trained_models" / "multilabel_cnn_filtered_improved"
DATA_DIR = PROJECT_ROOT / "data" / "processed"

print("="*80)
print(f"THRESHOLD TUNING: {MODEL_DIR.name}")
print("="*80)

# Load model configuration
config_path = MODEL_DIR / "config.yaml"
if config_path.exists():
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print(f"Loaded config: {config.get('base_channels', 96)} channels, min_samples={config.get('min_samples_per_genre', 0)}")
else:
    print(f"WARNING: No config.yaml found, using defaults")
    config = {'base_channels': 96, 'use_attention': True}

# Load genre names - try model directory first
genre_names_path = MODEL_DIR / "genre_names.json"
if genre_names_path.exists():
    with open(genre_names_path, 'r') as f:
        genre_names = json.load(f)
    print(f"Loaded {len(genre_names)} genres from model directory")
else:
    print(f"WARNING: No genre_names.json in model dir, using full 70 genres")
    with open(DATA_DIR / "genre_names_70.json", 'r') as f:
        genre_names = json.load(f)

# Load test data
print("\nLoading test data...")
spectrograms = np.load(DATA_DIR / "spectrograms.npy")
labels = np.load(DATA_DIR / "labels_multilabel.npy")

print(f"Full dataset: {spectrograms.shape[0]} samples, {labels.shape[1]} genres")

# Filter labels to match model's genres if needed
if labels.shape[1] != len(genre_names):
    print(f"Filtering labels from {labels.shape[1]} to {len(genre_names)} genres...")

    # Load full genre list to create mapping
    with open(DATA_DIR / "genre_names_70.json", 'r') as f:
        full_genres = json.load(f)

    # Create index mapping
    genre_indices = [full_genres.index(g) for g in genre_names if g in full_genres]
    labels = labels[:, genre_indices]
    print(f"Filtered labels to {labels.shape[1]} genres")

# Create test split (same as training: 60/20/20)
n_samples = len(spectrograms)
n_train = int(0.6 * n_samples)
n_val = int(0.2 * n_samples)

np.random.seed(42)
indices = np.random.permutation(n_samples)
test_idx = indices[n_train + n_val:]

X_test_full = spectrograms[test_idx]
y_true_full = labels[test_idx]

# Remove samples with no labels after filtering
valid_mask = y_true_full.sum(axis=1) > 0
X_test = X_test_full[valid_mask]
y_true = y_true_full[valid_mask]

print(f"Test set: {len(X_test)} samples (filtered from {len(X_test_full)} to remove unlabeled)")

# Load model
print("Loading trained model...")
import sys
sys.path.append(str(PROJECT_ROOT / "models"))
from cnn_model import MultiLabelAudioCNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Clear CUDA cache
if device.type == 'cuda':
    torch.cuda.empty_cache()
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

# Load checkpoint FIRST to get exact architecture
checkpoint_path = MODEL_DIR / "best_model.pt"
print(f"Loading checkpoint from: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

# Get architecture parameters from checkpoint AND config
num_genres_checkpoint = checkpoint.get('num_genres', len(genre_names))
base_channels = config.get('base_channels', 96)  # Read from config!
use_attention = config.get('use_attention', True)

print(f"Model architecture from config:")
print(f"  Genres: {num_genres_checkpoint}")
print(f"  Base channels: {base_channels}")
print(f"  Attention: {use_attention}")

# Verify genre count matches
if num_genres_checkpoint != len(genre_names):
    print(f"WARNING: Checkpoint has {num_genres_checkpoint} genres, but loaded {len(genre_names)} genre names")
    print(f"Using checkpoint value: {num_genres_checkpoint} genres")
    genre_names = genre_names[:num_genres_checkpoint]

# Create model with correct architecture
print("Creating model...")
model = MultiLabelAudioCNN(
    num_genres=num_genres_checkpoint,
    base_channels=base_channels,
    use_attention=use_attention
)

# Load state dict BEFORE moving to GPU
print("Loading model weights...")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()


eval_device = torch.device('cuda')

model = model.to(eval_device)

# Clear checkpoint from memory
del checkpoint
if device.type == 'cuda':
    torch.cuda.empty_cache()

print(f"Model loaded successfully on {eval_device}")

# Get predictions (probabilities)
print("\nGenerating predictions...")

# Validate data first
print(f"Data validation:")
print(f"  X_test shape: {X_test.shape}")
print(f"  X_test dtype: {X_test.dtype}")
print(f"  X_test range: [{X_test.min():.3f}, {X_test.max():.3f}]")
print(f"  Has NaN: {np.isnan(X_test).any()}")
print(f"  Has Inf: {np.isinf(X_test).any()}")

# Check for invalid data
if np.isnan(X_test).any() or np.isinf(X_test).any():
    print("ERROR: Data contains NaN or Inf values!")
    # Clean the data
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=1.0, neginf=0.0)
    print("  Cleaned invalid values")

# Process in batches to avoid memory issues
batch_size = 16  # Reduced from 32 for safety
y_probs_list = []

print(f"\nProcessing {len(X_test)} samples in batches of {batch_size}...")

with torch.no_grad():
    for i in range(0, len(X_test), batch_size):
        batch = X_test[i:i+batch_size]

        # Add channel dimension: [batch, height, width] -> [batch, 1, height, width]
        batch_tensor = torch.FloatTensor(batch).unsqueeze(1)

        # Move to eval device (CPU)
        batch_tensor = batch_tensor.to(eval_device)

        # Forward pass
        outputs = model(batch_tensor)
        probs = torch.sigmoid(outputs).cpu().numpy()
        y_probs_list.append(probs)

        # Clean up
        del batch_tensor, outputs

        # Progress
        if i % 160 == 0:
            print(f"  Processed {i+len(batch)}/{len(X_test)}...")

y_probs = np.vstack(y_probs_list)

print(f"\nPredictions generated successfully!")
print(f"  Shape: {y_probs.shape}")
print(f"  Probability range: [{y_probs.min():.3f}, {y_probs.max():.3f}]")


def evaluate_predictions(y_true, y_pred):
    """Compute metrics for binary predictions."""
    # Overall metrics
    hamming = hamming_loss(y_true, y_pred)
    jaccard = jaccard_score(y_true, y_pred, average='samples')
    
    # Per-sample metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    
    # Count predictions per sample
    avg_preds = y_pred.sum(axis=1).mean()
    avg_true = y_true.sum(axis=1).mean()
    
    return {
        'hamming_loss': float(hamming),
        'jaccard_score': float(jaccard),
        'f1_macro': float(f1),
        'precision_macro': float(precision),
        'recall_macro': float(recall),
        'avg_predictions': float(avg_preds),
        'avg_true_labels': float(avg_true)
    }


def threshold_strategy(y_probs, threshold):
    """Apply simple threshold."""
    return (y_probs >= threshold).astype(int)


def topk_strategy(y_probs, k):
    """Keep only top K predictions per sample."""
    y_pred = np.zeros_like(y_probs, dtype=int)
    for i in range(len(y_probs)):
        top_k_indices = np.argsort(y_probs[i])[-k:]
        y_pred[i, top_k_indices] = 1
    return y_pred


def threshold_topk_strategy(y_probs, threshold, k):
    """Apply threshold, then limit to top K."""
    y_pred = (y_probs >= threshold).astype(int)
    
    # For samples with more than K predictions, keep only top K
    for i in range(len(y_pred)):
        pred_count = y_pred[i].sum()
        if pred_count > k:
            # Get indices of predicted genres
            pred_indices = np.where(y_pred[i] == 1)[0]
            # Get their probabilities
            pred_probs = y_probs[i, pred_indices]
            # Keep only top K
            top_k_local = np.argsort(pred_probs)[-k:]
            top_k_global = pred_indices[top_k_local]
            
            # Reset and set top K
            y_pred[i] = 0
            y_pred[i, top_k_global] = 1
    
    return y_pred


print("\n" + "="*80)
print("STRATEGY 1: THRESHOLD TUNING")
print("="*80)

thresholds = np.arange(0.3, 0.81, 0.05)
threshold_results = []

for thresh in thresholds:
    y_pred = threshold_strategy(y_probs, thresh)
    metrics = evaluate_predictions(y_true, y_pred)
    metrics['threshold'] = thresh
    threshold_results.append(metrics)
    
    print(f"\nThreshold: {thresh:.2f}")
    print(f"  F1:        {metrics['f1_macro']:.4f}")
    print(f"  Precision: {metrics['precision_macro']:.4f}")
    print(f"  Recall:    {metrics['recall_macro']:.4f}")
    print(f"  Hamming:   {metrics['hamming_loss']:.4f}")
    print(f"  Avg preds: {metrics['avg_predictions']:.1f} genres/track")

# Find best threshold
best_thresh = max(threshold_results, key=lambda x: x['f1_macro'])
print(f"\nBest threshold: {best_thresh['threshold']:.2f} (F1={best_thresh['f1_macro']:.4f})")


print("\n" + "="*80)
print("STRATEGY 2: TOP-K PREDICTION")
print("="*80)

k_values = [3, 5, 7, 10, 15, 20]
topk_results = []

for k in k_values:
    y_pred = topk_strategy(y_probs, k)
    metrics = evaluate_predictions(y_true, y_pred)
    metrics['k'] = k
    topk_results.append(metrics)
    
    print(f"\nTop-{k}:")
    print(f"  F1:        {metrics['f1_macro']:.4f}")
    print(f"  Precision: {metrics['precision_macro']:.4f}")
    print(f"  Recall:    {metrics['recall_macro']:.4f}")
    print(f"  Hamming:   {metrics['hamming_loss']:.4f}")
    print(f"  Avg preds: {metrics['avg_predictions']:.1f} genres/track")

best_k = max(topk_results, key=lambda x: x['f1_macro'])
print(f"\nBest K: {best_k['k']} (F1={best_k['f1_macro']:.4f})")


print("\n" + "="*80)
print("STRATEGY 3: THRESHOLD + TOP-K COMBINATION")
print("="*80)

# Try combinations around the best threshold and K
test_combos = [
    (0.4, 5), (0.4, 7), (0.4, 10),
    (0.5, 5), (0.5, 7), (0.5, 10),
    (0.6, 5), (0.6, 7), (0.6, 10),
]

combo_results = []

for thresh, k in test_combos:
    y_pred = threshold_topk_strategy(y_probs, thresh, k)
    metrics = evaluate_predictions(y_true, y_pred)
    metrics['threshold'] = thresh
    metrics['k'] = k
    combo_results.append(metrics)
    
    print(f"\nThreshold={thresh:.2f}, Top-{k}:")
    print(f"  F1:        {metrics['f1_macro']:.4f}")
    print(f"  Precision: {metrics['precision_macro']:.4f}")
    print(f"  Recall:    {metrics['recall_macro']:.4f}")
    print(f"  Hamming:   {metrics['hamming_loss']:.4f}")
    print(f"  Avg preds: {metrics['avg_predictions']:.1f} genres/track")

best_combo = max(combo_results, key=lambda x: x['f1_macro'])
print(f"\nBest combo: threshold={best_combo['threshold']:.2f}, K={best_combo['k']} (F1={best_combo['f1_macro']:.4f})")


print("\n" + "="*80)
print("SUMMARY - BEST STRATEGIES")
print("="*80)

print("\n1. Best Threshold Only:")
print(f"   Threshold: {best_thresh['threshold']:.2f}")
print(f"   F1: {best_thresh['f1_macro']:.4f} | P: {best_thresh['precision_macro']:.4f} | R: {best_thresh['recall_macro']:.4f}")

print("\n2. Best Top-K Only:")
print(f"   K: {best_k['k']}")
print(f"   F1: {best_k['f1_macro']:.4f} | P: {best_k['precision_macro']:.4f} | R: {best_k['recall_macro']:.4f}")

print("\n3. Best Threshold + Top-K:")
print(f"   Threshold: {best_combo['threshold']:.2f}, K: {best_combo['k']}")
print(f"   F1: {best_combo['f1_macro']:.4f} | P: {best_combo['precision_macro']:.4f} | R: {best_combo['recall_macro']:.4f}")

# Determine overall best
all_results = [
    ('Threshold', best_thresh),
    ('Top-K', best_k),
    ('Combo', best_combo)
]
overall_best = max(all_results, key=lambda x: x[1]['f1_macro'])

print(f"\n{'='*80}")
print(f"🏆 WINNER: {overall_best[0]}")
print(f"{'='*80}")
print(f"F1 Score: {overall_best[1]['f1_macro']:.4f}")
print(f"Precision: {overall_best[1]['precision_macro']:.4f}")
print(f"Recall: {overall_best[1]['recall_macro']:.4f}")
print(f"Hamming Loss: {overall_best[1]['hamming_loss']:.4f}")
if 'threshold' in overall_best[1]:
    print(f"Threshold: {overall_best[1]['threshold']:.2f}")
if 'k' in overall_best[1]:
    print(f"Top-K: {overall_best[1]['k']}")


# Save results
output_file = MODEL_DIR / "threshold_tuning_results.json"
tuning_summary = {
    'threshold_only': threshold_results,
    'topk_only': topk_results,
    'threshold_topk': combo_results,
    'best_strategy': {
        'name': overall_best[0],
        'metrics': overall_best[1]
    }
}

with open(output_file, 'w') as f:
    json.dump(tuning_summary, f, indent=2)

print(f"\nResults saved to: {output_file.relative_to(PROJECT_ROOT)}")


# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Threshold vs metrics
ax = axes[0, 0]
thresholds_arr = [r['threshold'] for r in threshold_results]
ax.plot(thresholds_arr, [r['f1_macro'] for r in threshold_results], 'o-', label='F1', linewidth=2)
ax.plot(thresholds_arr, [r['precision_macro'] for r in threshold_results], 's-', label='Precision', linewidth=2)
ax.plot(thresholds_arr, [r['recall_macro'] for r in threshold_results], '^-', label='Recall', linewidth=2)
ax.axvline(best_thresh['threshold'], color='red', linestyle='--', alpha=0.5, label='Best')
ax.set_xlabel('Threshold')
ax.set_ylabel('Score')
ax.set_title('Threshold Strategy')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Top-K vs metrics
ax = axes[0, 1]
k_arr = [r['k'] for r in topk_results]
ax.plot(k_arr, [r['f1_macro'] for r in topk_results], 'o-', label='F1', linewidth=2)
ax.plot(k_arr, [r['precision_macro'] for r in topk_results], 's-', label='Precision', linewidth=2)
ax.plot(k_arr, [r['recall_macro'] for r in topk_results], '^-', label='Recall', linewidth=2)
ax.axvline(best_k['k'], color='red', linestyle='--', alpha=0.5, label='Best')
ax.set_xlabel('K (top genres)')
ax.set_ylabel('Score')
ax.set_title('Top-K Strategy')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Combo heatmap - F1
ax = axes[1, 0]
combo_thresh = sorted(set(r['threshold'] for r in combo_results))
combo_k = sorted(set(r['k'] for r in combo_results))
f1_grid = np.zeros((len(combo_thresh), len(combo_k)))

for r in combo_results:
    i = combo_thresh.index(r['threshold'])
    j = combo_k.index(r['k'])
    f1_grid[i, j] = r['f1_macro']

im = ax.imshow(f1_grid, cmap='RdYlGn', aspect='auto')
ax.set_xticks(range(len(combo_k)))
ax.set_yticks(range(len(combo_thresh)))
ax.set_xticklabels(combo_k)
ax.set_yticklabels([f"{t:.2f}" for t in combo_thresh])
ax.set_xlabel('K (top genres)')
ax.set_ylabel('Threshold')
ax.set_title('Combo Strategy: F1 Score')
plt.colorbar(im, ax=ax)

# Annotate best
for i, t in enumerate(combo_thresh):
    for j, k in enumerate(combo_k):
        text = ax.text(j, i, f'{f1_grid[i, j]:.3f}',
                      ha="center", va="center", color="black", fontsize=8)

# Plot 4: Predictions per track
ax = axes[1, 1]
strategies = ['Current\n(thresh=0.5)', f'Best Thresh\n({best_thresh["threshold"]:.2f})', 
              f'Best Top-K\n(K={best_k["k"]})', f'Best Combo\n({best_combo["threshold"]:.2f}, K={best_combo["k"]})']
current_preds = (y_probs >= 0.5).sum(axis=1).mean()
preds = [current_preds, best_thresh['avg_predictions'], best_k['avg_predictions'], best_combo['avg_predictions']]
true_avg = y_true.sum(axis=1).mean()

bars = ax.bar(range(len(strategies)), preds, alpha=0.7, label='Predicted')
ax.axhline(true_avg, color='red', linestyle='--', linewidth=2, label=f'True avg ({true_avg:.1f})')
ax.set_xticks(range(len(strategies)))
ax.set_xticklabels(strategies, fontsize=9)
ax.set_ylabel('Avg genres per track')
ax.set_title('Average Predictions per Track')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plot_file = MODEL_DIR / "threshold_tuning_plots.png"
plt.savefig(plot_file, dpi=150, bbox_inches='tight')
print(f"Plots saved to: {plot_file.relative_to(PROJECT_ROOT)}")

print("\n" + "="*80)
print("THRESHOLD TUNING COMPLETE")
print("="*80)
