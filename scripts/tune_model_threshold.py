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
MODEL_DIR = PROJECT_ROOT / "models" / "trained_models" / "multilabel_cnn_70genres_20251203_113316"
DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Load model and data to get predictions
print("Loading model and test data...")

# Load genre names
with open(DATA_DIR / "genre_names_70.json", 'r') as f:
    genre_names = json.load(f)

# Load test data
spectrograms = np.load(DATA_DIR / "spectrograms.npy")
labels = np.load(DATA_DIR / "labels_multilabel.npy")

print(f"Loaded dataset: {spectrograms.shape[0]} samples, {len(genre_names)} genres")

# Create test split (same as training: 60/20/20)
n_samples = len(spectrograms)
n_train = int(0.6 * n_samples)
n_val = int(0.2 * n_samples)

np.random.seed(42)
indices = np.random.permutation(n_samples)
test_idx = indices[n_train + n_val:]

X_test = spectrograms[test_idx]
y_true = labels[test_idx]

print(f"Test set: {len(X_test)} samples")

# Load model
print("Loading trained model...")
import sys
sys.path.append(str(PROJECT_ROOT / "models"))
from cnn_model import MultiLabelAudioCNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MultiLabelAudioCNN(num_genres=len(genre_names), base_channels=96, use_attention=True).to(device)

# Load checkpoint (contains model_state_dict + metadata)
checkpoint = torch.load(MODEL_DIR / "best_model.pt", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"Model loaded on {device}")
print(f"Checkpoint info: epoch {checkpoint['epoch']}, val_F1={checkpoint['val_f1']:.4f}")

# Get predictions (probabilities)
print("Generating predictions...")

# Process in batches to avoid memory issues
batch_size = 32
y_probs_list = []

with torch.no_grad():
    for i in range(0, len(X_test), batch_size):
        batch = X_test[i:i+batch_size]
        # Add channel dimension: [batch, height, width] -> [batch, 1, height, width]
        batch_tensor = torch.FloatTensor(batch).unsqueeze(1).to(device)
        outputs = model(batch_tensor)
        probs = torch.sigmoid(outputs).cpu().numpy()
        y_probs_list.append(probs)

y_probs = np.vstack(y_probs_list)

print(f"Probability range: [{y_probs.min():.3f}, {y_probs.max():.3f}]")


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
print(f"\n✓ Best threshold: {best_thresh['threshold']:.2f} (F1={best_thresh['f1_macro']:.4f})")


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
print(f"\n✓ Best K: {best_k['k']} (F1={best_k['f1_macro']:.4f})")


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
print(f"\n✓ Best combo: threshold={best_combo['threshold']:.2f}, K={best_combo['k']} (F1={best_combo['f1_macro']:.4f})")


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

print(f"\n✓ Results saved to: {output_file.relative_to(PROJECT_ROOT)}")


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
print(f"✓ Plots saved to: {plot_file.relative_to(PROJECT_ROOT)}")

print("\n" + "="*80)
print("THRESHOLD TUNING COMPLETE")
print("="*80)
