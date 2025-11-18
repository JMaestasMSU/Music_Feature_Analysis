"""
Genre classification neural network model.
Includes model definition, wrapper for inference, and utilities.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler


class GenreClassifier(nn.Module):
    """
    Fully connected neural network for music genre classification.
    
    Architecture:
        Input → FC(128) → ReLU → Dropout(0.3) →
        FC(64) → ReLU → Dropout(0.3) →
        FC(32) → ReLU →
        FC(num_classes) → Output
    """
    
    def __init__(self, input_dim: int = 20, num_classes: int = 8, dropout: float = 0.3):
        super(GenreClassifier, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Layers
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, num_classes)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        """Forward pass through network."""
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        
        x = self.relu(self.fc3(x))
        
        x = self.fc4(x)  # No activation (use with CrossEntropyLoss)
        
        return x
    
    def predict_proba(self, x):
        """Get probability predictions."""
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)
        return probs


class ModelWrapper:
    """
    Wrapper for model inference with preprocessing.
    Handles feature scaling, prediction, and post-processing.
    """
    
    def __init__(
        self,
        model: GenreClassifier,
        scaler: Optional[StandardScaler] = None,
        genre_names: Optional[List[str]] = None,
        device: str = 'cpu'
    ):
        self.model = model.to(device)
        self.model.eval()
        self.scaler = scaler
        self.genre_names = genre_names or [f'Genre_{i}' for i in range(model.num_classes)]
        self.device = device
    
    def predict(
        self,
        features: np.ndarray,
        return_probs: bool = False,
        top_k: int = 3
    ) -> Dict:
        """
        Predict genre for given features.
        
        Args:
            features: Input features (n_features,) or (batch_size, n_features)
            return_probs: Whether to return all class probabilities
            top_k: Number of top predictions to return
        
        Returns:
            Dictionary with prediction results
        """
        # Handle single sample
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Scale features
        if self.scaler is not None:
            features = self.scaler.transform(features)
        
        # Convert to tensor
        features_tensor = torch.FloatTensor(features).to(self.device)
        
        # Predict
        with torch.no_grad():
            logits = self.model(features_tensor)
            probs = torch.softmax(logits, dim=1)
        
        probs_np = probs.cpu().numpy()[0]
        
        # Get top predictions
        top_indices = np.argsort(probs_np)[-top_k:][::-1]
        top_predictions = [
            {
                'genre': self.genre_names[idx],
                'confidence': float(probs_np[idx])
            }
            for idx in top_indices
        ]
        
        result = {
            'predicted_genre': self.genre_names[top_indices[0]],
            'confidence': float(probs_np[top_indices[0]]),
            'top_predictions': top_predictions
        }
        
        if return_probs:
            result['all_probabilities'] = {
                genre: float(prob)
                for genre, prob in zip(self.genre_names, probs_np)
            }
        
        return result
    
    def predict_batch(
        self,
        features_list: List[np.ndarray],
        return_probs: bool = False
    ) -> List[Dict]:
        """Predict genres for batch of features."""
        return [
            self.predict(features, return_probs=return_probs)
            for features in features_list
        ]


if __name__ == '__main__':
    # Test model
    print("Testing GenreClassifier...")
    
    model = GenreClassifier(input_dim=20, num_classes=8)
    print(f"Model created: {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass
    test_input = torch.randn(2, 20)
    output = model(test_input)
    print(f"Forward pass: {test_input.shape} → {output.shape}")
    
    # Test wrapper
    wrapper = ModelWrapper(model, genre_names=['Rock', 'Pop', 'Jazz', 'Classical', 'Electronic', 'Hip-Hop', 'Folk', 'Experimental'])
    test_features = np.random.randn(20)
    result = wrapper.predict(test_features, return_probs=True, top_k=3)
    
    print(f"\nPrediction result:")
    print(f"  Predicted genre: {result['predicted_genre']}")
    print(f"  Confidence: {result['confidence']:.4f}")
    print(f"  Top 3: {[p['genre'] for p in result['top_predictions']]}")
