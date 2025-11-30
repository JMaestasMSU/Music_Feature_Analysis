"""
Production-ready genre classification model.
Fully connected neural network for music genre classification from audio features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np


class GenreClassifier(nn.Module):
    """
    Fully connected neural network for music genre classification.
    
    Architecture:
        Input (20 features) → FC(128) → ReLU → Dropout(0.3) →
        FC(64) → ReLU → Dropout(0.3) →
        FC(32) → ReLU →
        FC(8 genres) → Softmax
    
    Args:
        input_dim (int): Number of input features (default: 20)
        hidden_dims (List[int]): Hidden layer dimensions (default: [128, 64, 32])
        num_classes (int): Number of output classes (default: 8)
        dropout (float): Dropout probability (default: 0.3)
    """
    
    def __init__(
        self,
        input_dim: int = 20,
        hidden_dims: Optional[List[int]] = None,
        num_classes: int = 8,
        dropout: float = 0.3
    ):
        super(GenreClassifier, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.dropout_prob = dropout
        
        # Build layers dynamically
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            
            # Add dropout except for last hidden layer
            if i < len(hidden_dims) - 1:
                layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier/Glorot initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input features of shape (batch_size, input_dim)
        
        Returns:
            torch.Tensor: Logits of shape (batch_size, num_classes)
        """
        return self.network(x)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get class probabilities using softmax.
        
        Args:
            x (torch.Tensor): Input features of shape (batch_size, input_dim)
        
        Returns:
            torch.Tensor: Probabilities of shape (batch_size, num_classes)
        """
        logits = self.forward(x)
        return F.softmax(logits, dim=1)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get predicted class labels.
        
        Args:
            x (torch.Tensor): Input features of shape (batch_size, input_dim)
        
        Returns:
            torch.Tensor: Predicted class indices of shape (batch_size,)
        """
        probs = self.predict_proba(x)
        return torch.argmax(probs, dim=1)
    
    def get_config(self) -> Dict:
        """Get model configuration for serialization."""
        return {
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
            'num_classes': self.num_classes,
            'dropout': self.dropout_prob
        }
    
    @classmethod
    def from_config(cls, config: Dict) -> 'GenreClassifier':
        """Create model from configuration dictionary."""
        return cls(**config)


class ModelWrapper:
    """
    Wrapper for production model with preprocessing and postprocessing.
    Handles feature scaling, prediction, and result formatting.
    """
    
    def __init__(
        self,
        model: GenreClassifier,
        scaler: Optional[object] = None,
        genre_names: Optional[List[str]] = None,
        device: str = 'cpu'
    ):
        """
        Initialize model wrapper.
        
        Args:
            model: Trained GenreClassifier model
            scaler: sklearn StandardScaler for feature normalization
            genre_names: List of genre names for label decoding
            device: Device to run inference on ('cpu' or 'cuda')
        """
        self.model = model.to(device)
        self.model.eval()
        self.scaler = scaler
        self.genre_names = genre_names or [
            'Rock', 'Electronic', 'Hip-Hop', 'Classical', 
            'Jazz', 'Folk', 'Pop', 'Experimental'
        ]
        self.device = device
    
    def preprocess(self, features: np.ndarray) -> torch.Tensor:
        """
        Preprocess features before model inference.
        
        Args:
            features: Raw features of shape (batch_size, n_features) or (n_features,)
        
        Returns:
            Preprocessed features as torch tensor
        """
        # Ensure 2D array
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Scale features if scaler is available
        if self.scaler is not None:
            features = self.scaler.transform(features)
        
        # Convert to tensor
        tensor = torch.FloatTensor(features).to(self.device)
        return tensor
    
    def predict(
        self,
        features: np.ndarray,
        return_probs: bool = False,
        top_k: int = 3
    ) -> Dict:
        """
        Predict genre for given features.
        
        Args:
            features: Input features (batch_size, n_features) or (n_features,)
            return_probs: Whether to return all class probabilities
            top_k: Number of top predictions to return
        
        Returns:
            Dictionary with prediction results
        """
        with torch.no_grad():
            # Preprocess
            x = self.preprocess(features)
            
            # Get probabilities
            probs = self.model.predict_proba(x)
            probs_np = probs.cpu().numpy()
            
            # Get top-k predictions
            top_k = min(top_k, len(self.genre_names))
            top_indices = np.argsort(probs_np[0])[::-1][:top_k]
            top_probs = probs_np[0][top_indices]
            
            # Format results
            results = {
                'predicted_genre': self.genre_names[top_indices[0]],
                'predicted_index': int(top_indices[0]),
                'confidence': float(top_probs[0]),
                'top_predictions': [
                    {
                        'genre': self.genre_names[idx],
                        'confidence': float(prob)
                    }
                    for idx, prob in zip(top_indices, top_probs)
                ]
            }
            
            if return_probs:
                results['all_probabilities'] = {
                    genre: float(prob)
                    for genre, prob in zip(self.genre_names, probs_np[0])
                }
            
            return results
    
    def predict_batch(
        self,
        features_list: List[np.ndarray],
        return_probs: bool = False
    ) -> List[Dict]:
        """
        Predict genres for batch of features.
        
        Args:
            features_list: List of feature arrays
            return_probs: Whether to return all class probabilities
        
        Returns:
            List of prediction dictionaries
        """
        # Stack features
        features_batch = np.vstack(features_list)
        
        with torch.no_grad():
            x = self.preprocess(features_batch)
            probs = self.model.predict_proba(x)
            probs_np = probs.cpu().numpy()
            
            results = []
            for i in range(len(features_list)):
                pred_idx = np.argmax(probs_np[i])
                result = {
                    'predicted_genre': self.genre_names[pred_idx],
                    'predicted_index': int(pred_idx),
                    'confidence': float(probs_np[i, pred_idx])
                }
                
                if return_probs:
                    result['all_probabilities'] = {
                        genre: float(prob)
                        for genre, prob in zip(self.genre_names, probs_np[i])
                    }
                
                results.append(result)
            
            return results


def create_model(
    input_dim: int = 20,
    num_classes: int = 8,
    hidden_dims: Optional[List[int]] = None,
    dropout: float = 0.3
) -> GenreClassifier:
    """
    Factory function to create a GenreClassifier model.
    
    Args:
        input_dim: Number of input features
        num_classes: Number of output classes
        hidden_dims: List of hidden layer dimensions
        dropout: Dropout probability
    
    Returns:
        Initialized GenreClassifier model
    """
    return GenreClassifier(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        num_classes=num_classes,
        dropout=dropout
    )


if __name__ == '__main__':
    # Test model creation
    model = create_model()
    print(f"Model created: {model}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 20)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test prediction
    probs = model.predict_proba(x)
    preds = model.predict(x)
    print(f"Probabilities shape: {probs.shape}")
    print(f"Predictions shape: {preds.shape}")
    print(f"Predictions: {preds}")
