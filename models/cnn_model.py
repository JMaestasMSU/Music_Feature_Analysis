import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

class AudioCNN(nn.Module):
    """
    Convolutional Neural Network for spectrogram-based genre classification.
    Processes raw spectrograms (time-frequency representations) end-to-end.
    """
    
    def __init__(self, num_genres=8, input_channels=1):
        super(AudioCNN, self).__init__()
        
        # Convolutional blocks - extract hierarchical features from spectrograms
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(4, 4), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(4, 4), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Fully connected layers - classification head
        self.fc = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_genres)
        )
    
    def forward(self, x):
        """Forward pass through CNN."""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class CNNTrainer:
    """Training pipeline for CNN model with validation and checkpointing."""
    
    def __init__(self, model, device='cpu', learning_rate=0.001, weight_decay=1e-5):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        self.history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (X, y) in enumerate(train_loader):
            X, y = X.to(self.device), y.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(X)
            loss = self.criterion(outputs, y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)
        
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = correct / total
        return epoch_loss, epoch_acc
    
    def validate(self, val_loader):
        """Validate on validation set."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(self.device), y.to(self.device)
                outputs = self.model(X)
                loss = self.criterion(outputs, y)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == y).sum().item()
                total += y.size(0)
        
        epoch_loss = total_loss / len(val_loader)
        epoch_acc = correct / total
        return epoch_loss, epoch_acc
    
    def train(self, train_loader, val_loader, epochs=50, patience=10):
        """
        Full training loop with early stopping.
        
        Args:
            train_loader: PyTorch DataLoader for training data
            val_loader: PyTorch DataLoader for validation data
            epochs: Maximum number of epochs
            patience: Early stopping patience
        """
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            self.scheduler.step(val_loss)
            
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_cnn_model.pt')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    self.model.load_state_dict(torch.load('best_cnn_model.pt'))
                    break
        
        return self.history
    
    def plot_history(self):
        """Plot training history."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].plot(self.history['train_loss'], label='Train Loss', linewidth=2)
        axes[0].plot(self.history['val_loss'], label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(self.history['val_acc'], label='Val Accuracy', linewidth=2, color='green')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('outputs/cnn_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def evaluate(self, test_loader, genre_names):
        """
        Comprehensive evaluation on test set.
        
        Args:
            test_loader: PyTorch DataLoader for test data
            genre_names: List of genre names for reporting
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for X, y in test_loader:
                X = X.to(self.device)
                outputs = self.model(X)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(y.numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Classification report
        print("\n" + "="*70)
        print("CNN MODEL EVALUATION")
        print("="*70)
        print(classification_report(all_labels, all_preds, target_names=genre_names))
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=genre_names, yticklabels=genre_names)
        plt.title('CNN Model: Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('outputs/cnn_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return all_preds, all_labels


def create_spectrogram_dataset(spectrograms, labels, train_idx, val_idx, test_idx):
    """
    Create PyTorch datasets from spectrograms.
    
    Args:
        spectrograms: Array of shape (n_samples, height, width)
        labels: Array of genre labels
        train_idx, val_idx, test_idx: Index arrays for splits
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Add channel dimension
    specs = spectrograms[:, np.newaxis, :, :]
    specs_tensor = torch.FloatTensor(specs)
    labels_tensor = torch.LongTensor(labels)
    
    # Create datasets
    train_dataset = TensorDataset(specs_tensor[train_idx], labels_tensor[train_idx])
    val_dataset = TensorDataset(specs_tensor[val_idx], labels_tensor[val_idx])
    test_dataset = TensorDataset(specs_tensor[test_idx], labels_tensor[test_idx])
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader
