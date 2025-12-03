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


# ============================================================================
# ADVANCED ARCHITECTURES - Multi-label, Residual, Attention
# ============================================================================


class ResidualBlock(nn.Module):
    """
    Residual block with skip connections for deeper networks.
    Enables training of 20+ layer networks without vanishing gradients.
    """

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ChannelAttention(nn.Module):
    """
    Channel attention mechanism - learns which frequency bands matter most.
    Inspired by Squeeze-and-Excitation Networks (SENet).
    """

    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()

        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))

        attention = avg_out + max_out
        return x * attention.view(b, c, 1, 1)


class MultiLabelAudioCNN(nn.Module):
    """
    Advanced CNN for multi-label genre classification with residual connections.

    Key Features:
    - Multi-label support: Songs can belong to multiple genres
    - Dynamic genre count: No hardcoded limits
    - Residual blocks: Enables deeper networks (20+ layers)
    - Channel attention: Learns important frequency bands
    - Flexible architecture: Easy to configure depth/width

    Args:
        num_genres: Number of genre classes (default 50 for expanded taxonomy)
        input_channels: Number of input channels (1 for mono spectrograms)
        base_channels: Base number of filters (scales with depth)
        use_attention: Enable channel attention mechanisms
    """

    def __init__(self, num_genres=50, input_channels=1, base_channels=64, use_attention=True):
        super(MultiLabelAudioCNN, self).__init__()

        self.num_genres = num_genres
        self.use_attention = use_attention

        # Initial convolution - process raw spectrogram
        self.conv1 = nn.Conv2d(input_channels, base_channels, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual blocks - hierarchical feature extraction
        self.layer1 = self._make_layer(base_channels, base_channels, 2)
        self.layer2 = self._make_layer(base_channels, base_channels * 2, 2, stride=2)
        self.layer3 = self._make_layer(base_channels * 2, base_channels * 4, 2, stride=2)
        self.layer4 = self._make_layer(base_channels * 4, base_channels * 8, 2, stride=2)

        # Channel attention
        if use_attention:
            self.attention1 = ChannelAttention(base_channels)
            self.attention2 = ChannelAttention(base_channels * 2)
            self.attention3 = ChannelAttention(base_channels * 4)
            self.attention4 = ChannelAttention(base_channels * 8)

        # Global pooling and classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(base_channels * 8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_genres)
        )

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        """Create a layer with multiple residual blocks."""
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride, downsample))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input spectrogram (batch_size, channels, freq_bins, time_steps)

        Returns:
            Logits for multi-label classification (batch_size, num_genres)
        """
        # Initial processing
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Residual blocks with optional attention
        x = self.layer1(x)
        if self.use_attention:
            x = self.attention1(x)

        x = self.layer2(x)
        if self.use_attention:
            x = self.attention2(x)

        x = self.layer3(x)
        if self.use_attention:
            x = self.attention3(x)

        x = self.layer4(x)
        if self.use_attention:
            x = self.attention4(x)

        # Global pooling and classification
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def get_embeddings(self, x):
        """
        Extract feature embeddings (before final classification layer).
        Useful for similarity search, visualization, and transfer learning.

        Returns:
            256-dimensional embedding vector
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        if self.use_attention:
            x = self.attention1(x)

        x = self.layer2(x)
        if self.use_attention:
            x = self.attention2(x)

        x = self.layer3(x)
        if self.use_attention:
            x = self.attention3(x)

        x = self.layer4(x)
        if self.use_attention:
            x = self.attention4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # Return embedding before final layer
        embedding = self.fc[:-1](x)
        return embedding


class MultiLabelTrainer:
    """
    Training pipeline for multi-label CNN with BCEWithLogitsLoss.

    Supports:
    - Multi-label classification (sigmoid outputs)
    - Threshold tuning for optimal precision/recall
    - Per-class metrics reporting
    - Model checkpointing and early stopping
    """

    def __init__(self, model, device='cpu', learning_rate=0.001, weight_decay=1e-5,
                 pos_weight=None):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate,
                                     weight_decay=weight_decay)

        # BCE with logits loss for multi-label (more numerically stable than BCE)
        if pos_weight is not None:
            pos_weight = pos_weight.to(device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_f1': [],
            'val_precision': [],
            'val_recall': []
        }
        self.threshold = 0.5

    def train_epoch(self, train_loader):
        """Train for one epoch with multi-label loss."""
        self.model.train()
        total_loss = 0

        for X, y in train_loader:
            X, y = X.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(X)
            loss = self.criterion(outputs, y.float())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def validate(self, val_loader):
        """Validate with multi-label metrics."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(self.device), y.to(self.device)
                outputs = self.model(X)
                loss = self.criterion(outputs, y.float())

                total_loss += loss.item()

                # Apply sigmoid and threshold
                probs = torch.sigmoid(outputs)
                preds = (probs > self.threshold).float()

                all_preds.append(preds.cpu())
                all_labels.append(y.cpu())

        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()

        # Calculate metrics
        from sklearn.metrics import f1_score, precision_score, recall_score

        f1 = f1_score(all_labels, all_preds, average='samples', zero_division=0)
        precision = precision_score(all_labels, all_preds, average='samples', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='samples', zero_division=0)

        return total_loss / len(val_loader), f1, precision, recall

    def train(self, train_loader, val_loader, epochs=50, patience=10, save_path='best_multilabel_cnn.pt'):
        """Full training loop with early stopping."""
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_f1, val_precision, val_recall = self.validate(val_loader)

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_f1'].append(val_f1)
            self.history['val_precision'].append(val_precision)
            self.history['val_recall'].append(val_recall)

            self.scheduler.step(val_loss)

            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}")
            print(f"  Val F1:     {val_f1:.4f} | Precision: {val_precision:.4f} | Recall: {val_recall:.4f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_f1': val_f1,
                    'num_genres': self.model.num_genres,
                    'threshold': self.threshold
                }, save_path)
                print(f"  → Model saved to {save_path}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    checkpoint = torch.load(save_path)
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    break

        return self.history

    def plot_history(self, save_path='outputs/multilabel_training_history.png'):
        """Plot multi-label training metrics."""
        _, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].plot(self.history['train_loss'], label='Train Loss', linewidth=2)
        axes[0].plot(self.history['val_loss'], label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(self.history['val_f1'], label='F1-Score', linewidth=2)
        axes[1].plot(self.history['val_precision'], label='Precision', linewidth=2)
        axes[1].plot(self.history['val_recall'], label='Recall', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Score')
        axes[1].set_title('Validation Metrics')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def predict(self, dataloader, threshold=None):
        """
        Make predictions on new data.

        Args:
            dataloader: PyTorch DataLoader
            threshold: Classification threshold (default: use self.threshold)

        Returns:
            predictions: Binary predictions (batch_size, num_genres)
            probabilities: Raw probabilities (batch_size, num_genres)
        """
        if threshold is None:
            threshold = self.threshold

        self.model.eval()
        all_preds = []
        all_probs = []

        with torch.no_grad():
            for X, _ in dataloader:
                X = X.to(self.device)
                outputs = self.model(X)
                probs = torch.sigmoid(outputs)
                preds = (probs > threshold).float()

                all_preds.append(preds.cpu())
                all_probs.append(probs.cpu())

        predictions = torch.cat(all_preds).numpy()
        probabilities = torch.cat(all_probs).numpy()

        return predictions, probabilities
