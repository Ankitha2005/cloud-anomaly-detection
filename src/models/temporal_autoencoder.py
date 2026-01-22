"""
Temporal Autoencoder for Cloud Anomaly Detection
LSTM-based autoencoder with attention mechanism that learns normal temporal patterns.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Optional, List
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class AttentionLayer(nn.Module):
    """Self-attention layer for temporal sequences."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, lstm_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # lstm_output: (batch, seq_len, hidden_dim)
        attention_weights = F.softmax(self.attention(lstm_output), dim=1)  # (batch, seq_len, 1)
        context = torch.sum(attention_weights * lstm_output, dim=1)  # (batch, hidden_dim)
        return context, attention_weights.squeeze(-1)


class LSTMEncoder(nn.Module):
    """LSTM-based encoder with attention for temporal sequences."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.2,
        use_attention: bool = True
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_attention = use_attention

        # Input projection for better feature learning
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        # Attention layer
        if use_attention:
            self.attention = AttentionLayer(hidden_dim * 2)  # *2 for bidirectional

        self.fc = nn.Linear(hidden_dim * 2, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (batch, seq_len, input_dim)
        x = self.input_proj(x)  # (batch, seq_len, hidden_dim)
        lstm_out, (h_n, c_n) = self.lstm(x)  # (batch, seq_len, hidden_dim*2)

        if self.use_attention:
            context, attn_weights = self.attention(lstm_out)
            latent = self.fc(context)
        else:
            # Concatenate forward and backward final states
            h_forward = h_n[-2]
            h_backward = h_n[-1]
            latent = self.fc(torch.cat([h_forward, h_backward], dim=1))
            attn_weights = None

        return latent, attn_weights


class LSTMDecoder(nn.Module):
    """LSTM-based decoder for temporal sequences with skip connections."""

    def __init__(
        self,
        output_dim: int,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        num_layers: int = 2,
        seq_len: int = 10,
        dropout: float = 0.2
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.output_dim = output_dim

        # Latent to hidden projection
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Learnable position embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, hidden_dim * 2))

        self.lstm = nn.LSTM(
            input_size=hidden_dim * 2,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        batch_size = latent.size(0)

        # Expand latent to sequence with position embeddings
        hidden = self.fc(latent)  # (batch, hidden_dim*2)
        hidden = hidden.unsqueeze(1).expand(-1, self.seq_len, -1)  # (batch, seq_len, hidden_dim*2)
        hidden = hidden + self.pos_embedding  # Add positional info

        # Decode
        lstm_out, _ = self.lstm(hidden)  # (batch, seq_len, hidden_dim)
        output = self.output_layer(lstm_out)  # (batch, seq_len, output_dim)

        return output


class TemporalAutoencoder(nn.Module):
    """Complete LSTM Autoencoder with attention for temporal anomaly detection."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        num_layers: int = 2,
        seq_len: int = 10,
        dropout: float = 0.2,
        use_attention: bool = True
    ):
        super().__init__()
        self.input_dim = input_dim

        self.encoder = LSTMEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            dropout=dropout,
            use_attention=use_attention
        )

        self.decoder = LSTMDecoder(
            output_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            seq_len=seq_len,
            dropout=dropout
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latent, _ = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction, latent
    
    def get_reconstruction_error(self, x: torch.Tensor, method: str = 'mse') -> torch.Tensor:
        """
        Compute per-sample reconstruction error.

        Args:
            x: Input tensor (batch, seq_len, features)
            method: 'mse', 'mae', or 'combined'
        """
        reconstruction, _ = self.forward(x)

        if method == 'mse':
            error = torch.mean((x - reconstruction) ** 2, dim=(1, 2))
        elif method == 'mae':
            error = torch.mean(torch.abs(x - reconstruction), dim=(1, 2))
        elif method == 'combined':
            mse = torch.mean((x - reconstruction) ** 2, dim=(1, 2))
            mae = torch.mean(torch.abs(x - reconstruction), dim=(1, 2))
            error = 0.5 * mse + 0.5 * mae
        else:
            error = torch.mean((x - reconstruction) ** 2, dim=(1, 2))

        return error

    def get_feature_errors(self, x: torch.Tensor) -> torch.Tensor:
        """Get per-feature reconstruction errors for explainability."""
        reconstruction, _ = self.forward(x)
        # Mean over sequence, keep features
        feature_errors = torch.mean((x - reconstruction) ** 2, dim=1)  # (batch, features)
        return feature_errors


class TemporalAnomalyDetector:
    """Wrapper class for training and inference with temporal autoencoder."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        num_layers: int = 2,
        seq_len: int = 10,
        dropout: float = 0.2,
        device: str = None
    ):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        
        self.model = TemporalAutoencoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            seq_len=seq_len,
            dropout=dropout
        ).to(device)
        
        self.threshold = None
        self.fitted = False

    def fit(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray = None,
        epochs: int = 50,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        early_stopping_patience: int = 10,
        threshold_percentile: float = 95.0,
        verbose: bool = True
    ) -> dict:
        """
        Train the autoencoder on normal data.

        Args:
            X_train: Training sequences (n_samples, seq_len, n_features)
            X_val: Validation sequences (optional)
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            early_stopping_patience: Patience for early stopping
            threshold_percentile: Percentile for anomaly threshold
            verbose: Print training progress

        Returns:
            Training history dict
        """
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        train_loader = DataLoader(
            TensorDataset(X_train_tensor),
            batch_size=batch_size,
            shuffle=True
        )

        if X_val is not None:
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_losses = []

            for batch in train_loader:
                x = batch[0]
                optimizer.zero_grad()
                reconstruction, _ = self.model(x)
                loss = criterion(reconstruction, x)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            avg_train_loss = np.mean(train_losses)
            history['train_loss'].append(avg_train_loss)

            # Validation
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    reconstruction, _ = self.model(X_val_tensor)
                    val_loss = criterion(reconstruction, X_val_tensor).item()
                history['val_loss'].append(val_loss)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        if verbose:
                            print(f"Early stopping at epoch {epoch+1}")
                        break

            if verbose and (epoch + 1) % 10 == 0:
                msg = f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f}"
                if X_val is not None:
                    msg += f" - Val Loss: {val_loss:.6f}"
                print(msg)

        # Set threshold based on training reconstruction errors
        self.model.eval()
        with torch.no_grad():
            train_errors = self.model.get_reconstruction_error(X_train_tensor).cpu().numpy()
        self.threshold = np.percentile(train_errors, threshold_percentile)

        if verbose:
            print(f"Anomaly threshold set to {self.threshold:.6f} ({threshold_percentile}th percentile)")

        self.fitted = True
        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly labels (1 = anomaly, 0 = normal)."""
        scores = self.score_samples(X)
        return (scores > self.threshold).astype(int)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Return reconstruction error as anomaly scores."""
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            errors = self.model.get_reconstruction_error(X_tensor).cpu().numpy()

        return errors

    def save(self, path: str):
        """Save model to disk."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'threshold': self.threshold
        }, path)

    def load(self, path: str):
        """Load model from disk."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.threshold = checkpoint['threshold']
        self.fitted = True

