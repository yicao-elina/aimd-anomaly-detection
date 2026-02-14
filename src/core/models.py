"""
LSTM Autoencoder for temporal anomaly detection (Level 3).
Learns to reconstruct normal AIMD windows; high reconstruction error = anomaly.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from typing import Optional, Tuple, List
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class LSTMEncoder(nn.Module):
    def __init__(self, n_features: int, hidden_sizes: List[int] = [64, 32]):
        super().__init__()
        self.lstm1 = nn.LSTM(n_features, hidden_sizes[0], batch_first=True)
        self.lstm2 = nn.LSTM(hidden_sizes[0], hidden_sizes[1], batch_first=True)
        self.bottleneck_size = hidden_sizes[1]

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        out, _ = self.lstm1(x)
        out, (h, _) = self.lstm2(out)
        # h: (1, batch, hidden_sizes[1]) → bottleneck
        return h.squeeze(0)  # (batch, hidden_sizes[1])


class LSTMDecoder(nn.Module):
    def __init__(self, n_features: int, seq_len: int, hidden_sizes: List[int] = [32, 64]):
        super().__init__()
        self.seq_len = seq_len
        self.lstm1 = nn.LSTM(hidden_sizes[0], hidden_sizes[1], batch_first=True)
        self.output_layer = nn.Linear(hidden_sizes[1], n_features)

    def forward(self, z):
        # z: (batch, bottleneck) → repeat to (batch, seq_len, bottleneck)
        z_rep = z.unsqueeze(1).repeat(1, self.seq_len, 1)
        out, _ = self.lstm1(z_rep)
        return self.output_layer(out)  # (batch, seq_len, n_features)


class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features: int, seq_len: int, hidden_sizes: List[int] = [64, 32]):
        super().__init__()
        self.encoder = LSTMEncoder(n_features, hidden_sizes)
        self.decoder = LSTMDecoder(n_features, seq_len, list(reversed(hidden_sizes)))

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


class LSTMAnomalyDetector:
    """
    Wrapper around LSTMAutoencoder for training, inference, and anomaly scoring.
    """

    def __init__(
        self,
        n_features: int,
        seq_len: int,
        hidden_sizes: List[int] = [64, 32],
        device: Optional[str] = None,
    ):
        self.n_features = n_features
        self.seq_len = seq_len
        self.device = torch.device(
            device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        )
        self.model = LSTMAutoencoder(n_features, seq_len, hidden_sizes).to(self.device)
        self.threshold: Optional[float] = None
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []

    def fit(
        self,
        X_train: np.ndarray,   # (n_windows, seq_len, n_features)
        X_val: np.ndarray,
        epochs: int = 60,
        batch_size: int = 32,
        lr: float = 1e-3,
        patience: int = 10,
    ):
        """Train autoencoder on normal AIMD windows."""
        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        X_val_t = torch.tensor(X_val, dtype=torch.float32)

        train_loader = DataLoader(
            TensorDataset(X_train_t),
            batch_size=batch_size, shuffle=True,
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5
        )

        best_val_loss = float('inf')
        best_state = None
        no_improve = 0

        print(f"Training LSTM Autoencoder on {self.device}  "
              f"({len(X_train)} train, {len(X_val)} val windows)...")

        for epoch in range(1, epochs + 1):
            # Train
            self.model.train()
            train_loss = 0.0
            for (batch,) in train_loader:
                batch = batch.to(self.device)
                recon = self.model(batch)
                loss = criterion(recon, batch)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item() * len(batch)
            train_loss /= len(X_train)

            # Validate
            self.model.eval()
            with torch.no_grad():
                val_recon = self.model(X_val_t.to(self.device))
                val_loss = criterion(val_recon, X_val_t.to(self.device)).item()

            scheduler.step(val_loss)
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            if epoch % 10 == 0:
                print(f"  Epoch {epoch:3d}/{epochs}  "
                      f"train={train_loss:.5f}  val={val_loss:.5f}")

            # Early stopping
            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"  Early stop at epoch {epoch}")
                    break

        if best_state:
            self.model.load_state_dict(best_state)

        # Set anomaly threshold at 95th percentile of val reconstruction error
        errors = self.reconstruction_errors(X_val)
        self.threshold = float(np.percentile(errors, 95))
        print(f"✓ Training complete. Anomaly threshold = {self.threshold:.5f}")

    def reconstruction_errors(self, X: np.ndarray) -> np.ndarray:
        """Per-window MSE reconstruction error. Shape: (n_windows,)"""
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            recon = self.model(X_t)
        errors = torch.mean((recon - X_t) ** 2, dim=(1, 2)).cpu().numpy()
        return errors.astype(np.float32)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return 1 for anomaly, 0 for normal. Shape: (n_windows,)"""
        assert self.threshold is not None, "Call fit() first"
        errors = self.reconstruction_errors(X)
        return (errors > self.threshold).astype(int)

    def save(self, path: str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state': self.model.state_dict(),
            'n_features': self.n_features,
            'seq_len': self.seq_len,
            'threshold': self.threshold,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }, str(path))
        print(f"✓ LSTM Autoencoder saved → {path}")

    @classmethod
    def load(cls, path: str) -> 'LSTMAnomalyDetector':
        ckpt = torch.load(str(path), map_location='cpu')
        det = cls(ckpt['n_features'], ckpt['seq_len'])
        det.model.load_state_dict(ckpt['model_state'])
        det.threshold = ckpt['threshold']
        det.train_losses = ckpt.get('train_losses', [])
        det.val_losses = ckpt.get('val_losses', [])
        return det

    def plot_training_curves(self, save_path: str):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(self.train_losses, label='Train Loss')
        ax.plot(self.val_losses, label='Val Loss')
        ax.axhline(self.threshold, color='red', linestyle='--', label=f'Threshold={self.threshold:.4f}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE Loss')
        ax.set_title('LSTM Autoencoder Training Curves')
        ax.legend()
        fig.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"✓ Training curves → {save_path}")
