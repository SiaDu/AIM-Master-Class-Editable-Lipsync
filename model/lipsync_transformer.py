import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import List, Tuple


class LipsyncTransformer(nn.Module):
    """Audio‑to‑offset transformer network for fast training or inference.

    Parameters
    ----------
    audio_dim : int
        Dimensionality of audio features (default 464).
    n_heads : int
        Number of attention heads.
    n_layers : int
        Number of encoder layers.
    ff_dim : int
        Feed‑forward layer width.
    dropout : float
        Dropout probability.
    max_seq_len : int
        Maximum sequence length supported by learnable positional embedding.
    """

    def __init__(
        self,
        audio_dim: int = 464,
        n_heads: int = 8,
        n_layers: int = 2,
        ff_dim: int = 512,
        dropout: float = 0.5,
        max_seq_len: int = 30,
    ) -> None:
        super().__init__()

        # 1. Audio feature projection
        self.input_proj = nn.Linear(audio_dim, ff_dim)

        # 2. Learnable positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, ff_dim))

        # 3. Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=ff_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # 4. Output projection to 284‑dim PCA offset
        self.output_proj = nn.Linear(ff_dim, 284)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Forward pass

        Parameters
        ----------
        audio : torch.Tensor
            Shape (B, T, 464)
        Returns
        -------
        torch.Tensor
            Shape (B, T, 284)
        """
        B, T, _ = audio.shape
        x = self.input_proj(audio)
        x = x + self.pos_embed[:, :T, :]
        x = self.transformer(x)
        return self.output_proj(x)


def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: ReduceLROnPlateau,
    num_epochs: int = 30,
    device: str = "cuda",
    patience: int = 20,
    save_path: str = "best_model_Trans.pth",
) -> Tuple[List[float], List[float]]:
    """Train LipsyncTransformer with early stopping and gradient clipping.

    Returns
    -------
    train_losses, val_losses : list[float]
    """
    model = model.to(device)
    train_losses: List[float] = []
    val_losses: List[float] = []
    best_val = float("inf")
    wait = 0

    for epoch in range(num_epochs):
        # ---- Train ----
        model.train()
        running = 0.0
        for audio, expr in train_loader:
            audio, expr = audio.to(device), expr.to(device)
            pred = model(audio)
            loss = ((pred - expr) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += loss.item()
        train_loss = running / len(train_loader)
        train_losses.append(train_loss)

        # ---- Validate ----
        model.eval()
        v_running = 0.0
        with torch.no_grad():
            for audio, expr in val_loader:
                audio, expr = audio.to(device), expr.to(device)
                pred = model(audio)
                v_running += ((pred - expr) ** 2).mean().item()
        val_loss = v_running / len(val_loader)
        val_losses.append(val_loss)

        scheduler.step(val_loss)
        print(f"Epoch {epoch+1}/{num_epochs} | train {train_loss:.4f} | val {val_loss:.4f}")

        # early stopping
        if val_loss < best_val:
            best_val = val_loss
            wait = 0
            torch.save(model.state_dict(), save_path)
            print(f"  🔥 best model saved to {save_path}")
        else:
            wait += 1
            if wait >= patience:
                print("⏹️  early stopping triggered")
                break

    return train_losses, val_losses


def plot_loss_curve(train_losses: List[float], val_losses: List[float]) -> None:
    """Plot training / validation MSE curves."""
    plt.figure(figsize=(7, 4))
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Lipsync Training Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
