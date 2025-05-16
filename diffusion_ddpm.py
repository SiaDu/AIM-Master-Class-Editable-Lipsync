import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm

# -----------------------------------------------------------------------------
#  Utility: pre‑compute DDPM schedules
# -----------------------------------------------------------------------------

def ddpm_schedules(beta1: float, beta2: float, T: int) -> Dict[str, torch.Tensor]:
    """Return buffers (length T+1) used by vanilla DDPM.

    All tensors are on CPU; they will be registered as buffers and move with
    the parent module when `.to(device)` is called.
    """
    assert 0.0 < beta1 < beta2 < 1.0, "betas must satisfy 0 < beta1 < beta2 < 1"

    idx = torch.arange(T + 1, dtype=torch.float32)
    beta_t = beta1 + (beta2 - beta1) * idx / T            # linear schedule

    alpha_t       = 1.0 - beta_t
    log_alpha_t   = torch.log(alpha_t)
    alphabar_t    = torch.cumsum(log_alpha_t, dim=0).exp()

    out: Dict[str, torch.Tensor] = {
        "sqrt_beta_t"      : beta_t.sqrt(),
        "alpha_t"          : alpha_t,
        "oneover_sqrta"    : (1.0 / alpha_t.sqrt()),
        "alphabar_t"       : alphabar_t,
        "sqrtab"           : alphabar_t.sqrt(),
        "sqrtmab"          : (1.0 - alphabar_t).sqrt(),
        "mab_over_sqrtmab": (1.0 - alpha_t) / (1.0 - alphabar_t).sqrt(),
    }
    return out

# -----------------------------------------------------------------------------
#  DDPM wrapper
# -----------------------------------------------------------------------------

class DDPM(nn.Module):
    """Generic DDPM shell that wraps a noise‑prediction network (epsilon‑theta).

    Parameters
    ----------
    nn_model : nn.Module
        Network mapping `(x_t, cond, t) -> ε`.
    betas : tuple[float, float]
        Linear schedule (beta1, beta2).
    n_T : int
        Number of diffusion steps.
    device : str
        'cpu' or 'cuda'.
    """

    def __init__(self, nn_model: nn.Module, betas: Tuple[float, float], n_T: int, device: str = "cuda"):
        super().__init__()
        self.nn_model = nn_model.to(device)
        self.n_T      = n_T
        self.device   = device
        self.loss_fn  = nn.MSELoss()

        # register diffusion buffers
        for k, v in ddpm_schedules(*betas, n_T).items():
            self.register_buffer(k, v)

    # ------------------------------------------------------------------ train
    def forward(self, x0: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """DDPM training step (predict noise)."""
        B = x0.size(0)
        t = torch.randint(1, self.n_T, (B,), device=self.device)
        eps = torch.randn_like(x0)

        x_t = self.sqrtab[t, None, None] * x0 + self.sqrtmab[t, None, None] * eps
        eps_pred = self.nn_model(x_t, cond, t)
        return self.loss_fn(eps, eps_pred)

    # ---------------------------------------------------------------- sample
    @torch.no_grad()
    def sample(self, n: int, cond: torch.Tensor, size: Tuple[int, ...], guide_w: float = 0.0) -> torch.Tensor:
        """Classifier‑free guidance sampling."""
        x = torch.randn(n, *size, device=self.device)
        cond = cond.float()

        for i in tqdm(range(self.n_T, 0, -1)):
            t = torch.full((n,), i, device=self.device, dtype=torch.long)

            eps_cond = self.nn_model(x, cond, t)
            eps_uncond = self.nn_model(x, torch.zeros_like(cond), t)
            eps = eps_uncond + guide_w * (eps_cond - eps_uncond)

            z = torch.randn_like(x) if i > 1 else 0.0
            x = (self.oneover_sqrta[i] * (x - eps * self.mab_over_sqrtmab[i]) +
                 self.sqrt_beta_t[i] * z)
        return x

# -----------------------------------------------------------------------------
#  Denoise Transformer
# -----------------------------------------------------------------------------

class DenoiseTransformer(nn.Module):
    """Backbone that predicts ε given (x_t, audio, t)."""

    def __init__(
        self,
        pose_dim: int = 284,
        audio_dim: int = 464,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 4,
        dropout: float = 0.1,
        max_seq: int = 30,
    ) -> None:
        super().__init__()

        self.pose_input_proj  = nn.Linear(pose_dim, d_model)
        self.audio_proj = nn.Linear(audio_dim, d_model)
        self.pos_embed  = nn.Parameter(torch.randn(1, max_seq, d_model))

        self.time_embed = nn.Sequential(
            nn.Linear(1, d_model), nn.ReLU(), nn.Linear(d_model, d_model)
        )

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.output_proj = nn.Linear(d_model, pose_dim)

    def forward(self, x_t: torch.Tensor, audio: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        B, T, _ = x_t.shape

        h = self.pose_input_proj(x_t) + self.audio_proj(audio) + self.pos_embed[:, :T]
        t = (t.float() / 1000).view(B, 1)
        h = h + self.time_embed(t).unsqueeze(1)

        h = self.transformer(h)
        return self.output_proj(h)
