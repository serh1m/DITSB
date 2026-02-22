"""
Continuous-Time Vector Field Parameterisation

The vector field v_θ(Z_t, t) defines the instantaneous velocity (drift) of the
state Z_t at time t on the data manifold.  The ODE solver integrates this field
to move samples along the learned optimal-transport trajectory.

Two architectures are provided:
  - ContinuousVectorField : lightweight MLP (good for 2-D demos)
  - DeepVectorField       : deeper residual MLP with sinusoidal time embedding
"""

import math
import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
#  Time embedding helpers
# --------------------------------------------------------------------------- #

class SinusoidalTimeEmbedding(nn.Module):
    """Fourier-feature positional encoding for the scalar time variable t."""

    def __init__(self, embed_dim: int):
        super().__init__()
        assert embed_dim % 2 == 0, "embed_dim must be even"
        self.embed_dim = embed_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) or scalar
        if t.dim() == 0:
            t = t.unsqueeze(0)
        half = self.embed_dim // 2
        freqs = torch.exp(
            -math.log(10_000) * torch.arange(half, device=t.device, dtype=t.dtype) / half
        )
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)       # (B, half)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, embed_dim)


# --------------------------------------------------------------------------- #
#  Lightweight vector field (for quick prototyping / 2-D demos)
# --------------------------------------------------------------------------- #

class ContinuousVectorField(nn.Module):
    """
    Simple MLP that concatenates the scalar time t to the state z and outputs
    the velocity dz/dt.  Matches the ODE-solver signature ``forward(t, z)``.
    """

    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # API required by torchdiffeq: forward(t, state)
        # t can be a scalar (from ODE solver) or (B,) / (B,1) (from loss fn)
        if z.dim() == 1:
            z = z.unsqueeze(0)
        if t.dim() == 0:
            # Scalar t → expand to (B, 1)
            t_vec = t.unsqueeze(0).expand(z.shape[0]).unsqueeze(-1)
        else:
            t_vec = t.reshape(-1, 1)
        zt = torch.cat([z, t_vec], dim=-1)
        return self.net(zt)


# --------------------------------------------------------------------------- #
#  Deeper residual vector field (better gradient flow, richer manifolds)
# --------------------------------------------------------------------------- #

class _ResBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class DeepVectorField(nn.Module):
    """
    Residual MLP with sinusoidal time embedding.  Better suited for
    higher-dimensional state spaces and longer training runs.
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 256,
        time_embed_dim: int = 64,
        num_res_blocks: int = 3,
    ):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)
        self.input_proj = nn.Linear(state_dim + time_embed_dim, hidden_dim)
        self.res_blocks = nn.Sequential(
            *[_ResBlock(hidden_dim) for _ in range(num_res_blocks)]
        )
        self.output_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 1:
            z = z.unsqueeze(0)
        # t can be scalar (ODE solver) or (B,) (loss function)
        if t.dim() == 0:
            t_batch = t.unsqueeze(0).expand(z.shape[0])
        else:
            t_batch = t.reshape(-1)
        t_emb = self.time_embed(t_batch)  # (B, time_embed_dim)
        h = torch.cat([z, t_emb], dim=-1)
        h = self.input_proj(h)
        h = self.res_blocks(h)
        return self.output_proj(h)
