"""
Fourier Neural Operator (FNO) — Continuous Spectral Convolution

Replaces discrete Attention with a global integral operator in the frequency
domain.  The key insight: global spatial interactions (Attention's purpose)
become pointwise multiplications in Fourier space.

    (K · u)(x) = F^{-1}[ R_phi · F[u] ](x)

where F is the FFT, R_phi is a learnable complex weight tensor, and the
entire operation runs in O(L log L) — breaking the O(L^2) Attention barrier.

Three levels of abstraction:
    1. SpectralConv1d    — single Fourier convolution layer
    2. FNOBlock          — spectral conv + residual MLP + time conditioning
    3. FNOBackbone       — full stack of FNO blocks (replaces Transformer)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SpectralConv1d(nn.Module):
    """
    1-D Spectral Convolution via FFT.

    Learns a complex weight tensor R_phi of shape (in_channels, out_channels, modes)
    in the frequency domain.  Only the lowest `modes` Fourier modes are modulated;
    higher modes pass through unaltered (implicit low-pass + learned filter).

    Parameters
    ----------
    in_channels : int
    out_channels : int
    modes : int
        Number of Fourier modes to learn (controls expressiveness vs cost).
    """

    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        # Complex-valued weight for frequency-domain multiplication
        scale = 1.0 / (in_channels * out_channels)
        self.weight = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    def _complex_mul(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Batched complex matrix-vector multiply in frequency domain.
        x: (B, C_in, modes)   w: (C_in, C_out, modes) → (B, C_out, modes)
        """
        return torch.einsum("bcm,com->bom", x, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, C, L) — spatial-domain signal
        Returns : (B, C_out, L)
        """
        B, C, L = x.shape

        # Forward FFT (real → complex, only positive frequencies)
        x_ft = torch.fft.rfft(x, dim=-1)  # (B, C, L//2+1)

        # Modulate the lowest `modes` frequencies
        out_ft = torch.zeros(
            B, self.out_channels, L // 2 + 1,
            device=x.device, dtype=torch.cfloat
        )
        k = min(self.modes, L // 2 + 1)
        out_ft[:, :, :k] = self._complex_mul(x_ft[:, :, :k], self.weight[:, :, :k])

        # Fix BUG-4: High-frequency pass-through
        # Preserve high-freq details that live above the truncated modes
        if self.in_channels == self.out_channels and (L // 2 + 1) > k:
            out_ft[:, :, k:] = x_ft[:, :, k:]

        # Inverse FFT back to spatial domain
        return torch.fft.irfft(out_ft, n=L, dim=-1)  # (B, C_out, L)


class FNOBlock(nn.Module):
    """
    Single FNO block: spectral convolution + pointwise MLP + residual + time.

    Architecture:
        h = SpectralConv(x) + Linear(x)     (global + local)
        h = h + time_embedding               (time conditioning)
        h = activation(h)
        h = MLP(h) + x                       (residual)
    """

    def __init__(
        self,
        channels: int,
        modes: int,
        time_dim: int = 0,
        mlp_ratio: float = 2.0,
    ):
        super().__init__()
        self.spectral_conv = SpectralConv1d(channels, channels, modes)
        # Pointwise (1x1) for channel mixing
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1)
        
        # Design Fix: Depthwise Conv1d (3x3) to inject local positional bias
        # and break pure spectral shift-equivariance
        self.local_conv = nn.Conv1d(
            channels, channels, kernel_size=3, padding=1, groups=channels
        )

        # Engineering Fix: Robust GroupNorm finding GCD
        num_groups = math.gcd(8, channels)
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=channels)

        mlp_hidden = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv1d(channels, mlp_hidden, 1),
            nn.GELU(),
            nn.Conv1d(mlp_hidden, channels, 1),
        )

        # Time conditioning (additive bias per channel)
        self.has_time = time_dim > 0
        if self.has_time:
            self.time_proj = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_dim, channels),
            )

    def forward(self, x: torch.Tensor, t_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x : (B, C, L)
        t_emb : (B, time_dim) or None
        """
        # Global (spectral) + local (pointwise) + local (depthwise pos)
        h = self.spectral_conv(x) + self.pointwise(x) + self.local_conv(x)

        # Time conditioning
        if self.has_time and t_emb is not None:
            t_bias = self.time_proj(t_emb).unsqueeze(-1)  # (B, C, 1)
            h = h + t_bias

        h = self.norm(h)
        h = F.gelu(h)

        # Residual MLP
        h = self.mlp(h) + x
        return h


class SinusoidalTimeEmbed(nn.Module):
    """Fourier features for scalar time."""

    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 0:
            t = t.unsqueeze(0)
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10_000) * torch.arange(half, device=t.device, dtype=t.dtype) / half
        )
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class FNOBackbone(nn.Module):
    """
    Full FNO backbone — replaces the Transformer encoder entirely.

    No Q, K, V matrices.  No Attention.  Global context is captured
    purely through spectral convolutions in O(L log L).

    Parameters
    ----------
    d_model : int
        Channel width (analogous to Transformer d_model).
    n_layers : int
        Number of FNO blocks.
    modes : int
        Number of Fourier modes per spectral convolution.
    time_embed_dim : int
        Dimension of sinusoidal time embedding.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 4,
        modes: int = 32,
        time_embed_dim: int = 64,
    ):
        super().__init__()
        self.d_model = d_model
        self.time_embed = SinusoidalTimeEmbed(time_embed_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(time_embed_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        self.blocks = nn.ModuleList([
            FNOBlock(d_model, modes, time_dim=d_model)
            for _ in range(n_layers)
        ])
        num_groups = math.gcd(8, d_model)
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=d_model)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x : (B, C, L) — channel-first representation
        t : (B,) — time

        Returns : (B, C, L)
        """
        t_emb = self.time_proj(self.time_embed(t))  # (B, D)

        for block in self.blocks:
            x = block(x, t_emb)

        return self.norm(x)
