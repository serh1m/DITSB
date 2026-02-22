"""
Industrial-Grade Components for DITSB

Standardizes the architecture to match SOTA LLMs (LLaMA/Gemini) and introduces
advanced spectral operations for robust scaling.

Components:
1. RMSNorm:      More stable than LayerNorm, scales gradients by 1/RMS.
2. SwiGLU:       Gated linear unit with Swish activation, superior to GELU.
3. SpectralRoPE: Frequency-domain Rotational Positional Embedding.
                 Applies relative phase shifts to frequency modes.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .fno import SpectralConv1d


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class SwiGLU(nn.Module):
    """
    Swish Gated Linear Unit.
    """
    def __init__(
        self,
        d_model: int,
        hidden_dim: int | None = None,
        multiple_of: int = 256,
        bias: bool = False,
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * d_model
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(d_model, hidden_dim, bias=bias)
        self.w2 = nn.Linear(d_model, hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, d_model, bias=bias)

    def forward(self, x):
        # x is (B, L, C) usually, but FNO uses (B, C, L).
        # We need to be careful with dimensions.
        # SwiGLU expects last dim = d_model.
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class SpectralRoPE(nn.Module):
    """
    Spectral Rotary Positional Embedding (Phase Shift Mixing).
    
    Learns a phase shift delta_c for each channel.
    The frequency mode k is rotated by angle theta = -2pi * k * delta_c.
    
    This corresponds to shifting the spatial signal of channel c by delta_c * L.
    Because delta_c varies across channels, the subsequent pointwise mixing
    (1x1 Conv) operates on features from DIFFERENT relative positions.
    """
    def __init__(self, channels: int, modes: int):
        super().__init__()
        self.channels = channels
        self.modes = modes
        
        # Learnable shift parameter per channel
        # Initialized small to simulate "near-local" mixing
        self.shifts = nn.Parameter(torch.randn(channels) * 0.05)
        
        # Fixed frequencies k = 0, 1, ..., modes-1
        self.register_buffer("freqs", torch.arange(modes).float()) 

    def forward(self, x_ft: torch.Tensor) -> torch.Tensor:
        """
        x_ft: (B, C, modes) -- complex frequency coefficients
        Length L is implicit in the phase scaling, we normalize to [0, 1] interval.
        """
        # Theta(c, k) = -2pi * k * shift_c
        # Shape: (C, 1) * (1, M) -> (C, M)
        theta = -2 * math.pi * self.shifts.unsqueeze(-1) * self.freqs.unsqueeze(0)
        
        # Complex rotation e^{i theta}
        rot = torch.polar(torch.ones_like(theta), theta) # (C, M)
        
        # Apply rotation (broadcasting over Batch)
        return x_ft * rot.unsqueeze(0)


class IndustrialFNOBlock(nn.Module):
    """
    Industrial-Grade FNO Block.
    
    Upgrades:
    - Norm: RMSNorm
    - Act:  SwiGLU (using Linear layers, requires transpose)
    - Pos:  Depthwise Conv (LPI) + SpectralRoPE (Phase Mixing)
    """
    def __init__(
        self,
        channels: int,
        modes: int,
        time_dim: int = 0,
        mlp_ratio: float = 2.0,
    ):
        super().__init__()
        self.channels = channels
        self.modes = modes
        
        # 1. Spectral Branch
        # We manually implement the spectral conv logic to insert RoPE
        # We reuse the weight initialization logic from SpectralConv1d
        scale = 1.0 / (channels * channels)
        self.spectral_weight = nn.Parameter(
            scale * torch.randn(channels, channels, modes, dtype=torch.cfloat)
        )
        
        self.rope = SpectralRoPE(channels, modes)
        
        # 2. Local Branch
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1)
        self.local_conv = nn.Conv1d(
            channels, channels, kernel_size=3, padding=1, groups=channels
        )

        # 3. Norm & Act
        self.norm = RMSNorm(channels)
        self.swiglu = SwiGLU(channels, multiple_of=16)

        # 4. Time
        self.has_time = time_dim > 0
        if self.has_time:
            self.time_proj = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_dim, channels),
            )

    def _complex_mul(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bcm,com->bom", x, w)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: (B, C, L)
        """
        B, C, L = x.shape
        residual = x
        
        # --- Spectral Path with RoPE ---
        # FFT
        x_ft = torch.fft.rfft(x, dim=-1)
        
        # Select modes
        k = min(self.modes, L // 2 + 1)
        
        # Weight mult -> (B, C, k)
        # Note: self.spectral_weight slice might need channel alignment if in!=out
        # But here in==out==channels.
        w_spec = self._complex_mul(
            x_ft[:, :, :k], 
            self.spectral_weight[:, :, :k]
        )
        
        # Apply RoPE (Phase Shift)
        w_spec = self.rope(w_spec)
        
        # Prepare output spectrum
        out_ft = torch.zeros(B, C, L // 2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :k] = w_spec
        
        # High-freq pass-through (BUG-4 Fix)
        if (L // 2 + 1) > k:
             out_ft[:, :, k:] = x_ft[:, :, k:]
             
        # IFFT
        global_out = torch.fft.irfft(out_ft, n=L, dim=-1)
        
        # --- Local Path ---
        local_out = self.pointwise(x) + self.local_conv(x)
        
        h = global_out + local_out
        
        # Time
        if self.has_time and t_emb is not None:
            t_bias = self.time_proj(t_emb).unsqueeze(-1)
            h = h + t_bias
            
        # Norm (B, C, L) -> (B, L, C) -> Norm -> (B, C, L)
        h = h.transpose(1, 2)
        h = self.norm(h)
        
        # SwiGLU (B, L, C)
        h = self.swiglu(h)
        
        # Back to (B, C, L)
        h = h.transpose(1, 2)
        
        return h + residual
