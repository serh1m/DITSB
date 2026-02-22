"""
FNO-Based Discrete Flow Field — Zero-Attention Language Model

Replaces the Transformer backbone in DiscreteFlowField with a pure
Fourier Neural Operator.  Global context is captured entirely through
spectral convolutions — no Q, K, V, no Attention, O(L log L).

The architecture mirrors DiscreteFlowField's interface exactly:
    forward(x_t, t) -> logits (B, L, V)

so it is a drop-in replacement for training and generation.
"""

import torch
import torch.nn as nn
from .fno import FNOBackbone


class FNODiscreteFlowField(nn.Module):
    """
    FNO-based denoiser for discrete flow matching.

    Parameters
    ----------
    vocab_size : int
    max_seq_len : int
    d_model : int
        FNO channel width.
    n_layers : int
        Number of FNO blocks.
    modes : int
        Fourier modes per spectral conv (controls frequency resolution).
        Rule of thumb: modes ≈ seq_len // 4.
    time_embed_dim : int
    dropout : float
    """

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 512,
        d_model: int = 256,
        n_layers: int = 4,
        modes: int = 32,
        time_embed_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        # Token + position embeddings
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.embed_drop = nn.Dropout(dropout)

        # FNO backbone (replaces Transformer)
        self.backbone = FNOBackbone(
            d_model=d_model,
            n_layers=n_layers,
            modes=modes,
            time_embed_dim=time_embed_dim,
        )

        # Output head
        self.ln_out = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        self.head.weight = self.tok_embed.weight

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            # Fix BUG-5: Don't re-init tied head
            if p.dim() > 1 and not torch.is_complex(p) and "head" not in name:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x_t : (B, L) long — noised token IDs
        t   : (B,) float — noise time

        Returns
        -------
        logits : (B, L, V)
        """
        B, L = x_t.shape
        device = x_t.device

        # Embed tokens + positions → (B, L, D)
        positions = torch.arange(L, device=device).unsqueeze(0)
        h = self.tok_embed(x_t) + self.pos_embed(positions)
        h = self.embed_drop(h)

        # Transpose to channel-first for FNO: (B, L, D) → (B, D, L)
        h = h.transpose(1, 2)

        # FNO: global spectral interaction in O(L log L)
        h = self.backbone(h, t)

        # Back to sequence-first: (B, D, L) → (B, L, D)
        h = h.transpose(1, 2)

        # Project to vocab logits
        h = self.ln_out(h)
        logits = self.head(h)  # (B, L, V)

        return logits


class FNOContinuousVectorField(nn.Module):
    """
    FNO-based continuous vector field for 2D/ND flow matching.

    Unlike the Transformer variant, this uses spectral convolutions
    to model interactions across the state dimensions.

    For low-dimensional state spaces (2D demos), this is equivalent
    to a very efficient global MLP, but scales gracefully to high-D.
    """

    def __init__(self, state_dim: int, hidden_dim: int = 128, modes: int = 16, n_layers: int = 3):
        super().__init__()
        self.state_dim = state_dim
        self.backbone = FNOBackbone(
            d_model=hidden_dim,
            n_layers=n_layers,
            modes=modes,
            time_embed_dim=64,
        )
        self.input_proj = nn.Linear(state_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, state_dim)

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        t : scalar or (B,)
        z : (B, D)
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)
        B = z.shape[0]

        if t.dim() == 0:
            t_batch = t.unsqueeze(0).expand(B)
        else:
            t_batch = t.reshape(-1)

        # Project to hidden dim → treat as (B, hidden_dim, 1) for FNO
        h = self.input_proj(z)           # (B, hidden_dim)
        h = h.unsqueeze(-1)              # (B, hidden_dim, 1) — "length-1 sequence"
        h = self.backbone(h, t_batch)    # (B, hidden_dim, 1)
        h = h.squeeze(-1)               # (B, hidden_dim)
        return self.output_proj(h)       # (B, state_dim)
