"""
Discrete Flow Field — Transformer-based denoising network for
Continuous-Time Markov Chain (CTMC) flow matching on the probability simplex.

The network takes noised token sequences (x_t, t) and predicts logits
over the vocabulary for the clean tokens x_1.
"""

import math
import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """Fourier-feature encoding for scalar time t."""

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


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block with causal self-attention."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.ln1(x)
        h, _ = self.attn(h, h, h, attn_mask=attn_mask, is_causal=False)
        x = x + h
        x = x + self.ff(self.ln2(x))
        return x


class DiscreteFlowField(nn.Module):
    """
    Transformer-based denoiser for discrete flow matching (CTMC).

    Given a noised token sequence x_t and time t, predicts logits for
    the clean sequence x_1.

    Parameters
    ----------
    vocab_size : int
        Size of the discrete vocabulary.
    max_seq_len : int
        Maximum sequence length.
    d_model : int
        Transformer hidden dimension.
    n_heads : int
        Number of attention heads.
    n_layers : int
        Number of Transformer blocks.
    time_embed_dim : int
        Dimension for the sinusoidal time embedding.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 512,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        time_embed_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model

        # Token + position embeddings
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

        # Time conditioning
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(time_embed_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # Transformer body
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout) for _ in range(n_layers)
        ])
        self.ln_out = nn.LayerNorm(d_model)

        # Output projection → vocab logits
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        self.head.weight = self.tok_embed.weight

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1 and "head" not in name:  # Fix BUG-5: Don't re-init tied head
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x_t : (B, L) long — noised token IDs.
        t   : (B,) float — noise level / time.

        Returns
        -------
        logits : (B, L, V) — predicted clean-token logits.
        """
        B, L = x_t.shape
        device = x_t.device

        # Token + positional embedding
        positions = torch.arange(L, device=device).unsqueeze(0)
        h = self.tok_embed(x_t) + self.pos_embed(positions)  # (B, L, D)

        # Add time conditioning (broadcast across sequence)
        t_emb = self.time_proj(self.time_embed(t))            # (B, D)
        h = h + t_emb.unsqueeze(1)                            # (B, L, D)

        # Transformer blocks (bidirectional attention — no causal mask)
        for block in self.blocks:
            h = block(h)

        h = self.ln_out(h)
        logits = self.head(h)                                 # (B, L, V)
        return logits
