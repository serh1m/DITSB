"""
Discrete Flow Matching Loss — CTMC on the Probability Simplex

Forward noise process:
    With probability t, replace each token with a uniform random token.
    With probability (1-t), keep the original token.

Training objective:
    Predict the original clean token x_1 from the noised x_t and time t.
    Loss = E_t[ CrossEntropy( model(x_t, t),  x_1 ) ]

This is mathematically equivalent to learning the reverse CTMC rate matrix
on the (V-1)-dimensional probability simplex.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .discrete_flow import CategoricalFlowMatcher


def discrete_flow_matching_loss(
    model: nn.Module,
    x_1: torch.Tensor,
    vocab_size: int,
    mask_token: int | None = None,
) -> torch.Tensor:
    """
    Parameters
    ----------
    model : nn.Module
        Callable with signature ``forward(x_t, t) -> logits (B, L, V)``.
    x_1 : (B, L) long tensor
        Clean token sequences from the data distribution.
    vocab_size : int
        Size of the vocabulary.
    mask_token : int or None
        If given, use an absorbing-state process (mask with a dedicated token).
        If None, use a uniform-noise process (replace with random tokens).

    Returns
    -------
    loss : scalar tensor  (cross-entropy).
    """
    B, L = x_1.shape
    device = x_1.device

    # 1. Sample time t ~ U[eps, 1] for each sequence in the batch
    #    (avoid t=0 exactly to prevent degenerate gradients)
    t = torch.rand(B, device=device) * 0.99 + 0.01       # (B,)

    # 2. Forward noise process: corrupt each position independently
    #    mask_prob[b, l] = t[b]  (same corruption rate for all positions)
    mask_prob = t.unsqueeze(-1).expand(B, L)               # (B, L)
    # corrupt_mask: True where token IS modified
    corrupt_mask = torch.rand(B, L, device=device) < mask_prob

    if mask_token is not None:
        # Absorbing-state process: replace with [MASK] token
        noise = torch.full_like(x_1, mask_token)
    else:
        # Uniform-noise process: replace with random token ~ Uniform(V)
        noise = torch.randint(0, vocab_size, (B, L), device=device)

    x_t = torch.where(corrupt_mask, noise, x_1)

    # 3. Forward pass: predict clean tokens
    logits = model(x_t, t)                                 # (B, L, V)

    # 4. Use the new DITSB-v2 CTMC Exact Flow Loss
    ctmc = CategoricalFlowMatcher(vocab_size).to(device)
    x1_onehot = F.one_hot(x_1, num_classes=vocab_size).float()
    
    # We pass logits, onehot, and time (time is not actively used in the simplified v2 base, but kept for signature)
    t_expanded = t.view(B, 1, 1).expand(B, L, 1)
    
    loss_unreduced = ctmc.compute_ctmc_loss(logits, x1_onehot, t_expanded)

    return loss_unreduced
