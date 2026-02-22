"""
Discrete Flow Generation — Iterative Denoising via Reverse CTMC

Generation proceeds from fully noised tokens (t=1 → uniform random)
to clean tokens (t=0) through T discrete denoising steps.

At each step:
    1. Model predicts clean token probabilities p(x_1 | x_t, t)
    2. Sample candidate clean tokens from these probabilities
    3. Re-corrupt with reduced noise level t_{k-1} < t_k
    4. Repeat until t → 0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def discrete_generate(
    model: nn.Module,
    seq_len: int,
    vocab_size: int,
    num_samples: int = 1,
    steps: int = 50,
    device: torch.device | str = "cpu",
    temperature: float = 1.0,
    mask_token: int | None = None,
) -> torch.Tensor:
    """
    Generate token sequences by iteratively denoising from uniform noise.

    Parameters
    ----------
    model : nn.Module
        Trained DiscreteFlowField.
    seq_len : int
        Length of sequences to generate.
    vocab_size : int
        Vocabulary size.
    num_samples : int
        Batch size for generation.
    steps : int
        Number of denoising steps.
    device : torch.device
        Device.
    temperature : float
        Sampling temperature (lower = more deterministic).
    mask_token : int or None
        If given, start from all-mask tokens (absorbing state).

    Returns
    -------
    x : (num_samples, seq_len) long tensor — generated token sequences.
    """
    model.eval()
    B, L, V = num_samples, seq_len, vocab_size

    # Start from fully noised state (t = 1)
    if mask_token is not None:
        x_t = torch.full((B, L), mask_token, dtype=torch.long, device=device)
    else:
        x_t = torch.randint(0, V, (B, L), device=device)

    # Time schedule: from t=1 down to t≈0
    time_steps = torch.linspace(1.0, 0.01, steps=steps, device=device)

    for i in range(len(time_steps) - 1):
        t_now = time_steps[i]
        t_next = time_steps[i + 1]

        # Predict clean token logits
        t_batch = t_now.expand(B)
        logits = model(x_t, t_batch)                        # (B, L, V)

        # Sample predicted clean tokens
        probs = F.softmax(logits / temperature, dim=-1)     # (B, L, V)
        x_pred = torch.multinomial(
            probs.reshape(-1, V), num_samples=1
        ).reshape(B, L)

        # Determine which positions to keep clean vs re-corrupt
        # At t_next, each position has corruption probability t_next
        keep_clean = torch.rand(B, L, device=device) >= t_next

        if mask_token is not None:
            noise = torch.full_like(x_t, mask_token)
        else:
            noise = torch.randint(0, V, (B, L), device=device)

        # Positions that are "clean" get the predicted token;
        # positions that are "still noisy" get re-corrupted
        x_t = torch.where(keep_clean, x_pred, noise)

    # Final prediction at t ≈ 0 (fully clean)
    t_final = torch.full((B,), 0.01, device=device)
    logits = model(x_t, t_final)
    x_out = logits.argmax(dim=-1)

    return x_out
