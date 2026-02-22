"""
HJB Terminal Cost Loss — Reward-Augmented Flow Matching

Modified OT loss that incorporates a terminal reward signal R(x):

    L = E_t[ ||v_theta(z_t, t) - u_t||^2 ] - lambda * E[ R(z_1_pred) ]

The reward term encourages the learned flow to transport noise towards
high-reward regions of the data manifold.
"""

import torch
import torch.nn as nn


def hjb_terminal_cost_loss(
    vector_field: nn.Module,
    real_data: torch.Tensor,
    reward_fn: nn.Module,
    reward_weight: float = 0.1,
    sigma_min: float = 0.0,
) -> tuple[torch.Tensor, dict]:
    """
    Parameters
    ----------
    vector_field : nn.Module
        Callable with signature ``forward(t, z) -> dz/dt``.
    real_data : (B, D) tensor
    reward_fn : nn.Module
        Maps (B, D) → (B,) scalar rewards.
    reward_weight : float
        Lambda weighting for the terminal reward term.
    sigma_min : float
        Optional noise floor.

    Returns
    -------
    loss : scalar tensor
    info : dict with 'ot_loss' and 'reward_term' for logging.
    """
    batch_size, state_dim = real_data.shape
    device = real_data.device
    dtype = real_data.dtype

    # Standard OT flow matching components
    z_0 = torch.randn(batch_size, state_dim, device=device, dtype=dtype)
    z_1 = real_data
    t = torch.rand(batch_size, device=device, dtype=dtype)
    t_expand = t.unsqueeze(-1)

    z_t = (1.0 - t_expand) * z_0 + t_expand * z_1

    if sigma_min > 0.0:
        z_t = z_t + sigma_min * torch.randn_like(z_t)

    target_velocity = z_1 - z_0
    predicted_velocity = vector_field(t, z_t)

    # OT matching loss
    ot_loss = torch.mean((predicted_velocity - target_velocity) ** 2)

    # Terminal reward: predict where the flow would end up
    # Approximate: z_1_pred ≈ z_t + (1 - t) * v_theta(z_t, t)
    z_1_pred = z_t + (1.0 - t_expand) * predicted_velocity

    # Reward on predicted endpoints (we want to maximise this)
    reward = reward_fn(z_1_pred)
    reward_term = reward.mean()

    # Combined loss: minimise OT error, maximise reward
    loss = ot_loss - reward_weight * reward_term

    info = {
        "ot_loss": ot_loss.item(),
        "reward_term": reward_term.item(),
        "total_loss": loss.item(),
    }

    return loss, info
