"""
Optimal Transport Flow Matching Loss

Implements Conditional OT (Lipman et al., 2023) / Rectified Flow (Liu et al., 2023).

Given a noise-data pair (z₀, z₁):
    z_t = (1 − t)·z₀ + t·z₁          (linear interpolation)
    target velocity  u_t = z₁ − z₀    (constant along the geodesic)

The loss drives the learned vector field v_θ(z_t, t) towards u_t under MSE.
"""

import torch
import torch.nn as nn


def optimal_transport_loss(
    vector_field: nn.Module,
    real_data: torch.Tensor,
    sigma_min: float = 1e-5,
) -> torch.Tensor:
    """
    Parameters
    ----------
    vector_field : nn.Module
        Callable with signature ``forward(t, z) -> dz/dt``.
    real_data : (B, D) tensor
        A mini-batch drawn from the data distribution μ₁.
    sigma_min : float
        Optional small noise floor added to z_t for regularisation.

    Returns
    -------
    loss : scalar tensor (mean squared error).
    """
    batch_size, state_dim = real_data.shape
    device = real_data.device
    dtype = real_data.dtype

    # 1. Sample from prior μ₀ = N(0, I)
    z_0 = torch.randn(batch_size, state_dim, device=device, dtype=dtype)

    # 2. Target data z₁
    z_1 = real_data

    # 3. Random time t ~ U[0, 1]
    t = torch.rand(batch_size, device=device, dtype=dtype)

    # 4. Optimal-transport interpolation  z_t = (1 − t)·z₀ + t·z₁
    t_expand = t.unsqueeze(-1)                           # (B, 1)
    z_t = (1.0 - t_expand) * z_0 + t_expand * z_1

    # Optional noise floor
    if sigma_min > 0.0:
        z_t = z_t + sigma_min * torch.randn_like(z_t)

    # 5. Target velocity (constant along the straight-line geodesic)
    target_velocity = z_1 - z_0                          # (B, D)

    # 6. Predicted velocity from the network
    predicted_velocity = vector_field(t, z_t)             # (B, D)

    # 7. MSE loss — Hamilton-Jacobi-Bellman flow matching objective
    loss = torch.mean((predicted_velocity - target_velocity) ** 2)
    return loss
