"""
Zero-Dissipation Generation (Inference)

After training, generation is a deterministic ODE solve from pure noise z₀
along the learned vector field up to t = 1.
"""

import torch
from .flow import DITSB_GenerativeFlow


@torch.no_grad()
def generate_samples(
    flow_model: DITSB_GenerativeFlow,
    num_samples: int,
    state_dim: int,
    device: torch.device | str = "cpu",
    steps: int = 100,
    return_trajectory: bool = False,
) -> torch.Tensor:
    """
    Parameters
    ----------
    flow_model : DITSB_GenerativeFlow
        A trained generative flow.
    num_samples : int
        Number of samples to generate.
    state_dim : int
        Dimensionality of each sample.
    device : torch.device | str
        Device for generation.
    steps : int
        Number of time-discretisation steps in [0, 1].
    return_trajectory : bool
        If True, return the full (T, B, D) trajectory; otherwise only
        the final state (B, D).

    Returns
    -------
    samples : (B, D) or (T, B, D) tensor.
    """
    flow_model.eval()

    # Initial entropy state: pure isotropic Gaussian noise
    z_init = torch.randn(num_samples, state_dim, device=device)

    # Uniform time grid from 0 → 1
    t_span = torch.linspace(0.0, 1.0, steps=steps, device=device)

    # Integrate the ODE
    trajectory = flow_model(z_init, t_span)  # (T, B, D)

    if return_trajectory:
        return trajectory

    # Return only the final manifold state
    return trajectory[-1]
