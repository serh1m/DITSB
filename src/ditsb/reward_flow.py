"""
Reward-Guided Flow — HJB Terminal Cost for Alignment

In classical RLHF, a reward model R(x) is used post-hoc to fine-tune a
pre-trained model via PPO.  In the Hamiltonian-Jacobi-Bellman (HJB) framework,
the reward enters as a terminal potential in the optimal control problem,
naturally bending the flow trajectories towards high-reward regions.

Two modes of operation:

1. **Training-time reward**:  Modified OT loss with terminal reward term.
   L = E[ ||v_theta - u_t||^2 ] - lambda * E[ R(x_1_pred) ]

2. **Inference-time guidance**: Classifier-free guidance style.
   v_guided(x, t) = v_theta(x, t) + lambda * grad_x R(x)
"""

import torch
import torch.nn as nn


class RewardGuidedFlow(nn.Module):
    """
    Wraps a base flow model and adds gradient-based reward guidance
    at inference time.

    Parameters
    ----------
    base_flow : nn.Module
        A trained DITSB_GenerativeFlow.
    reward_fn : callable
        Maps state x (B, D) → scalar reward (B,).
    guidance_scale : float
        Strength of reward guidance (lambda). Higher = stronger steering.
    """

    def __init__(
        self,
        base_flow: nn.Module,
        reward_fn: callable,
        guidance_scale: float = 1.0,
    ):
        super().__init__()
        self.base_flow = base_flow
        self.reward_fn = reward_fn
        self.guidance_scale = guidance_scale

    def guided_vector_field(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Reward-guided velocity:
            v_guided(x, t) = v_theta(x, t) + lambda * grad_x R(x)
        """
        # Base velocity
        v_base = self.base_flow.vector_field(t, x)

        # Reward gradient (classifier-free guidance)
        x_grad = x.detach().requires_grad_(True)
        R = self.reward_fn(x_grad)
        if R.dim() > 1:
            R = R.sum(dim=-1)
        grad_R = torch.autograd.grad(R.sum(), x_grad, create_graph=False)[0]

        return v_base + self.guidance_scale * grad_R

    def generate(
        self,
        num_samples: int,
        state_dim: int,
        device: torch.device | str = "cpu",
        steps: int = 100,
    ) -> torch.Tensor:
        """
        Generate reward-guided samples via Euler integration.

        Unlike the base flow which uses odeint, we manually integrate
        to inject the reward gradient at each step.
        """
        self.eval()
        z = torch.randn(num_samples, state_dim, device=device)
        t_span = torch.linspace(0.0, 1.0, steps=steps, device=device)

        for i in range(steps - 1):
            dt = t_span[i + 1] - t_span[i]
            t_now = t_span[i]

            # Base velocity (no grad needed)
            with torch.no_grad():
                v_base = self.base_flow.vector_field(t_now, z)

            # Reward gradient (needs grad locally)
            z_req = z.detach().requires_grad_(True)
            with torch.enable_grad():
                R = self.reward_fn(z_req)
                if R.dim() > 1:
                    R = R.sum(dim=-1)
                grad_R = torch.autograd.grad(R.sum(), z_req)[0]

            v_guided = v_base + self.guidance_scale * grad_R.detach()
            z = z + dt * v_guided

        return z


class RewardFunction(nn.Module):
    """
    Simple learnable reward function R(x) : R^D -> R.

    Can be pre-trained on preference data (like a reward model in RLHF)
    or defined analytically for experiments.
    """

    def __init__(self, state_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
