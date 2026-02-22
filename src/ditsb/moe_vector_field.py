"""
Mixture-of-Expert Vector Field — Basis Function Expansion on the Tangent Space

In the continuous-flow MoE, the velocity field is a weighted sum of K expert
tangent fields:

    v_theta(x, t) = sum_k  alpha_k(x, t) * phi_k(x, t)

where:
    alpha_k(x, t) : soft routing weights (scalar field on the manifold)
    phi_k(x, t)   : expert basis vector fields (local tangent directions)

This is the continuous-geometry analogue of discrete MoE routing used in
architectures like Mixtral / Switch Transformer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertBasisField(nn.Module):
    """
    A single expert that defines a local tangent vector field phi_k(x, t).

    Each expert specialises in modelling velocity in a specific region
    of the state-time manifold.
    """

    def __init__(self, state_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, x: torch.Tensor, t_vec: torch.Tensor) -> torch.Tensor:
        """
        x     : (B, D) state
        t_vec : (B, 1) time

        Returns : (B, D) tangent vector
        """
        return self.net(torch.cat([x, t_vec], dim=-1))


class MoEVectorField(nn.Module):
    """
    Mixture-of-Expert vector field.

    The router produces soft weights alpha_k(x, t) and the output is
    the weighted combination of K expert tangent vectors.

    Parameters
    ----------
    state_dim : int
    hidden_dim : int
    num_experts : int
    top_k : int
        Number of experts activated per sample (sparse routing).
        Set to num_experts for dense routing.
    aux_loss_weight : float
        Weight for the load-balancing auxiliary loss.
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 128,
        num_experts: int = 4,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.aux_loss_weight = aux_loss_weight

        # Router: (x, t) → K logits
        self.router = nn.Sequential(
            nn.Linear(state_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_experts),
        )

        # K expert basis fields
        self.experts = nn.ModuleList([
            ExpertBasisField(state_dim, hidden_dim)
            for _ in range(num_experts)
        ])

        # Auxiliary loss state (populated during forward)
        self._aux_loss: torch.Tensor | None = None

    @property
    def aux_loss(self) -> torch.Tensor:
        """Load-balancing loss, to be added to the main loss during training."""
        if self._aux_loss is None:
            return torch.tensor(0.0)
        return self._aux_loss * self.aux_loss_weight

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Matches the torchdiffeq signature: forward(t, state).

        t : scalar or (B,) time
        x : (B, D) state

        Returns : (B, D) velocity
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        B = x.shape[0]

        # Build time vector
        if t.dim() == 0:
            t_vec = t.unsqueeze(0).expand(B).unsqueeze(-1)
        else:
            t_vec = t.reshape(-1, 1)

        # Router logits and top-k selection
        logits = self.router(torch.cat([x, t_vec], dim=-1))   # (B, K)

        if self.top_k < self.num_experts:
            # Sparse routing: keep only top-k experts
            topk_vals, topk_idx = torch.topk(logits, self.top_k, dim=-1)
            mask = torch.zeros_like(logits).scatter_(-1, topk_idx, 1.0)
            
            # Save raw logits for aux loss (Fix DESIGN-3)
            raw_logits = logits.detach()
            
            logits = logits.masked_fill(mask == 0, float("-inf"))
        else:
            raw_logits = logits.detach()

        alpha = F.softmax(logits, dim=-1)                      # (B, K)

        # Compute load-balancing auxiliary loss using RAW logits (unmasked)
        f = alpha.mean(dim=0)                                  # (K,)
        P = F.softmax(raw_logits, dim=-1).mean(dim=0)          # (K,)
        self._aux_loss = self.num_experts * (f * P).sum()

        # Compute weighted sum of expert outputs
        expert_outputs = torch.stack(
            [expert(x, t_vec) for expert in self.experts], dim=1
        )  # (B, K, D)

        # Weighted combination: v = sum_k alpha_k * phi_k
        velocity = torch.einsum("bk,bkd->bd", alpha, expert_outputs)  # (B, D)

        return velocity
