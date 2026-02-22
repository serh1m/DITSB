"""
Hamiltonian Generative Flow — Phase-Space ODE with Symplectic Integration

Instead of the standard first-order ODE  dz/dt = v(z,t), we lift the state
to the cotangent bundle (phase space) and learn a Hamiltonian H(q, p, t).

The dynamics are derived via Hamilton's equations:
    dq/dt =  dH/dp     (position evolves with kinetic gradient)
    dp/dt = -dH/dq     (momentum evolves with potential gradient)

This guarantees volume-preservation (Liouville's theorem) and, with symplectic
integrators, bounded energy error even over very long sequences.
"""

import torch
import torch.nn as nn
from .symplectic import symplectic_integrate


class HamiltonianVectorField(nn.Module):
    """
    Neural Hamiltonian: learns H(q, p, t) and derives the vector field
    via automatic differentiation of Hamilton's equations.

    Parameters
    ----------
    state_dim : int
        Dimension of q (and p).
    hidden_dim : int
        MLP hidden width.
    """

    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        # Input: concat(q, p, t) → scalar H
        self.net = nn.Sequential(
            nn.Linear(2 * state_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def hamiltonian(self, q: torch.Tensor, p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute scalar Hamiltonian H(q, p, t).  Returns (B, 1)."""
        if isinstance(t, float):
            t = torch.tensor(t, device=q.device, dtype=q.dtype)
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(q.shape[0])
        t_vec = t.unsqueeze(-1)
        inp = torch.cat([q, p, t_vec], dim=-1)
        return self.net(inp)

    def forward(self, q: torch.Tensor, p: torch.Tensor, t: torch.Tensor):
        """
        Compute Hamilton's equations:
            dq/dt =  dH/dp
            dp/dt = -dH/dq

        Returns (dq_dt, dp_dt), each (B, D).
        """
        q = q.detach().requires_grad_(True)
        p = p.detach().requires_grad_(True)
        H = self.hamiltonian(q, p, t).sum()
        dH_dq, dH_dp = torch.autograd.grad(H, [q, p], create_graph=True)
        return dH_dp, -dH_dq   # (dq/dt, dp/dt)


class HamiltonianGenerativeFlow(nn.Module):
    """
    Generative flow on phase space (q, p) using symplectic integration.

    Parameters
    ----------
    hamiltonian_field : HamiltonianVectorField
        The neural Hamiltonian.
    method : str
        Symplectic integrator: 'leapfrog', 'symplectic_euler', 'yoshida4'.
    """

    def __init__(self, hamiltonian_field: HamiltonianVectorField, method: str = "leapfrog"):
        super().__init__()
        self.hfield = hamiltonian_field
        self.method = method

    def forward(
        self,
        q_init: torch.Tensor,
        p_init: torch.Tensor,
        t_span: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Integrate Hamilton's equations from (q_init, p_init) over t_span.

        Parameters
        ----------
        q_init : (B, D)
        p_init : (B, D)
        t_span : (T,) time grid

        Returns
        -------
        q_traj, p_traj : each (T, B, D)
        """
        
        def grad_V(q, p, t):
            """
            grad_q V(q, p, t) = -dp/dt direction from Hamilton's equations.
            Calculates dH/dq at the CURRENT state (q, p) and time t.
            """
            q_ = q.detach().requires_grad_(True)
            # Use actual Momentum p in calculation (Fix BUG-1)
            # No need to track gradients w.r.t p inside grad_V, just pass value
            p_ = p.detach() 
            
            # Use actual Time t (Fix BUG-2)
            t_tensor = torch.tensor(t, device=q.device, dtype=q.dtype)
            
            H = self.hfield.hamiltonian(q_, p_, t_tensor).sum()
            return torch.autograd.grad(H, q_, create_graph=False)[0]

        q_traj, p_traj = symplectic_integrate(
            grad_V, q_init, p_init, t_span, method=self.method
        )
        return q_traj, p_traj
