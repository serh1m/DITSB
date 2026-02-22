"""
Manifold Dynamics System — DITSB Generative Flow

Wraps a vector field and uses `torchdiffeq.odeint_adjoint` to integrate the
ODE  dZ/dt = v_θ(Z_t, t)  over the time interval [0, 1].

The adjoint method back-propagates through the ODE *without* storing
intermediate activations, achieving O(1) memory cost regardless of the number
of integration steps.
"""

import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint


class DITSB_GenerativeFlow(nn.Module):
    """
    Parameters
    ----------
    vector_field : nn.Module
        Must satisfy the torchdiffeq signature ``forward(t, z) -> dz/dt``.
    solver : str
        ODE solver name: ``'euler'``, ``'rk4'``, ``'dopri5'`` (adaptive), etc.
    atol, rtol : float
        Absolute / relative tolerance for adaptive solvers.
    """

    def __init__(
        self,
        vector_field: nn.Module,
        solver: str = "euler",
        atol: float = 1e-5,
        rtol: float = 1e-5,
    ):
        super().__init__()
        self.vector_field = vector_field
        self.solver = solver
        self.atol = atol
        self.rtol = rtol

    def forward(
        self,
        z_init: torch.Tensor,
        t_span: torch.Tensor,
    ) -> torch.Tensor:
        """
        Integrate the ODE from z_init over the time points in t_span.

        Parameters
        ----------
        z_init : (B, D) initial state (e.g. Gaussian noise for generation).
        t_span : (T,)  monotonically increasing time grid.

        Returns
        -------
        trajectory : (T, B, D) — state at each requested time point.
        """
        trajectory = odeint(
            self.vector_field,
            z_init,
            t_span,
            method=self.solver,
            atol=self.atol,
            rtol=self.rtol,
        )
        return trajectory
