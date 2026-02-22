"""
Symplectic Integrators — Structure-Preserving Numerical Integration

Standard ODE solvers (Euler, RK4) are dissipative: they introduce artificial
energy drift over long integration horizons.  Symplectic integrators preserve
the symplectic 2-form, guaranteeing that energy errors remain bounded and
oscillatory (never exponentially divergent).

Three integrators are provided:
    - symplectic_euler   : 1st order, simplest
    - leapfrog           : 2nd order (Stormer-Verlet), recommended default
    - yoshida4           : 4th order (Yoshida composition), highest accuracy

All operate on (q, p) phase-space pairs from a Hamiltonian system.
Updated to support time-dependent potentials V(q, t) and non-separable H(q, p, t).
"""

import torch
import torch.nn as nn
from typing import Callable


def symplectic_euler_step(
    q: torch.Tensor,
    p: torch.Tensor,
    t: float,
    dt: float,
    grad_V: Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor],
    grad_T: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    First-order symplectic Euler.
    """
    if grad_T is None:
        grad_T = lambda p: p
    
    # p_{n+1} = p_n - dt * dV/dq(q_n, t_n)
    p_new = p - dt * grad_V(q, p, t)
    
    # q_{n+1} = q_n + dt * dT/dp(p_{n+1})
    q_new = q + dt * grad_T(p_new)
    
    return q_new, p_new


def leapfrog_step(
    q: torch.Tensor,
    p: torch.Tensor,
    t: float,
    dt: float,
    grad_V: Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor],
    grad_T: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Second-order Leapfrog / Stormer-Verlet integrator.
    
    Kick-Drift-Kick decomposition:
        p_{n+1/2} = p_n       - (dt/2) * dV/dq(q_n, p_n, t_n)
        q_{n+1}   = q_n       + dt     * dT/dp(p_{n+1/2})
        p_{n+1}   = p_{n+1/2} - (dt/2) * dV/dq(q_{n+1}, p_{n+1/2}, t_{n+1})
    """
    if grad_T is None:
        grad_T = lambda p: p
        
    t_half = t + 0.5 * dt
    t_next = t + dt
    
    # Half kick
    p_half = p - 0.5 * dt * grad_V(q, p, t)
    
    # Full drift
    q_new = q + dt * grad_T(p_half)
    
    # Half kick (at new position and time)
    p_new = p_half - 0.5 * dt * grad_V(q_new, p_half, t_next)
    
    return q_new, p_new


def yoshida4_step(
    q: torch.Tensor,
    p: torch.Tensor,
    t: float,
    dt: float,
    grad_V: Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor],
    grad_T: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fourth-order Yoshida symplectic integrator.
    Composed of three leapfrog sub-steps.
    """
    # Yoshida coefficients
    w1 = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
    w0 = -2.0 ** (1.0 / 3.0) * w1

    # Update state and time sequentially
    dq1, dp1 = w1 * dt, w1 * dt
    dt1 = w1 * dt
    q, p = leapfrog_step(q, p, t, dt1, grad_V, grad_T)
    
    t += dt1
    dt0 = w0 * dt
    q, p = leapfrog_step(q, p, t, dt0, grad_V, grad_T)
    
    t += dt0
    dt_last = w1 * dt
    q, p = leapfrog_step(q, p, t, dt_last, grad_V, grad_T)
    
    return q, p


_INTEGRATORS = {
    "symplectic_euler": symplectic_euler_step,
    "leapfrog": leapfrog_step,
    "yoshida4": yoshida4_step,
}


def symplectic_integrate(
    grad_V: Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor],
    q_init: torch.Tensor,
    p_init: torch.Tensor,
    t_span: torch.Tensor,
    method: str = "leapfrog",
    grad_T: Callable[[torch.Tensor], torch.Tensor] | None = None,
    resonance_threshold: float = 1e4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Integrate Hamiltonian system.
    
    grad_V signature must be: (q, p, t) -> tensor
    """
    step_fn = _INTEGRATORS[method]
    T = t_span.shape[0]
    q, p = q_init.clone(), p_init.clone()

    q_traj = [q_init]
    p_traj = [p_init]
    
    warning_triggered = False

    for i in range(T - 1):
        t_curr = t_span[i].item()
        dt = (t_span[i + 1] - t_span[i]).item()
        
        q, p = step_fn(q, p, t_curr, dt, grad_V, grad_T)
        
        # Stability check: detect symplectic resonance / explosion
        if not warning_triggered and (torch.isnan(q).any() or q.abs().max() > resonance_threshold):
            print(f"[Warning] Symplectic resonance or explosion detected at step {i} (t={t_curr:.3f}). Consider decreasing dt.")
            warning_triggered = True
            
        q_traj.append(q.clone())
        p_traj.append(p.clone())

    return torch.stack(q_traj), torch.stack(p_traj)
