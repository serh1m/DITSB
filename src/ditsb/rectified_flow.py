"""
Rectified Flow — One-Step Generation via Straight-Line Distillation

The core insight: if the OT flow matching loss perfectly converges, the
learned velocity field is CONSTANT along each trajectory (straight line).
This means a single Euler step suffices for exact generation:

    x_1 = x_0 + 1.0 * v_theta(x_0, t=0)

In practice, the learned field is not perfectly straight.  Reflow
(self-distillation) iteratively straightens the trajectories:

    1. Train teacher with standard OT flow matching
    2. Generate (x_0, x_1) pairs by integrating the teacher
    3. Retrain student to predict the straight line between (x_0, x_1)
    4. Repeat for straighter paths

After 1-2 reflow iterations, one-step Euler is competitive with
multi-step ODE solving.
"""

import torch
import torch.nn as nn
from typing import Optional

from .flow import DITSB_GenerativeFlow
from .loss import optimal_transport_loss


def compute_straightness(
    vector_field: nn.Module,
    z_0: torch.Tensor,
    z_1: torch.Tensor,
    n_eval: int = 20,
) -> float:
    """
    Measure how straight the learned trajectories are.

    Straightness = 1 means perfect straight lines (one-step exact).
    Lower values mean curved paths (need more steps).

    Metric: cosine similarity between v(z_t, t) and (z_1 - z_0)
    averaged over time and samples.
    """
    B = z_0.shape[0]
    device = z_0.device
    target = z_1 - z_0  # ideal constant velocity

    cos_sims = []
    for i in range(n_eval):
        t_val = i / (n_eval - 1)
        t = torch.full((B,), t_val, device=device)
        z_t = (1 - t_val) * z_0 + t_val * z_1

        with torch.no_grad():
            v = vector_field(t, z_t)

        cos = nn.functional.cosine_similarity(v, target, dim=-1).mean()
        cos_sims.append(cos.item())

    return sum(cos_sims) / len(cos_sims)


def generate_reflow_pairs(
    flow_model: DITSB_GenerativeFlow,
    num_pairs: int,
    state_dim: int,
    device: torch.device,
    steps: int = 100,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate (z_0, z_1) coupling pairs by running the teacher model.

    z_0 ~ N(0, I)
    z_1 = ODE_solve(z_0, t: 0→1)

    These paired samples define straighter target trajectories for
    the student (reflow) model.
    """
    flow_model.eval()
    with torch.no_grad():
        z_0 = torch.randn(num_pairs, state_dim, device=device)
        t_span = torch.linspace(0, 1, steps=steps, device=device)
        trajectory = flow_model(z_0, t_span)
        z_1 = trajectory[-1]
    return z_0, z_1


def reflow_loss(
    vector_field: nn.Module,
    z_0: torch.Tensor,
    z_1: torch.Tensor,
) -> torch.Tensor:
    """
    Reflow (distillation) loss: train the student to predict the
    straight-line velocity between pre-computed (z_0, z_1) pairs.

    This is identical to the OT loss but with COUPLED pairs instead
    of independently sampled noise and data.
    """
    B = z_0.shape[0]
    device = z_0.device

    t = torch.rand(B, device=device)
    t_expand = t.unsqueeze(-1)

    z_t = (1.0 - t_expand) * z_0 + t_expand * z_1
    target = z_1 - z_0  # straight-line velocity

    predicted = vector_field(t, z_t)
    return torch.mean((predicted - target) ** 2)


@torch.no_grad()
def one_step_generate(
    vector_field: nn.Module,
    num_samples: int,
    state_dim: int,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """
    THE ultimate inference: one forward pass, noise → data.

        x_1 = x_0 + v_theta(x_0, t=0)

    This is exact when trajectories are perfectly straight.
    """
    vector_field.eval()
    z_0 = torch.randn(num_samples, state_dim, device=device)
    t = torch.zeros(num_samples, device=device)
    v = vector_field(t, z_0)
    return z_0 + v


class RectifiedFlowTrainer:
    """
    End-to-end rectified flow training pipeline.

    Usage
    -----
    1. Train a teacher with standard OT flow matching.
    2. Create trainer: `rf = RectifiedFlowTrainer(teacher_vf, student_vf, ...)`
    3. Run reflow: `rf.reflow(num_iterations=2, ...)`
    4. One-step generate: `samples = one_step_generate(student_vf, ...)`
    """

    def __init__(
        self,
        teacher_vf: nn.Module,
        student_vf: nn.Module,
        teacher_flow: DITSB_GenerativeFlow,
        state_dim: int,
        device: torch.device,
    ):
        self.teacher_vf = teacher_vf
        self.student_vf = student_vf
        self.teacher_flow = teacher_flow
        self.state_dim = state_dim
        self.device = device

    def reflow(
        self,
        num_iterations: int = 1,
        num_pairs: int = 10_000,
        train_steps: int = 2000,
        batch_size: int = 256,
        lr: float = 1e-3,
        teacher_ode_steps: int = 100,
        verbose: bool = True,
        early_stop_straightness_drop: float = 0.1,
    ) -> list[float]:
        """
        Run reflow distillation iterations.

        Returns list of final losses per iteration.
        """
        import copy
        all_losses = []

        for iteration in range(num_iterations):
            if verbose:
                print(f"[Reflow] Iteration {iteration + 1}/{num_iterations}")

            # Generate coupling pairs from current teacher
            if verbose:
                print(f"  Generating {num_pairs} coupling pairs ...")
            z_0_all, z_1_all = generate_reflow_pairs(
                self.teacher_flow, num_pairs, self.state_dim,
                self.device, steps=teacher_ode_steps,
            )

            # Measure straightness before reflow
            idx = torch.randperm(num_pairs)[:min(500, num_pairs)]
            straightness = compute_straightness(
                self.teacher_vf, z_0_all[idx], z_1_all[idx]
            )
            if verbose:
                print(f"  Straightness (before): {straightness:.4f}")

            # Train student on coupled pairs
            optimiser = torch.optim.AdamW(
                self.student_vf.parameters(), lr=lr, weight_decay=1e-5
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimiser, T_max=train_steps
            )

            final_loss = 0.0
            for step in range(train_steps):
                # Fix BUG-6: Rename loop variable to avoid shadowing 'idx'
                batch_idx = torch.randint(0, num_pairs, (batch_size,))
                z_0_batch = z_0_all[batch_idx]
                z_1_batch = z_1_all[batch_idx]

                loss = reflow_loss(self.student_vf, z_0_batch, z_1_batch)
                optimiser.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.student_vf.parameters(), 1.0)
                optimiser.step()
                scheduler.step()
                final_loss = loss.item()

                if verbose and (step + 1) % 500 == 0:
                    print(f"    step {step+1}/{train_steps}  loss={final_loss:.6f}")

            all_losses.append(final_loss)

            # For next iteration, promote student to teacher
            self.teacher_vf.load_state_dict(self.student_vf.state_dict())

            # Measure straightness after reflow
            # Uses the ORIGINAL 'idx' permutation (not the last batch_idx)
            straightness_after = compute_straightness(
                self.student_vf, z_0_all[idx[:500]], z_1_all[idx[:500]]
            )
            if verbose:
                print(f"  Straightness (after):  {straightness_after:.4f}")

            # Early stopping check: if straightness significantly degrades, abort reflow
            if straightness_after < straightness - early_stop_straightness_drop:
                if verbose:
                    print(f"  [Warning] Straightness degraded significantly ({straightness:.4f} -> {straightness_after:.4f}). Early stopping reflow!")
                # Revert teacher promotion since this student was bad
                self.teacher_vf.load_state_dict(copy.deepcopy(self.teacher_flow.vector_field.state_dict()))
                break

        return all_losses
