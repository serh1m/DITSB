#!/usr/bin/env python
"""
demo.py — End-to-end DITSB demonstration

1. Train a flow-matching model on a 2-D synthetic dataset.
2. Generate samples by integrating the learned ODE.
3. Visualise real vs generated distributions and (optionally) ODE trajectories.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from ditsb import (
    ContinuousVectorField,
    DeepVectorField,
    DITSB_GenerativeFlow,
    optimal_transport_loss,
    generate_samples,
)

# Reuse data utilities from train.py
from train import load_dataset, infinite_data_loader


def plot_comparison(real: np.ndarray, generated: np.ndarray, title: str, path: str):
    """Side-by-side scatter plots of real vs generated data."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(real[:, 0], real[:, 1], s=2, alpha=0.5, c="#2196F3")
    axes[0].set_title("Real Data (μ₁)", fontsize=13)
    axes[0].set_xlim(-3.5, 3.5)
    axes[0].set_ylim(-3.5, 3.5)
    axes[0].set_aspect("equal")
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(generated[:, 0], generated[:, 1], s=2, alpha=0.5, c="#FF5722")
    axes[1].set_title("Generated Data (DITSB Flow)", fontsize=13)
    axes[1].set_xlim(-3.5, 3.5)
    axes[1].set_ylim(-3.5, 3.5)
    axes[1].set_aspect("equal")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[demo] Saved comparison plot → {path}")


def plot_trajectories(trajectory: np.ndarray, real: np.ndarray, path: str, n_traces: int = 200):
    """
    Plot ODE trajectories: how samples flow from noise (t=0) to data (t=1).

    trajectory : (T, B, 2)
    """
    fig, ax = plt.subplots(figsize=(7, 7))

    # Background: real data in light grey
    ax.scatter(real[:, 0], real[:, 1], s=1, alpha=0.15, c="#9E9E9E", label="Real data")

    # Draw a subset of trajectories
    T, B, D = trajectory.shape
    indices = np.random.choice(B, size=min(n_traces, B), replace=False)
    cmap = plt.cm.plasma
    for idx in indices:
        traj = trajectory[:, idx, :]   # (T, 2)
        # Color each segment by time
        for s in range(T - 1):
            ax.plot(
                traj[s:s+2, 0], traj[s:s+2, 1],
                color=cmap(s / T), alpha=0.4, linewidth=0.5,
            )
    # Mark start and end
    ax.scatter(trajectory[0, indices, 0], trajectory[0, indices, 1],
               s=8, c="#1565C0", zorder=5, label="t=0 (noise)")
    ax.scatter(trajectory[-1, indices, 0], trajectory[-1, indices, 1],
               s=8, c="#C62828", zorder=5, label="t=1 (data)")

    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)
    ax.set_aspect("equal")
    ax.legend(fontsize=9)
    ax.set_title("ODE Trajectories: Noise → Data", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[demo] Saved trajectory plot → {path}")


def main():
    parser = argparse.ArgumentParser(description="DITSB demo")
    parser.add_argument("--dataset", type=str, default="moons",
                        choices=["moons", "circles", "swissroll", "gaussian8", "pinwheel"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps_per_epoch", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--solver", type=str, default="euler")
    parser.add_argument("--model", type=str, default="simple", choices=["simple", "deep"])
    parser.add_argument("--gen_steps", type=int, default=100,
                        help="Number of ODE integration steps during generation")
    parser.add_argument("--num_gen", type=int, default=2000,
                        help="Number of samples to generate")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = (torch.device("cuda") if args.device == "auto" and torch.cuda.is_available()
              else torch.device("cpu"))
    print(f"[demo] Device: {device}")

    # ---- Data ------------------------------------------------------------ #
    data = load_dataset(args.dataset, n_samples=20_000)
    state_dim = data.shape[1]
    loader = infinite_data_loader(data, args.batch_size, device)
    print(f"[demo] Dataset: {args.dataset}  |  dim: {state_dim}")

    # ---- Model ----------------------------------------------------------- #
    if args.model == "deep":
        vf = DeepVectorField(state_dim=state_dim, hidden_dim=args.hidden_dim).to(device)
    else:
        vf = ContinuousVectorField(state_dim=state_dim, hidden_dim=args.hidden_dim).to(device)
    flow = DITSB_GenerativeFlow(vf, solver=args.solver).to(device)

    optimiser = torch.optim.AdamW(vf.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=args.epochs * args.steps_per_epoch
    )

    # ---- Train ----------------------------------------------------------- #
    print(f"[demo] Training for {args.epochs} epochs …")
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        for _ in range(args.steps_per_epoch):
            batch = next(loader)
            loss = optimal_transport_loss(vf, batch)
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vf.parameters(), max_norm=1.0)
            optimiser.step()
            scheduler.step()
            epoch_loss += loss.item()
        avg = epoch_loss / args.steps_per_epoch
        if epoch == 1 or epoch % 20 == 0 or epoch == args.epochs:
            print(f"  epoch {epoch:>4d}/{args.epochs}  |  loss {avg:.6f}")

    # ---- Generate -------------------------------------------------------- #
    print(f"[demo] Generating {args.num_gen} samples with {args.gen_steps} ODE steps …")
    trajectory = generate_samples(
        flow, args.num_gen, state_dim, device,
        steps=args.gen_steps, return_trajectory=True,
    )
    generated = trajectory[-1].cpu().numpy()
    traj_np = trajectory.cpu().numpy()

    # ---- Visualise ------------------------------------------------------- #
    os.makedirs(args.output_dir, exist_ok=True)

    plot_comparison(
        data[:2000], generated,
        title=f"DITSB · {args.dataset} · {args.epochs} epochs",
        path=os.path.join(args.output_dir, f"comparison_{args.dataset}.png"),
    )
    plot_trajectories(
        traj_np, data[:2000],
        path=os.path.join(args.output_dir, f"trajectories_{args.dataset}.png"),
    )

    print("[demo] Done.")


if __name__ == "__main__":
    main()
