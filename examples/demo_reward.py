#!/usr/bin/env python
"""
demo_reward.py — Reward-Guided Flow Demonstration

1. Train a base flow on 2D gaussian8 dataset.
2. Define an analytical reward: prefer the upper-right quadrant.
3. Generate samples with vs without reward guidance.
4. Visualise how guidance bends ODE trajectories.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from ditsb import (
    ContinuousVectorField,
    DITSB_GenerativeFlow,
    optimal_transport_loss,
    generate_samples,
)
from ditsb.reward_flow import RewardGuidedFlow
from ditsb.hjb_loss import hjb_terminal_cost_loss
from train import load_dataset, infinite_data_loader


class AnalyticalReward(nn.Module):
    """Reward function that prefers the upper-right quadrant."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Higher reward for x1 > 0 AND x2 > 0
        return x[:, 0] + x[:, 1]


def main():
    parser = argparse.ArgumentParser(description="Reward-guided flow demo")
    parser.add_argument("--dataset", type=str, default="gaussian8")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps_per_epoch", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--guidance_scale", type=float, default=2.0)
    parser.add_argument("--reward_weight", type=float, default=0.05)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = (torch.device("cuda") if args.device == "auto" and torch.cuda.is_available()
              else torch.device("cpu"))
    print(f"[reward_demo] Device: {device}")

    # Data
    data = load_dataset(args.dataset, n_samples=20_000)
    state_dim = data.shape[1]
    loader = infinite_data_loader(data, args.batch_size, device)
    print(f"[reward_demo] Dataset: {args.dataset}")

    # Model
    vf = ContinuousVectorField(state_dim=state_dim, hidden_dim=128).to(device)
    flow = DITSB_GenerativeFlow(vf, solver="euler").to(device)
    reward_fn = AnalyticalReward().to(device)

    optimiser = torch.optim.AdamW(vf.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=args.epochs * args.steps_per_epoch
    )

    # --- Phase 1: Train with standard OT loss --- #
    print(f"[reward_demo] Phase 1: Standard OT training ({args.epochs // 2} epochs)...")
    half_epochs = args.epochs // 2
    for epoch in range(1, half_epochs + 1):
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
        if epoch == 1 or epoch % 20 == 0:
            print(f"  epoch {epoch}/{half_epochs}  |  OT loss {epoch_loss / args.steps_per_epoch:.4f}")

    # --- Phase 2: Train with HJB terminal cost --- #
    print(f"[reward_demo] Phase 2: HJB reward-augmented training ({half_epochs} epochs)...")
    for epoch in range(1, half_epochs + 1):
        epoch_info = {"ot_loss": 0.0, "reward_term": 0.0}
        for _ in range(args.steps_per_epoch):
            batch = next(loader)
            loss, info = hjb_terminal_cost_loss(
                vf, batch, reward_fn, reward_weight=args.reward_weight
            )
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vf.parameters(), max_norm=1.0)
            optimiser.step()
            scheduler.step()
            for k in epoch_info:
                epoch_info[k] += info[k]
        if epoch == 1 or epoch % 20 == 0:
            n = args.steps_per_epoch
            print(f"  epoch {epoch}/{half_epochs}  |"
                  f"  OT {epoch_info['ot_loss']/n:.4f}"
                  f"  |  R {epoch_info['reward_term']/n:.4f}")

    # --- Generation --- #
    print("[reward_demo] Generating samples...")

    # Unguided generation
    unguided = generate_samples(flow, 2000, state_dim, device, steps=100)
    unguided_np = unguided.cpu().numpy()

    # Reward-guided generation
    guided_flow = RewardGuidedFlow(flow, reward_fn, guidance_scale=args.guidance_scale)
    guided = guided_flow.generate(2000, state_dim, device, steps=100)
    guided_np = guided.cpu().numpy()

    # --- Visualise --- #
    os.makedirs(args.output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].scatter(data[:2000, 0], data[:2000, 1], s=2, alpha=0.5, c="#2196F3")
    axes[0].set_title("Real Data", fontsize=13)
    axes[0].set_xlim(-3.5, 3.5); axes[0].set_ylim(-3.5, 3.5)
    axes[0].set_aspect("equal"); axes[0].grid(True, alpha=0.3)

    axes[1].scatter(unguided_np[:, 0], unguided_np[:, 1], s=2, alpha=0.5, c="#FF9800")
    axes[1].set_title("Unguided Generation", fontsize=13)
    axes[1].set_xlim(-3.5, 3.5); axes[1].set_ylim(-3.5, 3.5)
    axes[1].set_aspect("equal"); axes[1].grid(True, alpha=0.3)

    axes[2].scatter(guided_np[:, 0], guided_np[:, 1], s=2, alpha=0.5, c="#4CAF50")
    axes[2].set_title(f"Reward-Guided (scale={args.guidance_scale})", fontsize=13)
    axes[2].set_xlim(-3.5, 3.5); axes[2].set_ylim(-3.5, 3.5)
    axes[2].set_aspect("equal"); axes[2].grid(True, alpha=0.3)

    # Draw reward gradient direction
    for ax in axes:
        ax.annotate("", xy=(2.5, 2.5), xytext=(1.5, 1.5),
                     arrowprops=dict(arrowstyle="->", color="red", lw=2))
        ax.text(2.6, 2.6, "R(x)", color="red", fontsize=10, fontweight="bold")

    fig.suptitle("HJB Terminal Cost: Reward-Guided Flow vs Unguided", fontsize=15, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(args.output_dir, f"reward_guidance_{args.dataset}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[reward_demo] Saved: {path}")
    print("[reward_demo] Done.")


if __name__ == "__main__":
    main()
