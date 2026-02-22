#!/usr/bin/env python
"""
scaling_law.py — DITSB Scaling Law Sandbox

Systematically sweep model sizes and measure converged loss
to empirically estimate the DITSB scaling exponent:

    L(N) ∝ N^{-alpha}

Runs on 2D synthetic data for fast iteration.

Usage
-----
    python scaling_law.py --dataset moons --sizes 4 --steps 3000
"""

import argparse
import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from ditsb import ContinuousVectorField, optimal_transport_loss
from ditsb.moe_vector_field import MoEVectorField
from ditsb.fno_flow import FNOContinuousVectorField
from train import load_dataset, infinite_data_loader


# Model factories for different architectures
def make_mlp(state_dim: int, hidden_dim: int, **kw) -> torch.nn.Module:
    return ContinuousVectorField(state_dim, hidden_dim)

def make_moe(state_dim: int, hidden_dim: int, num_experts: int = 4, **kw) -> torch.nn.Module:
    return MoEVectorField(state_dim, hidden_dim, num_experts=num_experts, top_k=2)

def make_fno(state_dim: int, hidden_dim: int, **kw) -> torch.nn.Module:
    return FNOContinuousVectorField(state_dim, hidden_dim, modes=max(4, hidden_dim // 8), n_layers=3)


ARCHITECTURES = {
    "mlp": make_mlp,
    "moe": make_moe,
    "fno": make_fno,
}


def train_and_measure(
    arch_fn,
    state_dim: int,
    hidden_dim: int,
    data: np.ndarray,
    device: torch.device,
    train_steps: int = 3000,
    batch_size: int = 512,
    lr: float = 1e-3,
) -> dict:
    """Train a model and return final loss + parameter count."""
    vf = arch_fn(state_dim=state_dim, hidden_dim=hidden_dim).to(device)
    num_params = sum(p.numel() for p in vf.parameters())

    loader = infinite_data_loader(data, batch_size, device)
    optimiser = torch.optim.AdamW(vf.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=train_steps)

    # Train
    losses = []
    t0 = time.time()
    for step in range(train_steps):
        batch = next(loader)
        loss = optimal_transport_loss(vf, batch)

        # For MoE, add auxiliary loss
        if hasattr(vf, 'aux_loss'):
            loss = loss + vf.aux_loss

        optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(vf.parameters(), 1.0)
        optimiser.step()
        scheduler.step()
        losses.append(loss.item())

    elapsed = time.time() - t0
    # Use average of last 10% as converged loss
    tail = max(1, len(losses) // 10)
    converged_loss = np.mean(losses[-tail:])

    return {
        "num_params": num_params,
        "hidden_dim": hidden_dim,
        "converged_loss": float(converged_loss),
        "final_loss": float(losses[-1]),
        "elapsed_s": elapsed,
    }


def fit_power_law(params, losses):
    """Fit L(N) = a * N^(-alpha) + c via log-space linear regression."""
    log_n = np.log(params)
    log_l = np.log(losses)
    # Simple linear fit: log(L) = log(a) - alpha * log(N)
    coeffs = np.polyfit(log_n, log_l, 1)
    alpha = -coeffs[0]
    a = np.exp(coeffs[1])
    return alpha, a


def main():
    parser = argparse.ArgumentParser(description="DITSB Scaling Law Sandbox")
    parser.add_argument("--dataset", type=str, default="moons")
    parser.add_argument("--arch", type=str, default="mlp", choices=list(ARCHITECTURES.keys()))
    parser.add_argument("--sizes", type=int, default=6,
                        help="Number of model sizes to sweep")
    parser.add_argument("--steps", type=int, default=3000,
                        help="Training steps per size")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = (torch.device("cuda") if args.device == "auto" and torch.cuda.is_available()
              else torch.device("cpu"))

    data = load_dataset(args.dataset, n_samples=20_000)
    state_dim = data.shape[1]

    # Geometric sweep of hidden dimensions
    hidden_dims = [int(16 * (2 ** (i * 0.8))) for i in range(args.sizes)]
    # Remove duplicates and sort
    hidden_dims = sorted(set(hidden_dims))

    arch_fn = ARCHITECTURES[args.arch]
    print(f"[scaling] Architecture: {args.arch}")
    print(f"[scaling] Dataset: {args.dataset}")
    print(f"[scaling] Hidden dims: {hidden_dims}")
    print(f"[scaling] Steps per size: {args.steps}")
    print()

    results = []
    for hd in hidden_dims:
        print(f"  Training hidden_dim={hd} ...", end="", flush=True)
        r = train_and_measure(
            arch_fn, state_dim, hd, data, device,
            train_steps=args.steps, batch_size=args.batch_size,
        )
        results.append(r)
        print(f"  params={r['num_params']:>8,}  loss={r['converged_loss']:.6f}  ({r['elapsed_s']:.1f}s)")

    # Fit power law
    params_arr = np.array([r["num_params"] for r in results])
    losses_arr = np.array([r["converged_loss"] for r in results])
    alpha, a = fit_power_law(params_arr, losses_arr)
    print(f"\n[scaling] Fitted: L(N) = {a:.4f} * N^(-{alpha:.4f})")
    print(f"[scaling] Scaling exponent alpha = {alpha:.4f}")
    print(f"[scaling] (Chinchilla Transformer reference: alpha ≈ 0.076)")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, f"scaling_{args.arch}_{args.dataset}.json")
    with open(results_path, "w") as f:
        json.dump({"results": results, "alpha": alpha, "a": a}, f, indent=2)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(params_arr, losses_arr, "o-", color="#2196F3", markersize=8, label="Measured")

    # Fitted line
    n_fit = np.logspace(np.log10(params_arr.min()), np.log10(params_arr.max()), 50)
    l_fit = a * n_fit ** (-alpha)
    ax.loglog(n_fit, l_fit, "--", color="#FF5722", alpha=0.7,
              label=f"Fit: L = {a:.2f} N^(-{alpha:.4f})")

    ax.set_xlabel("Parameters (N)", fontsize=12)
    ax.set_ylabel("Converged Loss", fontsize=12)
    ax.set_title(f"DITSB Scaling Law — {args.arch} on {args.dataset}", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plot_path = os.path.join(args.output_dir, f"scaling_{args.arch}_{args.dataset}.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[scaling] Plot saved: {plot_path}")
    print(f"[scaling] Results saved: {results_path}")


if __name__ == "__main__":
    main()
