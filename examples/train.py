#!/usr/bin/env python
"""
train.py — DITSB Training Script

Train a continuous-time generative flow on synthetic 2-D datasets using
Optimal Transport Flow Matching with the adjoint method.

Usage
-----
    python train.py --dataset moons --epochs 200 --lr 1e-3 --solver euler
"""

import argparse
import os
import time

import numpy as np
import torch
from sklearn.datasets import make_moons, make_circles, make_swiss_roll

from ditsb import (
    ContinuousVectorField,
    DeepVectorField,
    DITSB_GenerativeFlow,
    optimal_transport_loss,
    generate_samples,
)


# ------------------------------------------------------------------ #
#  Synthetic data loaders
# ------------------------------------------------------------------ #

def load_dataset(name: str, n_samples: int = 10_000) -> np.ndarray:
    """Return (N, 2) numpy array of 2-D points."""
    if name == "moons":
        data, _ = make_moons(n_samples=n_samples, noise=0.06)
    elif name == "circles":
        data, _ = make_circles(n_samples=n_samples, noise=0.04, factor=0.5)
    elif name == "swissroll":
        data, _ = make_swiss_roll(n_samples=n_samples, noise=0.3)
        data = data[:, [0, 2]] / 10.0  # project to 2-D & normalise
    elif name == "gaussian8":
        # 8 Gaussians arranged in a circle
        centres = []
        for i in range(8):
            angle = 2 * np.pi * i / 8
            centres.append([2 * np.cos(angle), 2 * np.sin(angle)])
        centres = np.array(centres)
        idx = np.random.randint(0, 8, size=n_samples)
        data = centres[idx] + 0.05 * np.random.randn(n_samples, 2)
    elif name == "pinwheel":
        radial = lambda n: np.sqrt(np.random.rand(n)) * 3
        num_per_blade = n_samples // 5
        data_list = []
        for k in range(5):
            r = radial(num_per_blade)
            t = (2 * np.pi * k / 5
                 + r * 0.4
                 + np.random.randn(num_per_blade) * 0.05)
            data_list.append(np.column_stack([r * np.cos(t), r * np.sin(t)]))
        data = np.concatenate(data_list, axis=0)
    else:
        raise ValueError(f"Unknown dataset: {name}")

    # Centre and lightly scale
    data = data - data.mean(axis=0)
    data = data / (data.std() + 1e-8)
    return data.astype(np.float32)


def infinite_data_loader(data: np.ndarray, batch_size: int, device: torch.device):
    """Yield random mini-batches forever."""
    n = len(data)
    while True:
        idx = np.random.randint(0, n, size=batch_size)
        yield torch.from_numpy(data[idx]).to(device)


# ------------------------------------------------------------------ #
#  Main training loop
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description="Train DITSB on 2-D data")
    parser.add_argument("--dataset", type=str, default="moons",
                        choices=["moons", "circles", "swissroll", "gaussian8", "pinwheel"])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--steps_per_epoch", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--solver", type=str, default="euler",
                        choices=["euler", "rk4", "dopri5"])
    parser.add_argument("--model", type=str, default="simple",
                        choices=["simple", "deep"])
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    # Device selection
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[DITSB] Device: {device}")

    # Load data
    data = load_dataset(args.dataset, n_samples=20_000)
    state_dim = data.shape[1]
    loader = infinite_data_loader(data, args.batch_size, device)
    print(f"[DITSB] Dataset: {args.dataset}  |  samples: {len(data)}  |  dim: {state_dim}")

    # Build model
    if args.model == "deep":
        vf = DeepVectorField(state_dim=state_dim, hidden_dim=args.hidden_dim).to(device)
    else:
        vf = ContinuousVectorField(state_dim=state_dim, hidden_dim=args.hidden_dim).to(device)

    flow = DITSB_GenerativeFlow(vf, solver=args.solver).to(device)

    num_params = sum(p.numel() for p in vf.parameters())
    print(f"[DITSB] Model: {args.model}  |  parameters: {num_params:,}")

    optimiser = torch.optim.AdamW(vf.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=args.epochs * args.steps_per_epoch
    )

    # Training
    print(f"[DITSB] Training for {args.epochs} epochs × {args.steps_per_epoch} steps …")
    t0 = time.time()
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
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            elapsed = time.time() - t0
            lr_now = scheduler.get_last_lr()[0]
            print(f"  epoch {epoch:>4d}/{args.epochs}  |  loss {avg:.6f}  |  lr {lr_now:.2e}  |  {elapsed:.1f}s")

    # Save checkpoint
    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_path = os.path.join(args.save_dir, f"ditsb_{args.dataset}.pt")
    torch.save({
        "vector_field_state": vf.state_dict(),
        "args": vars(args),
    }, ckpt_path)
    print(f"[DITSB] Checkpoint saved → {ckpt_path}")


if __name__ == "__main__":
    main()
