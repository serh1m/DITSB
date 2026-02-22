#!/usr/bin/env python
"""
train_lm.py — Discrete Flow Matching Language Model

Train a character-level language model on tiny-shakespeare using
Continuous-Time Markov Chain (CTMC) flow matching on the probability simplex.

Usage
-----
    python train_lm.py --epochs 50 --lr 3e-4 --seq_len 128
"""

import argparse
import math
import os
import time
import urllib.request

import numpy as np
import torch

from ditsb.discrete_flow import DiscreteFlowField
from ditsb.discrete_loss import discrete_flow_matching_loss
from ditsb.discrete_generate import discrete_generate


# ------------------------------------------------------------------ #
#  Data utilities
# ------------------------------------------------------------------ #

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def download_shakespeare() -> str:
    """Download tiny-shakespeare and return the text."""
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, "shakespeare.txt")
    if not os.path.exists(path):
        print(f"[train_lm] Downloading tiny-shakespeare ...")
        urllib.request.urlretrieve(SHAKESPEARE_URL, path)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


class CharDataset:
    """Character-level dataset that yields random fixed-length windows."""

    def __init__(self, text: str, seq_len: int):
        self.chars = sorted(set(text))
        self.vocab_size = len(self.chars)
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        self.data = [self.stoi[c] for c in text]
        self.seq_len = seq_len

    def decode(self, ids) -> str:
        return "".join(self.itos[i] for i in ids)

    def random_batch(self, batch_size: int, device: torch.device) -> torch.Tensor:
        n = len(self.data) - self.seq_len
        indices = np.random.randint(0, n, size=batch_size)
        batch = np.array([self.data[i : i + self.seq_len] for i in indices])
        return torch.from_numpy(batch).long().to(device)


# ------------------------------------------------------------------ #
#  Main
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description="Train discrete-flow LM on tiny-shakespeare")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--steps_per_epoch", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--gen_steps", type=int, default=50,
                        help="Denoising steps for generation")
    parser.add_argument("--gen_len", type=int, default=256,
                        help="Length of generated text")
    args = parser.parse_args()

    device = (torch.device("cuda") if args.device == "auto" and torch.cuda.is_available()
              else torch.device("cpu"))
    print(f"[train_lm] Device: {device}")

    # Data
    text = download_shakespeare()
    dataset = CharDataset(text, args.seq_len)
    V = dataset.vocab_size
    print(f"[train_lm] Vocabulary: {V} chars  |  Text length: {len(text):,}")

    # Model
    model = DiscreteFlowField(
        vocab_size=V,
        max_seq_len=max(args.seq_len, args.gen_len),
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"[train_lm] Parameters: {num_params:,}")

    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=args.epochs * args.steps_per_epoch
    )

    # Training
    print(f"[train_lm] Training for {args.epochs} epochs ...")
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for _ in range(args.steps_per_epoch):
            x_1 = dataset.random_batch(args.batch_size, device)
            loss = discrete_flow_matching_loss(model, x_1, V)
            optimiser.zero_grad()
            loss.backward()
            # Engineering Fix: Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            scheduler.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / args.steps_per_epoch
        ppl = math.exp(min(avg_loss, 20.0))  # cap for numerical safety

        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            elapsed = time.time() - t0
            lr_now = scheduler.get_last_lr()[0]
            print(f"  epoch {epoch:>4d}/{args.epochs}  |  loss {avg_loss:.4f}"
                  f"  |  ppl {ppl:.1f}  |  lr {lr_now:.2e}  |  {elapsed:.1f}s")

        # Generate sample text every 10 epochs
        if epoch % 10 == 0 or epoch == args.epochs:
            model.eval()
            ids = discrete_generate(
                model, seq_len=args.gen_len, vocab_size=V,
                num_samples=1, steps=args.gen_steps, device=device,
                temperature=0.8,
            )
            sample = dataset.decode(ids[0].cpu().tolist())
            print(f"  --- sample (epoch {epoch}) ---")
            print(f"  {sample[:200]}")
            print(f"  ---")

    # Save
    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_path = os.path.join(args.save_dir, "ditsb_lm.pt")
    torch.save({
        "model_state": model.state_dict(),
        "vocab": dataset.stoi,
        "args": vars(args),
    }, ckpt_path)
    print(f"[train_lm] Checkpoint saved: {ckpt_path}")


if __name__ == "__main__":
    main()
