# DITSB-v2 — Dynamic Information Schrödinger Bridge (Grand Unified Theory)

A continuous-time generative framework built on **Optimal Transport Flow Matching** with the **adjoint sensitivity method** for O(1) memory back-propagation.

v2 introduces the **Grand Unified Manifold Framework**, integrating physical mechanics to solve edge cases in deep learning topology.

## Highlights

| Feature | Detail |
|---|---|
| **No autoregressive error accumulation** | Generation is a smooth ODE integration, not a sequential token-by-token decode |
| **O(1) memory training** | `odeint_adjoint` back-propagates by solving a reverse-time ODE — no activation checkpointing needed |
| **Straight-line transport** | OT loss learns geodesic paths ⟹ fewer integration steps at inference |
| **Entropic Optimal Transport (Sinkhorn)** | Eliminates chaotic noise trajectory crossings via $O(N \log N)$ entropy regularized minibus matching |
| **Continuous-Time Markov Chains (CTMC)** | Discrete probability simplex flow for exact token generation with 0 quantization error |
| **Implicit Symplectic Integration** | Breaks Dahlquist's barrier via Gauss-Legendre RK methods, removing stiff solver $NaN$ resonance |
| **Riemannian Geodesic Flow** | Maps curved data spaces dynamically to prevent Mode Collapse during straight-line reflow distillation |

## Quickstart

```bash
# Clone the repository
git clone git@github.com:serh1m/DITSB.git
cd DITSB

# Install the package in editable mode
pip install -e .

# Run full demo (train + generate + visualise)
python examples/demo.py --dataset moons --epochs 100
```

## Project Structure

```
DITSB/
├── src/
│   └── ditsb/              # Core package (vector fields, ODE integrators, OT loss)
│       ├── discrete_flow.py      # CTMC categorical matching
│       ├── implicit_integrator.py # Gauss-Legendre A-stable ODE solver
│       ├── riemannian_flow.py    # Metric-based curved space geodesics 
│       └── sinkhorn_ot.py        # Entropy-regularized minibatch flow matching
├── examples/               # Demonstrations and training scripts
│   ├── demo.py             # End-to-end 2D generation demo
│   ├── demo_reward.py      # Reward-guided OT flow demo
│   ├── scaling_law.py      # Empirical scaling scaling law tests
│   ├── train.py            # Basic training script for synthetic 2D data
│   └── train_lm.py         # Discrete flow learning for char-level language modeling
├── pyproject.toml          # Package build configuration
├── THEORY.md               # Original foundational math for DITSB
├── THEORY_V2.md            # Extended proofs for Topologies, SDE Integrators, and CTMC Flows
└── README.md
```

## Datasets

Five synthetic 2-D datasets are included for experimentation:

- `moons` — two interleaving half-moons
- `circles` — concentric circles
- `swissroll` — Swiss roll projected to 2-D
- `gaussian8` — 8 Gaussians on a ring
- `pinwheel` — 5-blade pinwheel

## Architecture

Two vector field parameterisations are provided:

1. **`ContinuousVectorField`** — lightweight MLP with time concatenation (good for quick demos)
2. **`DeepVectorField`** — residual MLP with sinusoidal time embedding (better for complex manifolds)

Select via `--model simple` or `--model deep`.

## Theory (v1 vs v2)

The baseline (v1) training objective minimises the conditional flow matching loss on the Euclidean manifold:

```
L(θ) = E_{t, z₀, z₁} ‖ v_θ(z_t, t) − (z₁ − z₀) ‖²
```

where `z_t = (1−t)·z₀ + t·z₁` is the OT interpolant between noise `z₀ ~ N(0,I)` and data `z₁ ~ p_data`.

In v2, the model solves for the exact topological connections:
1. `z_0` is no longer a random Normal draw relative to `z_1`; they are mapped efficiently by Sinkhorn divergence $\pi^* = \arg\min_{\pi} \sum \|z_0 - z_1\|^2 - \epsilon H(\pi)$.
2. Explicit tracking incorporates the derived Riemannian Christoffel symbols $\Gamma$ to eliminate network capacity saturation during self-distillation.

At inference, samples are generated via implicitly stable RK solvers:

```
dZ/dt = v_θ(Z_t, t),    Z₀ ~ N(0, I),    t ∈ [0, 1]
```
