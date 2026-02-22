"""
DITSB — Dynamic Information Schrodinger Bridge

A continuous-time generative framework based on Optimal Transport Flow Matching
with adjoint sensitivity for O(1) memory backpropagation.

Extensions:
  - Discrete Flow Matching (CTMC on probability simplex) for language modeling
  - Symplectic integrators for structure-preserving Hamiltonian dynamics
  - Mixture-of-Expert vector fields (basis function expansion)
  - Reward-guided flow (HJB terminal cost / RLHF equivalent)
"""

# --- Core (Phase 1) ---
from .vector_field import ContinuousVectorField, DeepVectorField
from .flow import DITSB_GenerativeFlow
from .loss import optimal_transport_loss
from .generate import generate_samples

# --- Extension 1: Discrete Flow Matching ---
from .discrete_flow import DiscreteFlowField
from .discrete_loss import discrete_flow_matching_loss
from .discrete_generate import discrete_generate

# --- Extension 2: Symplectic Integrators ---
from .symplectic import symplectic_euler_step, leapfrog_step, yoshida4_step, symplectic_integrate
from .hamiltonian_flow import HamiltonianVectorField, HamiltonianGenerativeFlow

# --- Extension 3: MoE Vector Field ---
from .moe_vector_field import MoEVectorField, ExpertBasisField

# --- Extension 4: Reward-Guided Flow ---
from .reward_flow import RewardGuidedFlow, RewardFunction
from .hjb_loss import hjb_terminal_cost_loss

# --- Phase 3: Beyond Transformer ---
from .fno import SpectralConv1d, FNOBlock, FNOBackbone
from .fno_flow import FNODiscreteFlowField, FNOContinuousVectorField
from .rectified_flow import (
    one_step_generate,
    reflow_loss,
    compute_straightness,
    generate_reflow_pairs,
    RectifiedFlowTrainer,
)

__all__ = [
    # Core
    "ContinuousVectorField",
    "DeepVectorField",
    "DITSB_GenerativeFlow",
    "optimal_transport_loss",
    "generate_samples",
    # Discrete flow
    "DiscreteFlowField",
    "discrete_flow_matching_loss",
    "discrete_generate",
    # Symplectic
    "symplectic_euler_step",
    "leapfrog_step",
    "yoshida4_step",
    "symplectic_integrate",
    "HamiltonianVectorField",
    "HamiltonianGenerativeFlow",
    # MoE
    "MoEVectorField",
    "ExpertBasisField",
    # Reward
    "RewardGuidedFlow",
    "RewardFunction",
    "hjb_terminal_cost_loss",
    # FNO
    "SpectralConv1d",
    "FNOBlock",
    "FNOBackbone",
    "FNODiscreteFlowField",
    "FNOContinuousVectorField",
    # Rectified Flow
    "one_step_generate",
    "reflow_loss",
    "compute_straightness",
    "generate_reflow_pairs",
    "RectifiedFlowTrainer",
]

__version__ = "0.3.0"
