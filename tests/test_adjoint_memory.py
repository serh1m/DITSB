import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint
import gc

class ContinuousDepthTransformerBlock(nn.Module):
    """
    A continuous-depth neural network block parameterized by time `t`.
    This represents the 'forward pass ODE' of a Continuous-Depth LLM.
    """
    def __init__(self, d_model):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_model)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(d_model, d_model)
        # Initialize near zero for stability (identity mapping initially)
        self.linear2.weight.data.fill_(0.0)
        self.linear2.bias.data.fill_(0.0)

    def forward(self, t, x):
        # Simple dynamics mimicking a residual block's derivative dx/dt
        out = self.linear1(x)
        out = self.act(out)
        out = self.linear2(out)
        return out

def trace_memory(forward_steps, block, x_initial):
    """
    Trace maximum memory overhead during forward and backward passes.
    """
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        device = "cuda"
    else:
        return 0, 0 # simple fallback
    
    x_initial = x_initial.to(device).requires_grad_(True)
    block = block.to(device)
    
    # Integration times: represents depth of the network
    # E.g. forward_steps = 100 means a 100-layer effective ODE-Transformer
    t_span = torch.linspace(0, 1, steps=forward_steps).to(device)

    # Profiling Forward and Backward
    with torch.autograd.profiler.profile(profile_memory=True) as prof:
        # Use odeint_adjoint strictly!
        out = odeint(block, x_initial, t_span, method='rk4', atol=1e-4, rtol=1e-4)
        
        # Loss
        loss = out[-1].sum()
        
        # Backprop: O(1) memory!
        loss.backward()

    # Get max memory used by CUDA during this block
    # Note: parsing prof.key_averages() can give exact allocations
    
    max_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    return max_mem_mb

def run_memory_profiling():
    print("="*60)
    print("O(1) Memory Profiling: Continuous-Depth LLM via Adjoint Method")
    print("="*60)
    
    # 1. Warmup
    d_model = 256
    block = ContinuousDepthTransformerBlock(d_model)
    x = torch.randn(1, 1024, d_model)
    trace_memory(10, block, x)
    
    # 2. Test different depth sizes (depth in ODE corresponds to `steps` parameter conceptually 
    # for the integrator or evaluating t_span length).
    # With standard backprop, memory scales O(N). With adjoint, memory is O(1).
    depths = [10, 50, 100, 200]
    
    if not torch.cuda.is_available():
        print("CUDA not available. Cannot accurately profile torch.cuda.max_memory_allocated().")
        print("Skipping detailed profiling results.")
        return

    memories = []
    for d in depths:
        mem = trace_memory(d, block, x)
        memories.append(mem)
        print(f"Integration Steps (Effective Depth): {d:3d}  -->  Max CUDA Memory: {mem:.2f} MB")
        
    # Validation constraint
    # Difference between depth 10 and depth 200 should be negligible (O(1) proof)
    mem_diff = abs(memories[-1] - memories[0])
    print("-" * 60)
    if mem_diff < 5.0: # Less than 5MB difference for huge depth increase
        print(f"[SUCCESS] O(1) Memory Constant Overhead verified! Diff: {mem_diff:.2f} MB")
    else:
        print(f"[FAILED] Memory scales significantly with depth. Diff: {mem_diff:.2f} MB")

if __name__ == "__main__":
    run_memory_profiling()
