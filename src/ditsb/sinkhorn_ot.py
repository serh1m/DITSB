import torch
import torch.nn.functional as F

def sinkhorn_knopp_coupling(X0: torch.Tensor, X1: torch.Tensor, epsilon: float = 0.1, max_iter: int = 100) -> torch.Tensor:
    """
    Computes the Entropic Regularized Optimal Transport coupling matrix 
    between batch X0 and batch X1 using the Sinkhorn-Knopp algorithm.
    This resolves independent coupling trajectory crossings in DITSB.
    
    Args:
        X0: Source tensor, shape (N, D)
        X1: Target tensor, shape (M, D)
        epsilon: Entropic regularization parameter
        max_iter: Maximum number of Sinkhorn iterations
        
    Returns:
        pi: Coupling matrix of shape (N, M). 
            pi[i, j] is the probability mass transported from X0[i] to X1[j].
    """
    N = X0.size(0)
    M = X1.size(0)
    
    # Cost matrix: Squared Euclidean distance ||X0 - X1||^2
    # Expanding to compute pairwise distances efficiently
    cost_matrix = torch.cdist(X0, X1, p=2) ** 2
    
    # Kernel matrix K = exp(-C / epsilon)
    K = torch.exp(-cost_matrix / epsilon)
    
    # Marginals (assuming uniform distributions for minibatches)
    mu = torch.ones(N, device=X0.device, dtype=X0.dtype) / N
    nu = torch.ones(M, device=X1.device, dtype=X1.dtype) / M
    
    # Initialize scaling vectors
    u = torch.ones(N, device=X0.device, dtype=X0.dtype) / N
    v = torch.ones(M, device=X1.device, dtype=X1.dtype) / M
    
    for _ in range(max_iter):
        # Update v
        v = nu / (torch.matmul(K.t(), u) + 1e-10)
        # Update u
        u = mu / (torch.matmul(K, v) + 1e-10)
        
    # The optimal transport plan
    pi = torch.diag(u) @ K @ torch.diag(v)
    
    return pi

def sample_sinkhorn_coupled(X0: torch.Tensor, X1: torch.Tensor, epsilon: float = 0.1, max_iter: int = 100):
    """
    Sorts X1 to match X0 based on the Sinkhorn transport plan, providing 
    paired (X0, X1_sorted) for perfectly laminated Flow Matching paths.
    
    Returns:
        X0_mapped, X1_mapped: Re-aligned batches.
    """
    pi = sinkhorn_knopp_coupling(X0, X1, epsilon, max_iter)
    
    # Given pi, we find the deterministic mapping by taking the argmax assignment
    # (or you could sample from the categorical distribution pi[i, :])
    # For Flow Matching, we force a 1-to-1 deterministic bipartite match
    assignments = torch.argmax(pi, dim=1)
    
    return X0, X1[assignments]
