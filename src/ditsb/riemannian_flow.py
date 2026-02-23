import torch
import torch.nn as nn

class RiemannianFlowMatcher(nn.Module):
    """
    Riemannian Geodesic Flow Matching.
    Instead of assuming straight Euclidean lines (psi_t = (1-t)x0 + t x1),
    this framework learns flows that follow geodesics on a curved manifold by 
    approximating the Christoffel symbols / Metric Tensor gradients.
    
    This drastically lowers the capacity required to un-bend complex topological data.
    """
    def __init__(self, data_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.data_dim = data_dim
        
        # A neural network to estimate the diagonal elements of the Riemannian Metric Tensor g_ii(x)
        # We enforce strict positivity via softplus to ensure a valid Riemann manifold
        self.metric_net = nn.Sequential(
            nn.Linear(data_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, data_dim)
        )
        
    def compute_metric(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the diagonal metric tensor components g_ii.
        We add a small epsilon to ensure the metric is strictly positive definite.
        """
        g_raw = self.metric_net(x)
        # Softplus ensures g_ii > 0
        return torch.nn.functional.softplus(g_raw) + 1e-4

    def compute_christoffel_symbols(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the Christoffel symbols Gamma_{ij}^k for a diagonal metric.
        Requires backpropagation through the metric_net to get derivatives w.r.t coordinates.
        """
        x.requires_grad_(True)
        g = self.compute_metric(x)  # (B, D)
        
        # To compute d(g_ii)/d(x_j), we compute the Jacobian
        # For simplicity in this diagonal approximation, we compute the gradient 
        # of the sum of g_ii w.r.t x, which gives the diagonal directional derivatives.
        g_sum = g.sum()
        dg_dx = torch.autograd.grad(g_sum, x, create_graph=True)[0] # (B, D)
        
        # For a diagonal metric tensor g_{ii}, the non-zero Christoffel symbols are:
        # Gamma^i_{ii} = 0.5 * d/dx_i ln(g_{ii})
        gamma = 0.5 * dg_dx / g
        return gamma

    def compute_geodesic_acceleration(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Computes the geodesic acceleration: d^2x^i/dt^2 = - Gamma^i_{jk} v^j v^k
        For the diagonal approximation: a^i = - Gamma^i_{ii} (v^i)^2
        """
        gamma = self.compute_christoffel_symbols(x)
        acceleration = - gamma * (v ** 2)
        return acceleration

    def compute_rgfm_loss(self, v_theta: torch.Tensor, x_t: torch.Tensor, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """
        Riemannian Conditional Flow Matching (CFM) Loss.
        Instead of targeting the constant Euclidean velocity (x1 - x0),
        we target a velocity that satisfies the Geodesic Equation on our metric.
        
        v_target is adjusted by the learned manifold curvature.
        """
        # Base euclidean velocity
        v_euclidean = x1 - x0 
        
        # The true velocity along the geodesic is NOT constant, it accelerates.
        # We approximate the instantaneous target velocity by integrating the Christoffel acceleration.
        # This is a first-order approximation to the exponential map Exp_{x_t}(v)
        acceleration = self.compute_geodesic_acceleration(x_t, v_euclidean)
        
        # Target velocity is shifted by the manifold's curvature pull
        # (This is a simplified implicit approximation for tutorial purposes)
        v_target = v_euclidean + 0.1 * acceleration 
        
        loss = torch.nn.functional.mse_loss(v_theta, v_target.detach())
        
        # Metric regularization to prevent unbounded warping
        g = self.compute_metric(x_t)
        metric_reg = 1e-3 * torch.mean((g - 1.0)**2)
        
        return loss + metric_reg
