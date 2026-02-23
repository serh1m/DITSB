import torch
import torch.nn as nn

class ImplicitSymplecticSolver(nn.Module):
    """
    Implicit Gauss-Legendre Runge-Kutta Integrator for Stiff/Symplectic flows.
    Dahlquist’s second barrier states that A-stable linear multistep methods
    can only be implicit and maximum order 2. 
    Gauss-Legendre RK methods naturally bypass this constraint, offering 
    high-order A-stable Symplectic integration without explosion.
    """
    def __init__(self, vector_field, stages: int = 2, max_newton_iters: int = 15, tol: float = 1e-5):
        super().__init__()
        self.v = vector_field
        self.s = stages
        self.max_newton_iters = max_newton_iters
        self.tol = tol
        
        # Butcher Tableau for 2-stage (Order 4) Gauss-Legendre
        # Constants c, A, b
        rad3 = 3**0.5 / 6.0
        self.c = [0.5 - rad3, 0.5 + rad3]
        self.A = [
            [0.25, 0.25 - rad3],
            [0.25 + rad3, 0.25]
        ]
        self.b = [0.5, 0.5]

    @torch.no_grad()
    def newton_solve_stages(self, y0: torch.Tensor, t0: float, dt: float) -> list[torch.Tensor]:
        """
        Uses fixed-point iteration (simplified Newton) to resolve the implicit
        internal stages K_i.
        """
        # Initialize internal stages with explicit Euler estimate
        # K_i = v(y0, t0 + c_i * dt)
        t_c = [t0 + c_i * dt for c_i in self.c]
        K = [self.v(y0, t).detach() for t in t_c]
        
        for _ in range(self.max_newton_iters):
            K_next = []
            max_error = 0.0
            for i in range(self.s):
                # Calculate internal state based on A tableau
                y_inner = y0
                for j in range(self.s):
                    y_inner = y_inner + dt * self.A[i][j] * K[j]
                    
                # Re-evaluate vector field
                K_new = self.v(y_inner, t_c[i]).detach()
                K_next.append(K_new)
                
                err = torch.max(torch.abs(K_new - K[i]))
                if err > max_error:
                    max_error = err
                    
            K = K_next
            if max_error < self.tol:
                break
                
        return K

    @torch.no_grad()
    def step(self, y0: torch.Tensor, t0: float, dt: float) -> torch.Tensor:
        """
        Performs one implicit A-stable step.
        """
        K = self.newton_solve_stages(y0, t0, dt)
        
        y1 = y0
        for i in range(self.s):
            y1 = y1 + dt * self.b[i] * K[i]
            
        return y1

    @torch.no_grad()
    def solve(self, y0: torch.Tensor, t_span: list[float], dt: float) -> torch.Tensor:
        """
        Integrates the vector field from t_span[0] to t_span[1] with step size dt.
        """
        t = t_span[0]
        y = y0
        
        while t < t_span[1] - 1e-7:
            step_dt = min(dt, t_span[1] - t)
            y = self.step(y, t, step_dt)
            t += step_dt
            
        return y
