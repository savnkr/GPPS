# %%
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torch.optim.lr_scheduler import StepLR

np.random.seed(12)
torch.manual_seed(12)

class FourierFeatureMapping(nn.Module):
    def __init__(self, in_features, mapping_size=256, scale=10):
        super().__init__()
        self.B = torch.randn((in_features, mapping_size)) * scale

    def forward(self, x):
        x_proj = x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class PINN(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=64, out_dim=1, num_layers=4, ff_features=256,scale=5):
        super().__init__()
        self.ff = FourierFeatureMapping(in_dim, ff_features,scale)
        layers = []
        in_features = ff_features * 2

        for i in range(num_layers):
            layers.append(nn.Linear(in_features if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x_ff = self.ff(x)
        return self.net(x_ff)

# PDE residual
def heat_equation_residual(model, xt, alpha=1.0):
    xt.requires_grad_(True)
    u = model(xt)
    grads = torch.autograd.grad(u, xt, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_t = grads[:, 1]
    u_x = grads[:, 0]
    u_xx = torch.autograd.grad(u_x, xt, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, 0]
    return u_t - alpha * u_xx

# Initial, boundary, and final-time condition samplers
def sample_domain(n):
    """Sample domain points with better coverage including structured grid points"""
    # Mix of random and structured sampling
    n_random = n // 2
    n_structured = n - n_random
    
    # Random sampling
    x_rand = torch.rand(n_random, 1)
    t_rand = torch.rand(n_random, 1)
    random_points = torch.cat([x_rand, t_rand], dim=1)
    
    # Structured sampling for better coverage
    if n_structured > 0:
        n_x = int(np.sqrt(n_structured)) + 1
        n_t = n_structured // n_x + 1
        x_struct = torch.linspace(0.05, 0.95, n_x).unsqueeze(1).repeat(n_t, 1)[:n_structured]
        t_struct = torch.linspace(0.05, 0.95, n_t).repeat(n_x, 1).t().flatten()[:n_structured].unsqueeze(1)
        structured_points = torch.cat([x_struct, t_struct], dim=1)
        
        return torch.cat([random_points, structured_points], dim=0)
    else:
        return random_points

def sample_boundary(n):
    """Sample boundary points with better temporal coverage"""
    # Ensure good temporal coverage
    t = torch.linspace(0, 1, n).unsqueeze(1)
    # Alternate between x=0 and x=1 boundaries
    xb = torch.zeros(n, 1)
    xb[n//2:] = 1.0
    return torch.cat([xb, t], dim=1)

def sample_initial(n):
    """Sample initial condition points with better spatial coverage"""
    # Use more structured sampling for initial conditions
    x = torch.linspace(0, 1, n).unsqueeze(1)
    t = torch.zeros_like(x)
    return torch.cat([x, t], dim=1), torch.sin(np.pi * x)

def sample_final(n):
    """Sample final time points with better spatial coverage"""
    # Use structured sampling for final time condition
    x = torch.linspace(0, 1, n).unsqueeze(1)
    t = torch.ones_like(x)
    return torch.cat([x, t], dim=1)

# Training loop
def train(model, epochs=5000, lr=1e-3, alpha=1.0):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=500, gamma=0.9)

    for epoch in range(epochs):
        # Increase number of collocation points for better coverage
        xt_domain = sample_domain(1000)  # Increased from 24
        xt_boundary = sample_boundary(50)  # Increased from 2
        xt_init, u_init = sample_initial(50)  # Increased from 2
        xt_final = sample_final(50)  # Increased from 2

        # Loss components with appropriate weighting
        loss_pde = torch.mean(heat_equation_residual(model, xt_domain, alpha) ** 2)
        loss_bc = torch.mean(model(xt_boundary) ** 2)
        loss_ic = torch.mean((model(xt_init) - u_init) ** 2)

        # Final-time derivative condition: ∂u/∂t(x, 1) = 0
        xt_final.requires_grad_(True)
        u_final = model(xt_final)
        grads_final = torch.autograd.grad(u_final, xt_final, grad_outputs=torch.ones_like(u_final), create_graph=True)[0]
        loss_ft = torch.mean(grads_final[:, 1] ** 2)

        # Weighted loss - emphasize initial and boundary conditions early in training
        weight_ic = 10.0 if epoch < 2000 else 5.0
        weight_bc = 10.0 if epoch < 2000 else 5.0
        weight_ft = 5.0
        
        loss = loss_pde + weight_bc * loss_bc + weight_ic * loss_ic + weight_ft * loss_ft

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 500 == 0:
            print(f"Epoch {epoch}: Total Loss = {loss.item():.4e}, PDE = {loss_pde.item():.2e}, IC = {loss_ic.item():.2e}, BC = {loss_bc.item():.2e}, Final Time = {loss_ft.item():.2e}")

def analytical_solution(x, t, alpha=1.0):
    """
    Analytical solution for heat equation with:
    - Initial condition: u(x,0) = sin(π*x)
    - Boundary conditions: u(0,t) = u(1,t) = 0
    - Solution: u(x,t) = sin(π*x) * exp(-π²*α*t)
    """
    return torch.sin(np.pi * x) * torch.exp(-np.pi**2 * alpha * t)

def evaluate(model, alpha=1.0, resolution=100):
    x = torch.linspace(0, 1, resolution)
    t = torch.linspace(0, 1, resolution)
    X, T = torch.meshgrid(x, t, indexing='ij')
    XT = torch.stack([X.flatten(), T.flatten()], dim=1)
    
    with torch.no_grad():
        u_pred = model(XT).numpy().flatten()
    
    u_exact = analytical_solution(XT[:, 0], XT[:, 1], alpha).numpy()
    
    mse = np.mean((u_pred - u_exact)**2)
    exact_variance = np.var(u_exact)
    relative_mse = mse / exact_variance
    
    rel_l2_error = np.linalg.norm(u_pred - u_exact) / np.linalg.norm(u_exact)
    
    print(f"Relative MSE: {relative_mse:.6e}")
    print(f"Relative L2 Error: {rel_l2_error:.6e}")
    print(f"Absolute MSE: {mse:.6e}")
    print(f"Max absolute error: {np.max(np.abs(u_pred - u_exact)):.6e}")
       
    return relative_mse, rel_l2_error

model = PINN(hidden_dim=128, num_layers=6, ff_features=256, scale=5.0)
print("Training PINN model with improved sampling...")
train(model, epochs=10000, lr=1e-3, alpha=1.0)
print("\nEvaluating model...")
rel_mse, rel_l2 = evaluate(model, alpha=1.0, resolution=100)


