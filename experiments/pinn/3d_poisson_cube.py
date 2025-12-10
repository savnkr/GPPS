# %%
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from torch.optim.lr_scheduler import StepLR
np.random.seed(0)
torch.manual_seed(0)

# Fourier feature mapping
class FourierFeatureMapping(nn.Module):
    def __init__(self, in_features, mapping_size=128, scale=5.0):
        super().__init__()
        self.B = torch.randn((in_features, mapping_size)) * scale

    def forward(self, x):
        x_proj = x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

# PINN model
class PINN(nn.Module):
    def __init__(self, in_dim=3, hidden_dim=64, out_dim=1, num_layers=4, ff_features=256,scale=5):
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

# Function f(x,y,z)
def forcing_term(x):
    x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]
    return -3 * (np.pi ** 2) * torch.sin(np.pi * x1) * torch.sin(np.pi * x2) * torch.sin(np.pi * x3)

# Sample domain points in [0,1]^3
def sample_domain(n):
    return torch.rand(n, 3)

# Sample boundary points on the cube
def sample_boundary(n):
    pts = torch.rand(n, 3)
    idx = torch.randint(0, 3, (n,))
    vals = torch.randint(0, 2, (n,), dtype=torch.float32)
    for i in range(n):
        pts[i, idx[i]] = vals[i]
    return pts

# PDE residual
def pde_residual(model, x):
    x.requires_grad_(True)
    u = model(x)
    grads = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(grads[:, 0], x, grad_outputs=torch.ones_like(grads[:, 0]), create_graph=True)[0][:, 0]
    u_yy = torch.autograd.grad(grads[:, 1], x, grad_outputs=torch.ones_like(grads[:, 1]), create_graph=True)[0][:, 1]
    u_zz = torch.autograd.grad(grads[:, 2], x, grad_outputs=torch.ones_like(grads[:, 2]), create_graph=True)[0][:, 2]
    f_val = forcing_term(x)
    return -(u_xx + u_yy + u_zz) - f_val

# Training
def train(model, epochs=5000, lr=1e-3, n_domain=1000, n_boundary=300,lamda=5):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=200, gamma=0.8)

    for epoch in range(epochs):
        x_domain = sample_domain(n_domain)
        x_boundary = sample_boundary(n_boundary)

        loss_pde = torch.mean(pde_residual(model, x_domain)**2)
        loss_bc = torch.mean(model(x_boundary)**2)
        loss = loss_pde + lamda*loss_bc

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Total Loss: {loss.item():.4e}, PDE Loss: {loss_pde.item():.4e}, BC Loss: {loss_bc.item():.4e}")

# Evaluation with relative MSE calculation
def evaluate(model, resolution=20):
    # Create structured grid points (similar to SDD solver)
    x = torch.linspace(0, 1, resolution)
    y = torch.linspace(0, 1, resolution)
    z = torch.linspace(0, 1, resolution)
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
    pts = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=1)

    with torch.no_grad():
        u_pred = model(pts).numpy().flatten()
    
    # Exact solution
    u_exact = (torch.sin(np.pi * pts[:, 0]) *
               torch.sin(np.pi * pts[:, 1]) *
               torch.sin(np.pi * pts[:, 2])).numpy()

    # Calculate relative MSE
    mse = np.mean((u_pred - u_exact)**2)
    exact_variance = np.var(u_exact)
    relative_mse = mse / exact_variance
    
    # Also calculate relative L2 error for comparison
    rel_l2_error = np.linalg.norm(u_pred - u_exact) / np.linalg.norm(u_exact)
    
    print(f"Relative MSE: {relative_mse:.6e}")
    print(f"Relative L2 Error: {rel_l2_error:.6e}")
    print(f"Absolute MSE: {mse:.6e}")
    print(f"Max absolute error: {np.max(np.abs(u_pred - u_exact)):.6e}")

    # Visualization slice at z=0.5
    pts_np = pts.numpy()  # Convert to numpy for indexing
    idx = np.abs(pts_np[:, 2] - 0.5) < (1 / (2 * resolution))
    if np.sum(idx) > 0:
        x_plot = pts_np[idx, 0]
        y_plot = pts_np[idx, 1]
        u_slice_pred = u_pred[idx]
        u_slice_exact = u_exact[idx]

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
        
        # Predicted solution
        im1 = ax1.tricontourf(x_plot, y_plot, u_slice_pred, levels=50)
        ax1.set_title("Predicted u(x,y,z=0.5)")
        plt.colorbar(im1, ax=ax1)
        
        # Exact solution
        im2 = ax2.tricontourf(x_plot, y_plot, u_slice_exact, levels=50)
        ax2.set_title("Exact u(x,y,z=0.5)")
        plt.colorbar(im2, ax=ax2)
        
        # Error
        error_slice = np.abs(u_slice_pred - u_slice_exact)
        im3 = ax3.tricontourf(x_plot, y_plot, error_slice, levels=50)
        ax3.set_title("Absolute Error")
        plt.colorbar(im3, ax=ax3)
        
        plt.tight_layout()
        plt.show()
    
    return relative_mse, rel_l2_error

# Main execution with improved training parameters
model = PINN(hidden_dim=128, num_layers=6, ff_features=512, scale=3.0)
print("Training PINN model...")
train(model, epochs=5000, lr=1e-3, n_domain=200, n_boundary=50, lamda=10)
print("\nEvaluating model...")
rel_mse, rel_l2 = evaluate(model, resolution=25)


