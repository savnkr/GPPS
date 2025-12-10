# %%
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torch.optim.lr_scheduler import StepLR
np.random.seed(0)
torch.manual_seed(0)

# Fourier feature mapping
class FourierFeatureMapping(nn.Module):
    def __init__(self, in_features, mapping_size=256, scale=10):
        super().__init__()
        self.B = torch.randn((in_features, mapping_size)) * scale

    def forward(self, x):
        x_proj = x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

# PINN model
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

# Sample points in the unit disk
def sample_domain(n):
    r = torch.sqrt(torch.rand(n, 1))
    theta = 2 * np.pi * torch.rand(n, 1)
    x1 = r * torch.cos(theta)
    x2 = r * torch.sin(theta)
    return torch.cat([x1, x2], dim=1)

def sample_boundary(n):
    theta = 2 * np.pi * torch.rand(n, 1)
    x1 = torch.cos(theta)
    x2 = torch.sin(theta)
    return torch.cat([x1, x2], dim=1)

# PDE residual
def pde_residual(model, x):
    x.requires_grad_(True)
    u = model(x)
    grads = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(grads[:, 0], x, grad_outputs=torch.ones_like(grads[:, 0]), create_graph=True)[0][:, 0]
    u_yy = torch.autograd.grad(grads[:, 1], x, grad_outputs=torch.ones_like(grads[:, 1]), create_graph=True)[0][:, 1]
    residual = - (u_xx + u_yy) - 1.0
    return residual

# Training loop
def train(model, epochs=10000, lr=1e-3, n_domain=20, n_boundary=5,lamda=1):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=200, gamma=0.8)

    for epoch in range(epochs):
        x_domain = sample_domain(n_domain)
        x_boundary = sample_boundary(n_boundary)

        residual = pde_residual(model, x_domain)
        u_boundary = model(x_boundary)

        loss_pde = torch.mean(residual ** 2)
        loss_bc = torch.mean(u_boundary ** 2)
        loss = loss_pde + lamda*loss_bc

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4e}, PDE Loss: {loss_pde.item():.4e}, BC Loss: {loss_bc.item():.4e}")

# Evaluation
def evaluate(model, resolution=100):
    x = torch.linspace(-1, 1, resolution)
    y = torch.linspace(-1, 1, resolution)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    points = torch.stack([X.flatten(), Y.flatten()], dim=1)

    # Mask for unit disk
    mask = (points[:, 0] ** 2 + points[:, 1] ** 2) <= 1
    inside_points = points[mask]

    # Predictions and exact solution
    u_pred = model(inside_points).detach().numpy().flatten()
    u_exact = (1 - inside_points[:, 0]**2 / 4 - inside_points[:, 1]**2 / 2).numpy()
    error = np.abs(u_pred - u_exact)

    print("Relative L2 Error:", np.linalg.norm(error) / np.linalg.norm(u_exact))

    # Create prediction and error maps
    Z_pred = np.full(X.shape, np.nan)
    Z_error = np.full(X.shape, np.nan)

    idx = 0
    for i in range(resolution):
        for j in range(resolution):
            point = torch.tensor([[X[i, j], Y[i, j]]])
            if (point ** 2).sum() <= 1:
                pred = model(point).item()
                exact = 1 - X[i, j]**2 / 4 - Y[i, j]**2 / 2
                Z_pred[i, j] = pred
                Z_error[i, j] = abs(pred - exact)

    # Plot prediction
    plt.figure(figsize=(6, 5))
    plt.imshow(Z_pred, extent=(-1, 1, -1, 1), origin='lower', cmap='viridis')
    plt.colorbar(label='u(x)')
    plt.title("PINN Prediction")
    plt.show()

    # Plot error
    plt.figure(figsize=(6, 5))
    plt.imshow(Z_error, extent=(-1, 1, -1, 1), origin='lower', cmap='hot')
    plt.colorbar(label='Absolute Error')
    plt.title("Prediction Error")
    plt.show()

# Main
model = PINN(scale=5)
train(model, epochs=10000,n_domain=16,n_boundary=4,lamda=1,lr=1e-3)
evaluate(model)


# %%


