import torch
import numpy as np
from scipy.stats import qmc


def generate_spacetime_domain(n_samples_per_side=10, dtype=torch.float64, device=None):
    device = device or torch.device('cpu')
    
    x = torch.linspace(0, 1, n_samples_per_side)
    t = torch.linspace(0, 1, n_samples_per_side)
    X, T = torch.meshgrid(x, t, indexing='ij')
    points = torch.stack([X.flatten(), T.flatten()], dim=1)

    interior_mask = (points[:, 0] > 0) & (points[:, 0] < 1) & (points[:, 1] > 0) & (points[:, 1] < 1)
    Xi = points[interior_mask]

    bottom_mask = (points[:, 1] == 0)
    left_mask = (points[:, 0] == 0)
    right_mask = (points[:, 0] == 1)
    top_mask = (points[:, 1] == 1)

    Xd = torch.cat((points[bottom_mask][1:-1], points[left_mask][:-1], points[right_mask][:-1]), dim=0)
    Xn = points[top_mask]

    f_Xi = torch.zeros(Xi.shape[0])
    g_Xd_bottom = torch.sin(torch.pi * points[bottom_mask][1:-1][:, 0])
    g_Xd_lr = torch.zeros(left_mask.sum() - 1 + right_mask.sum() - 1)
    g_Xd = torch.cat([g_Xd_bottom, g_Xd_lr], dim=0)
    g_Xn = torch.zeros(Xn.shape[0])

    return (Xi.to(dtype).to(device), Xd.to(dtype).to(device), Xn.to(dtype).to(device),
            f_Xi.to(dtype).to(device), g_Xd.to(dtype).to(device), g_Xn.to(dtype).to(device))


def generate_disc_domain(n_interior=100, n_boundary=50, r_interior=0.98, seed=None, dtype=torch.float64, device=None):
    device = device or torch.device('cpu')
    if seed is not None:
        torch.manual_seed(seed)
    
    theta = torch.rand(n_interior) * 2 * torch.pi
    r = r_interior * torch.sqrt(torch.rand(n_interior))
    Xi = torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=1)
    f_Xi = -torch.ones(n_interior, 1)
    
    theta_b = torch.linspace(0, 2 * torch.pi, n_boundary + 1)[:-1]
    Xb = torch.stack([torch.cos(theta_b), torch.sin(theta_b)], dim=1)
    g_Xb = torch.zeros(n_boundary, 1)
    
    return (Xi.to(dtype).to(device), Xb.to(dtype).to(device),
            f_Xi.to(dtype).to(device), g_Xb.to(dtype).to(device))


def sobol_sampling(n_points, x_range=(0, 1), t_range=(0, 1), seed=42, dtype=torch.float64, device=None):
    device = device or torch.device('cpu')
    sampler = qmc.Sobol(d=2, scramble=True, seed=seed)
    sample = sampler.random(n_points)
    sample[:, 0] = sample[:, 0] * (x_range[1] - x_range[0]) + x_range[0]
    sample[:, 1] = sample[:, 1] * (t_range[1] - t_range[0]) + t_range[0]
    return torch.tensor(sample, dtype=dtype, device=device)


def sobol_disk_sampling(n_points, r_max=0.98, seed=42, dtype=torch.float64, device=None):
    device = device or torch.device('cpu')
    sampler = qmc.Sobol(d=2, scramble=True, seed=seed)
    sample = sampler.random(n_points * 2)
    sample = 2 * sample - 1
    
    disk_points = np.zeros_like(sample)
    mask = (sample[:, 0]**2 + sample[:, 1]**2) > 0
    disk_points[mask, 0] = sample[mask, 0] * np.sqrt(1 - (sample[mask, 1]**2) / 2)
    disk_points[mask, 1] = sample[mask, 1] * np.sqrt(1 - (sample[mask, 0]**2) / 2)
    
    in_disk = np.sum(disk_points**2, axis=1) <= r_max**2
    disk_points = disk_points[in_disk][:n_points]
    
    return torch.tensor(disk_points, dtype=dtype, device=device)


def filter_candidates(X_pool, X_train, exclusion_radius=0.05):
    dists = torch.cdist(X_pool, X_train)
    mask = (dists.min(dim=1)[0] >= exclusion_radius)
    return X_pool[mask]
