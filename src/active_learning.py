#%%
import torch
import numpy as np
from sklearn.cluster import KMeans
from .domain import filter_candidates


def ucb_acquisition(mean, variance, kappa=2.0):
    return mean + kappa * torch.sqrt(variance)


def adaptive_sampling(solver, X_pool, X_train, Xd, Xn, y_train, na=5, kappa=2.0,
                     exclusion_radius=0.05, acquisition_function="variance", ar_ratio=0.3):
    """Adaptive sampling for mixed boundary conditions (heat equation)."""
    C_full = solver.compute_covariance_matrix(X_train, Xd, Xn)
    C_inv = solver.compute_inverse(C_full)
    
    X_pool_filtered = filter_candidates(X_pool, X_train, exclusion_radius)
    
    if X_pool_filtered.shape[0] == 0:
        print("Warning: No valid candidates remain after filtering.")
        return torch.zeros((0, 2), dtype=X_pool.dtype, device=X_pool.device)
    
    mean = solver.posterior_mean(X_pool_filtered, X_train, Xd, Xn, C_inv, y_train)
    cov_full = solver.posterior_covariance(X_pool_filtered, X_pool_filtered, X_train, Xd, Xn, C_inv)
    variance = torch.clamp(torch.diag(cov_full), min=1e-10).unsqueeze(1)
    
    if acquisition_function == "variance":
        acquisition_values = variance
    elif acquisition_function == "ucb":
        acquisition_values = ucb_acquisition(mean, variance, kappa)
    else:
        acquisition_values = variance
        
    sorted_indices = torch.argsort(acquisition_values.squeeze(), descending=True)
    sorted_candidates = X_pool_filtered[sorted_indices]
    
    ar = max(int(ar_ratio * sorted_candidates.size(0)), na * 2)
    retained = sorted_candidates[:ar]
    
    retained_np = retained.cpu().detach().numpy()
    n_clusters = min(na, retained_np.shape[0])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(retained_np)
    centers = kmeans.cluster_centers_
    
    selected = []
    for i in range(n_clusters):
        dists = np.sum((retained_np - centers[i])**2, axis=1)
        idx = np.argmin(dists)
        selected.append(retained[idx])
    
    return torch.stack(selected) if selected else retained[:na]


def adaptive_sampling_poisson(solver, X_pool, X_train, Xb, y_train, na=5, kappa=2.0,
                              exclusion_radius=0.05, acquisition_function="variance", 
                              ar_ratio=0.3, use_clustering=True):
    """Adaptive sampling for Poisson equation with Dirichlet boundary only."""
    C_full = solver.compute_covariance_matrix(X_train, Xb)
    C_inv = solver.compute_inverse(C_full)
    
    X_pool_filtered = filter_candidates(X_pool, X_train, exclusion_radius)
    
    if X_pool_filtered.shape[0] == 0:
        print("Warning: No valid candidates remain after filtering.")
        return torch.zeros((0, 2), dtype=X_pool.dtype, device=X_pool.device)
    
    mean = solver.posterior_mean(X_pool_filtered, X_train, Xb, C_inv, y_train)
    cov_full = solver.posterior_covariance(X_pool_filtered, X_pool_filtered, X_train, Xb, C_inv).detach()
    variance = torch.clamp(torch.diag(cov_full), min=1e-10).unsqueeze(1)
    
    if acquisition_function == "variance":
        acquisition_values = variance
    elif acquisition_function == "ucb":
        acquisition_values = ucb_acquisition(mean, variance, kappa)
    else:
        acquisition_values = variance
        
    sorted_indices = torch.argsort(acquisition_values.squeeze(), descending=True)
    sorted_candidates = X_pool_filtered[sorted_indices]
    
    if not use_clustering:
        na_available = min(na, sorted_indices.shape[0])
        return sorted_candidates[:na_available]
    
    ar = max(int(ar_ratio * sorted_candidates.size(0)), na * 2)
    retained = sorted_candidates[:ar]
    
    retained_np = retained.cpu().detach().numpy()
    n_clusters = min(na, retained_np.shape[0])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(retained_np)
    centers = kmeans.cluster_centers_
    
    selected = []
    for i in range(n_clusters):
        dists = np.sum((retained_np - centers[i])**2, axis=1)
        idx = np.argmin(dists)
        selected.append(retained[idx])
    
    return torch.stack(selected) if selected else retained[:na]
