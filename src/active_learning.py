import torch
import numpy as np
from sklearn.cluster import KMeans
from .domain import filter_candidates


def ucb_acquisition(mean, variance, kappa=2.0):
    return mean + kappa * torch.sqrt(variance)


def adaptive_sampling(solver, X_pool, X_train, Xd, Xn, y_train, na=5, kappa=2.0,
                     exclusion_radius=0.05, acquisition_function="variance", ar_ratio=0.3):
    C_full = solver.compute_covariance_matrix(X_train, Xd, Xn)
    C_inv = solver.compute_inverse(C_full)

    X_pool_filtered = filter_candidates(X_pool, X_train, exclusion_radius)
    dim = X_pool.shape[-1]

    if X_pool_filtered.shape[0] == 0:
        print("Warning: No valid candidates remain after filtering.")
        return torch.zeros((0, dim), dtype=X_pool.dtype, device=X_pool.device)

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


def adaptive_sampling_sdd(solver, X_pool, X_train, Xd, Xn, y_train, C_full, sdd_cfg,
                          na=5, kappa=2.0, exclusion_radius=0.05,
                          acquisition_function="variance", ar_ratio=0.3):
    """
    SDD-based adaptive sampling for PDEs with mixed boundary conditions (e.g. heat equation).
    Uses Stochastic Dual Descent instead of exact matrix inversion for scalability.

    Args:
        solver: GPPDESolver instance
        X_pool: Candidate pool [N_pool, d]
        X_train: Current interior training points [M, d]
        Xd: Dirichlet boundary points
        Xn: Neumann boundary points
        y_train: Training targets [N_total, 1]
        C_full: Pre-computed covariance matrix [N_total, N_total]
        sdd_cfg: Dict with SDD params (batch_size, beta, rho, r, num_epochs, jitter)
        na: Number of points to acquire
        kappa: UCB exploration parameter
        exclusion_radius: Minimum distance from existing points
        acquisition_function: "variance" or "ucb"
        ar_ratio: Fraction of top candidates retained before clustering
    """
    from .sdd import train_sdd

    X_pool_filtered = filter_candidates(X_pool, X_train, exclusion_radius)
    dim = X_pool.shape[-1]

    if X_pool_filtered.shape[0] == 0:
        print("Warning: No valid candidates remain after filtering.")
        return torch.zeros((0, dim), dtype=X_pool.dtype, device=X_pool.device)

    # SDD for mean: A_mean ≈ C^{-1} y
    A_mean = train_sdd(C_full, y_train, **sdd_cfg, verbose=False)

    # Posterior mean at pool points
    cov_vec = solver.compute_covariance_vector(X_pool_filtered, X_train, Xd, Xn).detach()
    mean = cov_vec @ A_mean

    # SDD for variance: A_var ≈ C^{-1} c(X_pool)^T
    sdd_cfg_var = {**sdd_cfg, 'num_epochs': max(sdd_cfg.get('num_epochs', 2000) // 2, 200)}
    A_var = train_sdd(C_full, cov_vec.T, **sdd_cfg_var, verbose=False)

    # Diagonal variance: Var(x) = k(x,x) - c(x)^T C^{-1} c(x)
    base_var = solver.kernel.variance.detach() ** 2
    correction = torch.sum(cov_vec * A_var.T, dim=1)
    variance = torch.clamp(base_var - correction, min=1e-10).unsqueeze(1)

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
    C_full = solver.compute_covariance_matrix(X_train, Xb)
    C_inv = solver.compute_inverse(C_full)
    dim = X_pool.shape[-1]

    X_pool_filtered = filter_candidates(X_pool, X_train, exclusion_radius)

    if X_pool_filtered.shape[0] == 0:
        print("Warning: No valid candidates remain after filtering.")
        return torch.zeros((0, dim), dtype=X_pool.dtype, device=X_pool.device)

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


def adaptive_sampling_poisson_sdd(solver, X_pool, X_train, Xb, y_train, C_full, sdd_cfg,
                                  na=5, kappa=2.0, exclusion_radius=0.05,
                                  acquisition_function="variance", ar_ratio=0.3,
                                  use_clustering=True):
    """
    SDD-based adaptive sampling for Poisson equation with Dirichlet BCs.
    Uses Stochastic Dual Descent instead of exact matrix inversion for scalability.

    Args:
        solver: GPPoissonSolver instance
        X_pool: Candidate pool [N_pool, d]
        X_train: Current interior training points [M, d]
        Xb: Boundary points
        y_train: Training targets [N_total, 1]
        C_full: Pre-computed covariance matrix [N_total, N_total]
        sdd_cfg: Dict with SDD params (batch_size, beta, rho, r, num_epochs, jitter)
        na: Number of points to acquire
        kappa: UCB exploration parameter
        exclusion_radius: Minimum distance from existing points
        acquisition_function: "variance" or "ucb"
        ar_ratio: Fraction of top candidates retained before clustering
        use_clustering: Whether to use KMeans clustering for spatial diversity
    """
    from .sdd import train_sdd

    X_pool_filtered = filter_candidates(X_pool, X_train, exclusion_radius)
    dim = X_pool.shape[-1]

    if X_pool_filtered.shape[0] == 0:
        print("Warning: No valid candidates remain after filtering.")
        return torch.zeros((0, dim), dtype=X_pool.dtype, device=X_pool.device)

    # SDD for mean: A_mean ≈ C^{-1} y
    A_mean = train_sdd(C_full, y_train, **sdd_cfg, verbose=False)

    # Posterior mean at pool points
    cov_vec = solver.compute_covariance_vector(X_pool_filtered, X_train, Xb).detach()
    mean = cov_vec @ A_mean

    # SDD for variance: A_var ≈ C^{-1} c(X_pool)^T
    sdd_cfg_var = {**sdd_cfg, 'num_epochs': max(sdd_cfg.get('num_epochs', 2000) // 2, 200)}
    A_var = train_sdd(C_full, cov_vec.T, **sdd_cfg_var, verbose=False)

    base_var = solver.kernel.variance.detach() ** 2
    correction = torch.sum(cov_vec * A_var.T, dim=1)
    variance = torch.clamp(base_var - correction, min=1e-10).unsqueeze(1)

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
