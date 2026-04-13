import torch
import numpy as np
from sklearn.cluster import KMeans
from .domain import filter_candidates


def ucb_acquisition(mean, variance, kappa=2.0):
    return mean + kappa * torch.sqrt(variance)


def _select_from_clusters(retained, retained_acq_values, na):
    """Select the top-acquisition point from each KMeans cluster."""
    retained_np = retained.cpu().detach().numpy()
    n_clusters = min(na, retained_np.shape[0])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(retained_np)
    labels = kmeans.labels_

    acq_np = retained_acq_values.cpu().detach().numpy().reshape(-1)

    selected = []
    for c in range(n_clusters):
        mask = labels == c
        if not np.any(mask):
            continue
        cluster_indices = np.where(mask)[0]
        best_in_cluster = cluster_indices[np.argmax(acq_np[cluster_indices])]
        selected.append(retained[best_in_cluster])

    return torch.stack(selected) if selected else retained[:na]


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
    sorted_acq = acquisition_values[sorted_indices]

    ar = max(int(ar_ratio * sorted_candidates.size(0)), na * 2)
    retained = sorted_candidates[:ar]
    retained_acq = sorted_acq[:ar]

    return _select_from_clusters(retained, retained_acq, na)


def adaptive_sampling_sdd(solver, X_pool, X_train, Xd, Xn, y_train, C_full, sdd_cfg,
                          na=5, kappa=2.0, exclusion_radius=0.05,
                          acquisition_function="variance", ar_ratio=0.3):
    """Run SDD-based adaptive sampling for mixed-boundary PDEs."""
    from .sdd import train_sdd

    X_pool_filtered = filter_candidates(X_pool, X_train, exclusion_radius)
    dim = X_pool.shape[-1]

    if X_pool_filtered.shape[0] == 0:
        print("Warning: No valid candidates remain after filtering.")
        return torch.zeros((0, dim), dtype=X_pool.dtype, device=X_pool.device)

    # SDD for mean
    A_mean = train_sdd(C_full, y_train, **sdd_cfg, verbose=False)

    # Posterior mean at pool points
    cov_vec = solver.compute_covariance_vector(X_pool_filtered, X_train, Xd, Xn).detach()
    mean = cov_vec @ A_mean

    # SDD for variance: use correction as acquisition signal directly
    # (avoids catastrophic cancellation in var = k(x,x) - correction)
    A_var = train_sdd(C_full, cov_vec.T, **sdd_cfg, verbose=False)

    correction = torch.sum(cov_vec * A_var.T, dim=1)
    base_var = solver.kernel.variance.detach() ** 2
    variance = torch.clamp(base_var - correction, min=1e-10).unsqueeze(1)

    # For acquisition: use NEGATIVE correction (higher correction = lower variance = less interesting)
    # This avoids catastrophic cancellation: ranking by -correction == ranking by variance
    acquisition_proxy = -correction.unsqueeze(1)

    if acquisition_function == "variance":
        acquisition_values = acquisition_proxy
    elif acquisition_function == "ucb":
        acquisition_values = ucb_acquisition(mean, variance, kappa)
    else:
        acquisition_values = acquisition_proxy

    sorted_indices = torch.argsort(acquisition_values.squeeze(), descending=True)
    sorted_candidates = X_pool_filtered[sorted_indices]
    sorted_acq = acquisition_values[sorted_indices]

    ar = max(int(ar_ratio * sorted_candidates.size(0)), na * 2)
    retained = sorted_candidates[:ar]
    retained_acq = sorted_acq[:ar]

    return _select_from_clusters(retained, retained_acq, na)


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

    sorted_acq = acquisition_values[sorted_indices]
    ar = max(int(ar_ratio * sorted_candidates.size(0)), na * 2)
    retained = sorted_candidates[:ar]
    retained_acq = sorted_acq[:ar]

    return _select_from_clusters(retained, retained_acq, na)


def adaptive_sampling_poisson_sdd(solver, X_pool, X_train, Xb, y_train, C_full, sdd_cfg,
                                  na=5, kappa=2.0, exclusion_radius=0.05,
                                  acquisition_function="variance", ar_ratio=0.3,
                                  use_clustering=True):
    """Run SDD-based adaptive sampling for Poisson problems."""
    from .sdd import train_sdd

    X_pool_filtered = filter_candidates(X_pool, X_train, exclusion_radius)
    dim = X_pool.shape[-1]

    if X_pool_filtered.shape[0] == 0:
        print("Warning: No valid candidates remain after filtering.")
        return torch.zeros((0, dim), dtype=X_pool.dtype, device=X_pool.device)

    # SDD for mean
    A_mean = train_sdd(C_full, y_train, **sdd_cfg, verbose=False)

    # Posterior mean at pool points
    cov_vec = solver.compute_covariance_vector(X_pool_filtered, X_train, Xb).detach()
    mean = cov_vec @ A_mean

    # SDD for variance
    A_var = train_sdd(C_full, cov_vec.T, **sdd_cfg, verbose=False)

    correction = torch.sum(cov_vec * A_var.T, dim=1)
    base_var = solver.kernel.variance.detach() ** 2
    variance = torch.clamp(base_var - correction, min=1e-10).unsqueeze(1)

    # Use negative correction as acquisition proxy to avoid catastrophic cancellation
    acquisition_proxy = -correction.unsqueeze(1)

    if acquisition_function == "variance":
        acquisition_values = acquisition_proxy
    elif acquisition_function == "ucb":
        acquisition_values = ucb_acquisition(mean, variance, kappa)
    else:
        acquisition_values = acquisition_proxy

    sorted_indices = torch.argsort(acquisition_values.squeeze(), descending=True)
    sorted_candidates = X_pool_filtered[sorted_indices]

    if not use_clustering:
        na_available = min(na, sorted_indices.shape[0])
        return sorted_candidates[:na_available]

    sorted_acq = acquisition_values[sorted_indices]
    ar = max(int(ar_ratio * sorted_candidates.size(0)), na * 2)
    retained = sorted_candidates[:ar]
    retained_acq = sorted_acq[:ar]

    return _select_from_clusters(retained, retained_acq, na)
