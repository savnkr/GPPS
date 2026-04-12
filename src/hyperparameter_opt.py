"""
Optional SDD-based kernel hyperparameter optimization.

For GP-PDE observations y = T u with linear operator/boundary functionals T,
the exact model evidence p(y | theta) is mathematically well-defined.

This module uses a solver-aware hybrid criterion:

- Exact GP marginal likelihood when the observation system is small enough.
- Balanced holdout predictive NLL across observation blocks when block sizes
  are provided.
- SDD residual checks so candidates that the deployed SDD budget cannot solve
  accurately are rejected.

For larger systems it can still fall back to the original approximate NLL:

    NLL ≈ 0.5 * y^T SDD(K, y) + 0.5 * SLQ_logdet(K) + const

- Data-fit term uses SDD to approximate K^{-1}y without O(N^3) inversion.
- Log-determinant uses Stochastic Lanczos Quadrature (SLQ), which only
  needs O(N^2) matrix-vector products.
- Optimization uses conservative local grid search around the current kernel.

Usage:
    from src.hyperparameter_opt import optimize_hyperparameters_sdd

    # Inside AL loop:
    compute_cov = lambda: solver.compute_covariance_matrix(Xi, Xb)
    optimize_hyperparameters_sdd(kernel, compute_cov, y_obs, sdd_cfg)
"""
import math
from contextlib import contextmanager, nullcontext
import numpy as np
import torch
from .sdd import train_sdd


def _eye_like(K):
    return torch.eye(K.shape[0], dtype=K.dtype, device=K.device)


@contextmanager
def _torch_seed_context(seed, device):
    if seed is None:
        with nullcontext():
            yield
        return

    cpu_state = torch.random.get_rng_state()
    cuda_state = None
    cuda_index = None
    if device.type == 'cuda':
        cuda_index = device.index if device.index is not None else torch.cuda.current_device()
        cuda_state = torch.cuda.get_rng_state(cuda_index)

    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)

    try:
        yield
    finally:
        torch.random.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state, cuda_index)


def _balanced_validation_splits(group_sizes, n_splits=3, val_fraction=0.2, seed=0):
    """Create balanced holdout splits that preserve PDE/boundary observation types."""
    if group_sizes is None:
        return []

    total = int(sum(group_sizes))
    if total <= 1:
        return []

    offsets = np.cumsum([0, *map(int, group_sizes)])
    splits = []

    for split_id in range(n_splits):
        rng = np.random.default_rng(seed + split_id)
        val_chunks = []

        for start, end in zip(offsets[:-1], offsets[1:]):
            size = int(end - start)
            if size <= 1:
                continue

            n_val = max(1, int(round(val_fraction * size)))
            n_val = min(n_val, size - 1)
            idx = rng.choice(np.arange(start, end), size=n_val, replace=False)
            val_chunks.append(np.sort(idx))

        if not val_chunks:
            continue

        val_idx = np.sort(np.concatenate(val_chunks))
        train_mask = np.ones(total, dtype=bool)
        train_mask[val_idx] = False
        train_idx = np.where(train_mask)[0]
        if train_idx.size == 0 or val_idx.size == 0:
            continue

        splits.append((train_idx, val_idx))

    return splits


@torch.no_grad()
def exact_log_marginal_likelihood(K, y, jitter=1e-6):
    """Exact GP log marginal likelihood for linear GP observations."""
    N, du = K.shape[0], y.shape[1]
    K_reg = K + jitter * _eye_like(K)
    L = torch.linalg.cholesky(K_reg)
    alpha = torch.cholesky_solve(y, L)

    data_fit = 0.5 * (y.T @ alpha).sum().item()
    log_det = 2.0 * torch.log(torch.diagonal(L)).sum().item()
    constant = 0.5 * N * du * math.log(2 * math.pi)

    return data_fit + 0.5 * du * log_det + constant


@torch.no_grad()
def heldout_predictive_nll(K, y, train_idx, val_idx, jitter=1e-6):
    """Predictive NLL on a holdout subset of operator observations."""
    du = y.shape[1]
    K_reg = K + jitter * _eye_like(K)

    K_tt = K_reg[train_idx][:, train_idx]
    K_vt = K_reg[val_idx][:, train_idx]
    K_vv = K_reg[val_idx][:, val_idx]
    y_t = y[train_idx]
    y_v = y[val_idx]

    L_tt = torch.linalg.cholesky(K_tt)
    alpha = torch.cholesky_solve(y_t, L_tt)
    pred_mean = K_vt @ alpha

    solved = torch.cholesky_solve(K_vt.T, L_tt)
    pred_cov = K_vv - K_vt @ solved
    pred_cov = 0.5 * (pred_cov + pred_cov.T)
    pred_cov = pred_cov + max(jitter, 1e-8) * _eye_like(pred_cov)

    L_v = torch.linalg.cholesky(pred_cov)
    resid = y_v - pred_mean
    beta = torch.cholesky_solve(resid, L_v)

    data_fit = 0.5 * (resid.T @ beta).sum().item()
    log_det = 2.0 * torch.log(torch.diagonal(L_v)).sum().item()
    constant = 0.5 * len(val_idx) * du * math.log(2 * math.pi)

    return data_fit + 0.5 * du * log_det + constant


@torch.no_grad()
def sdd_relative_residual(K, y, sdd_cfg, jitter=1e-6, seed=None):
    """Residual that the deployed SDD solver achieves for this kernel setting."""
    with _torch_seed_context(seed, K.device):
        alpha = train_sdd(K, y, **sdd_cfg, verbose=False)
    K_reg = K + jitter * _eye_like(K)
    denom = torch.linalg.norm(y).clamp(min=1e-12)
    return float(torch.linalg.norm(K_reg @ alpha - y) / denom)


def _parameter_to_numpy(parameter):
    return parameter.detach().cpu().reshape(-1).numpy().astype(float)


def _copy_parameter_data(parameter, values):
    target = parameter.data
    arr = np.asarray(values, dtype=float).reshape(-1)
    tensor = torch.as_tensor(arr, dtype=target.dtype, device=target.device)
    if tensor.numel() == 1 and target.numel() > 1:
        tensor = tensor.repeat(target.numel())
    target.copy_(tensor.reshape_as(target))


def _format_lengthscale(values):
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 1:
        return f"{arr[0]:.4f}"
    return "[" + ", ".join(f"{v:.4f}" for v in arr) + "]"


@torch.no_grad()
def stochastic_log_det(K, n_probes=10, n_lanczos=30, seed=None):
    """Estimate log|K| using Stochastic Lanczos Quadrature (SLQ).

    Approximates tr(log(K)) = log|K| using random probe vectors and
    Lanczos tridiagonalization. Only requires matrix-vector products.

    Args:
        K: Symmetric positive-definite matrix [N, N]
        n_probes: Number of random probe vectors (more = lower variance)
        n_lanczos: Number of Lanczos iterations (more = better approximation)

    Returns:
        Scalar estimate of log|K|
    """
    N = K.shape[0]
    dtype, device = K.dtype, K.device
    estimates = []

    with _torch_seed_context(seed, device):
        for _ in range(n_probes):
            # Rademacher random vector (lower variance than Gaussian)
            z = torch.sign(torch.randn(N, dtype=dtype, device=device))
            z[z == 0] = 1.0
            z_norm_sq = float(N)  # ||z||^2 = N for Rademacher

            q = z / math.sqrt(z_norm_sq)

            # Lanczos iteration: build tridiagonal T_m such that
            # z^T f(K) z ≈ ||z||^2 * e_1^T f(T_m) e_1
            m = min(n_lanczos, N)
            alphas = torch.zeros(m, dtype=dtype, device=device)
            betas = torch.zeros(m, dtype=dtype, device=device)

            q_prev = torch.zeros_like(q)
            q_curr = q

            for j in range(m):
                w = K @ q_curr
                if j > 0:
                    w = w - betas[j] * q_prev
                alphas[j] = q_curr @ w
                w = w - alphas[j] * q_curr

                # Re-orthogonalization for numerical stability
                w = w - q_curr * (q_curr @ w)
                if j > 0:
                    w = w - q_prev * (q_prev @ w)

                b = w.norm()
                if b < 1e-12 or j == m - 1:
                    m_eff = j + 1
                    break
                betas[j + 1] = b
                q_prev = q_curr
                q_curr = w / b
            else:
                m_eff = m

            # Build tridiagonal matrix T
            T = torch.zeros(m_eff, m_eff, dtype=dtype, device=device)
            for j in range(m_eff):
                T[j, j] = alphas[j]
                if j < m_eff - 1:
                    T[j, j + 1] = betas[j + 1]
                    T[j + 1, j] = betas[j + 1]

            # Eigendecompose T (cheap: m_eff x m_eff)
            eigvals, eigvecs = torch.linalg.eigh(T)

            # Clamp eigenvalues to avoid log(0)
            eigvals = torch.clamp(eigvals, min=1e-30)

            # SLQ estimate: log|K| ≈ ||z||^2 * sum_i (e1^T v_i)^2 * log(lambda_i)
            weights = eigvecs[0, :] ** 2
            estimate = z_norm_sq * torch.sum(weights * torch.log(eigvals))
            estimates.append(estimate.item())

    return float(np.mean(estimates))


@torch.no_grad()
def sdd_approximate_nll(K, y, sdd_cfg, jitter=1e-6,
                        n_probes=20, n_lanczos=30, seed=None):
    """Compute approximate negative log marginal likelihood using SDD + SLQ.

    NLL = 0.5 * y^T K^{-1} y + 0.5 * log|K| + 0.5 * N * log(2*pi)

    Args:
        K: Covariance matrix [N, N] (before jitter)
        y: Observations [N, du]
        sdd_cfg: Dict for train_sdd (batch_size, beta, rho, num_epochs)
        jitter: Diagonal regularization
        n_probes: SLQ probe vectors (20+ recommended for stable estimates)
        n_lanczos: SLQ Lanczos iterations

    Returns:
        Scalar NLL estimate
    """
    N = K.shape[0]
    dtype, device = K.dtype, K.device
    K_reg = K + jitter * torch.eye(N, dtype=dtype, device=device)

    # Data-fit: 0.5 * y^T K^{-1} y via SDD
    with _torch_seed_context(seed, device):
        alpha = train_sdd(K_reg, y, **sdd_cfg, verbose=False)
    data_fit = 0.5 * (y.T @ alpha).sum().item()

    # Complexity: 0.5 * log|K|  via SLQ
    log_det = stochastic_log_det(K_reg, n_probes, n_lanczos, seed=seed)
    complexity = 0.5 * log_det

    constant = 0.5 * N * math.log(2 * math.pi)

    return data_fit + complexity + constant


@torch.no_grad()
def optimize_hyperparameters_sdd(kernel, compute_cov_matrix, y_train, sdd_cfg,
                                 jitter=1e-6, n_probes=20, n_lanczos=30,
                                 n_grid=5, n_repeats=3,
                                 search_range=1.25, verbose=False,
                                 observation_group_sizes=None,
                                 validation_fraction=0.2,
                                 validation_splits=3,
                                 random_seed=0,
                                 exact_threshold=600,
                                 max_sdd_rel_residual=5e-3,
                                 max_residual_growth=1.10,
                                 optimize_variance=True,
                                 variance_search_range=1.10,
                                 coordinate_passes=2):
    """Optimize kernel hyperparameters with a solver-aware evidence criterion.

    For moderate system sizes this uses exact GP marginal likelihood and
    balanced holdout predictive NLL. For larger systems it falls back to
    the original SDD + SLQ approximate NLL. In all cases it rejects kernel
    settings that SDD cannot solve accurately enough under the provided
    iteration budget.

    Args:
        kernel: RBFKernel instance (modified in-place)
        compute_cov_matrix: Callable returning the GP-PDE covariance matrix
                           using current kernel state. Example:
                           lambda: solver.compute_covariance_matrix(Xi, Xb)
        y_train: Observation vector [N, du]
        sdd_cfg: Dict for train_sdd. Use moderate epochs, e.g.:
                 {'batch_size': 20, 'beta': 'auto', 'rho': 0.9, 'num_epochs': 1000}
        jitter: Diagonal regularization
        n_probes: SLQ probe count per NLL evaluation (20+ for stability)
        n_lanczos: SLQ Lanczos iterations
        n_grid: Number of grid points per coordinate search.
        n_repeats: NLL evaluations per candidate (averaged to reduce noise)
        search_range: Multiplicative trust region around current values.
        verbose: Print progress
        observation_group_sizes: Optional list of block sizes in y_train
            (e.g. [n_interior, n_boundary] or [n_interior, n_dirichlet, n_neumann]).
            When provided, candidates are ranked by balanced holdout predictive NLL.
        validation_fraction: Holdout fraction per observation block.
        validation_splits: Number of balanced holdout splits to average.
        random_seed: Seed for holdout split generation.
        exact_threshold: Use exact Cholesky marginal likelihood when N <= this.
        max_sdd_rel_residual: Absolute residual floor for well-conditioned solves.
        max_residual_growth: Relative residual budget against the current kernel.
        optimize_variance: Whether to include variance updates.
        variance_search_range: Smaller trust region used for variance updates.
        coordinate_passes: Number of alternating coordinate-descent passes.

    Returns:
        (best_lengthscale, best_variance) tuple
    """
    init_ls = _parameter_to_numpy(kernel.lengthscale)
    init_var = float(_parameter_to_numpy(kernel.variance)[0])
    N = y_train.shape[0]
    use_exact = N <= exact_threshold
    val_splits = _balanced_validation_splits(
        observation_group_sizes,
        n_splits=validation_splits,
        val_fraction=validation_fraction,
        seed=random_seed,
    )
    n_evals = 0

    def evaluate_candidate(ls_values, var_value):
        nonlocal n_evals
        _copy_parameter_data(kernel.lengthscale, ls_values)
        _copy_parameter_data(kernel.variance, [var_value])

        nlls = []
        try:
            K = compute_cov_matrix()
            if use_exact:
                train_score = exact_log_marginal_likelihood(K, y_train, jitter=jitter)
                nlls = [train_score]
                n_evals += 1
            else:
                for repeat_id in range(n_repeats):
                    eval_seed = random_seed + repeat_id
                    nll = sdd_approximate_nll(K, y_train, sdd_cfg, jitter,
                                              n_probes, n_lanczos, seed=eval_seed)
                    nlls.append(nll)
                train_score = float(np.mean(nlls))
                n_evals += n_repeats

            if val_splits:
                val_scores = [
                    heldout_predictive_nll(K, y_train, train_idx, val_idx, jitter=jitter)
                    for train_idx, val_idx in val_splits
                ]
                selection_score = float(np.mean(val_scores))
                selection_std = float(np.std(val_scores))
            else:
                selection_score = train_score
                selection_std = float(np.std(nlls))

            rel_residual = sdd_relative_residual(
                K, y_train, sdd_cfg, jitter=jitter, seed=random_seed + 10000
            )
        except Exception:
            train_score = float('inf')
            selection_score = float('inf')
            selection_std = float('nan')
            rel_residual = float('inf')

        return train_score, selection_score, selection_std, rel_residual

    baseline_train, baseline_score, _, baseline_residual = evaluate_candidate(init_ls, init_var)
    residual_limit = float('inf')
    if max_sdd_rel_residual is not None:
        residual_limit = max(max_sdd_rel_residual, baseline_residual * max_residual_growth)

    best_score = baseline_score
    best_ls = init_ls.copy()
    best_var = init_var

    def maybe_accept(ls_values, var_value, train_score, selection_score, selection_std, rel_residual):
        if rel_residual > residual_limit:
            selection_score = float('inf')
        if verbose:
            print(f"    HP: ls={_format_lengthscale(ls_values)}, var={var_value:.4f}, "
                  f"train={train_score:.2f}, score={selection_score:.2f}, "
                  f"res={rel_residual:.3e}, std={selection_std:.2f})")
        return selection_score

    best_score = maybe_accept(best_ls, best_var, baseline_train, baseline_score, float('nan'), baseline_residual)

    for _ in range(max(coordinate_passes, 1)):
        improved = False

        for dim in range(best_ls.size):
            center = best_ls[dim]
            ls_candidates = np.exp(np.linspace(
                np.log(max(center / search_range, 1e-6)),
                np.log(center * search_range),
                n_grid,
            ))

            dim_best_value = center
            dim_best_score = best_score
            for ls_value in ls_candidates:
                cand_ls = best_ls.copy()
                cand_ls[dim] = float(ls_value)
                train_score, selection_score, selection_std, rel_residual = evaluate_candidate(cand_ls, best_var)
                selection_score = maybe_accept(cand_ls, best_var, train_score, selection_score, selection_std, rel_residual)
                if selection_score < dim_best_score:
                    dim_best_score = selection_score
                    dim_best_value = float(ls_value)

            if dim_best_value != best_ls[dim]:
                best_ls[dim] = dim_best_value
                best_score = dim_best_score
                improved = True

        if optimize_variance:
            var_candidates = np.exp(np.linspace(
                np.log(max(best_var / variance_search_range, 1e-6)),
                np.log(best_var * variance_search_range),
                n_grid,
            ))
            var_best_value = best_var
            var_best_score = best_score
            for var_value in var_candidates:
                train_score, selection_score, selection_std, rel_residual = evaluate_candidate(best_ls, float(var_value))
                selection_score = maybe_accept(best_ls, float(var_value), train_score, selection_score, selection_std, rel_residual)
                if selection_score < var_best_score:
                    var_best_score = selection_score
                    var_best_value = float(var_value)

            if var_best_value != best_var:
                best_var = var_best_value
                best_score = var_best_score
                improved = True

        if not improved:
            break

    # Set kernel to best parameters
    _copy_parameter_data(kernel.lengthscale, best_ls)
    _copy_parameter_data(kernel.variance, [best_var])

    if verbose:
        print(f"    Optimized: ls={_format_lengthscale(best_ls)}, var={best_var:.4f} "
              f"(was ls={_format_lengthscale(init_ls)}, var={init_var:.4f}, "
              f"score={best_score:.2f}, {n_evals} evals)")

    if best_ls.size == 1:
        best_ls_out = float(best_ls[0])
    else:
        best_ls_out = best_ls.tolist()

    return best_ls_out, float(best_var)
