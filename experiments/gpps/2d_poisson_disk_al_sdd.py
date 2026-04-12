#%%
r"""
Active Learning + SDD for 2D Poisson Equation on Unit Disk
PDE: \Delta u = -1 with u = 0 on boundary
Exact solution: u(x,y) = (1 - x^2 - y^2) / 4

Combines active learning (variance / UCB acquisition with optional clustering)
with Stochastic Dual Descent (Lin et al.) to avoid O(N^3) matrix inversion.
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

import torch
import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src import (
    RBFKernel, PoissonOperators, GPPoissonSolver,
    generate_disc_domain, sobol_disk_sampling,
    adaptive_sampling_poisson_sdd, train_sdd,
    optimize_hyperparameters_sdd
)

torch.manual_seed(100)
np.random.seed(100)


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)['poisson_2d']


def setup_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def exact_solution(X):
    x1, x2 = X[:, 0], X[:, 1]
    if hasattr(x1, 'numpy'):
        x1, x2 = x1.numpy(), x2.numpy()
    return (1 - x1**2 - x2**2) / 4


def build_sdd_cfg(cfg):
    """Build SDD config dict from YAML config."""
    return {
        'batch_size': cfg['sdd']['batch_size'],
        'beta': cfg['sdd']['beta'],
        'rho': cfg['sdd']['rho'],
        'jitter': cfg['training']['jitter'],
    }


def format_lengthscale(value):
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size == 1:
        return f"{arr[0]:.4f}"
    return "[" + ", ".join(f"{v:.4f}" for v in arr) + "]"


def sdd_posterior_stats(solver, X_test, Xi, Xb, C_full, y_obs, sdd_cfg, epochs_mean, epochs_var):
    """Compute posterior mean and diagonal variance using SDD.

    Returns correction (k* C^-1 k*^T) separately since base_var - correction
    suffers from catastrophic cancellation when correction ≈ base_var.
    """
    cfg_mean = {**sdd_cfg, 'num_epochs': epochs_mean}
    A_mean = train_sdd(C_full, y_obs, **cfg_mean, verbose=False)

    cov_vec = solver.compute_covariance_vector(X_test, Xi, Xb).detach()
    pred_mean = (cov_vec @ A_mean).detach()

    cfg_var = {**sdd_cfg, 'num_epochs': epochs_var}
    A_var = train_sdd(C_full, cov_vec.T, **cfg_var, verbose=False)

    base_var = solver.kernel.variance.detach() ** 2
    correction = torch.sum(cov_vec * A_var.T, dim=1)
    pred_var = torch.clamp(base_var - correction, min=1e-10)

    return pred_mean, pred_var.cpu().numpy(), correction.cpu().numpy()


def run_active_learning_sdd(solver, Xi, Xb, g_Xb, X_pool, X_test, u_ref, cfg, device,
                            use_clustering=True, optimize_hp=False, hp_every=2):
    dtype = torch.float64
    n_per_iter = (cfg['training']['n_final'] - cfg['training']['n_initial']) // cfg['training']['n_iterations']
    sdd_cfg = build_sdd_cfg(cfg)

    # Fast SDD config for hyperparameter optimization (fewer epochs)
    hp_sdd_cfg = {**sdd_cfg, 'num_epochs': 1000}

    Xi_active = Xi.clone()
    errors, max_errors, n_points = [], [], [Xi_active.shape[0]]

    for it in range(cfg['training']['n_iterations'] + 1):
        f_Xi = -torch.ones(Xi_active.shape[0], 1, dtype=dtype, device=device)
        y_obs = torch.cat((f_Xi, g_Xb), dim=0)

        # Optionally optimize hyperparameters
        if optimize_hp and it > 0 and it % hp_every == 0:
            compute_cov = lambda: solver.compute_covariance_matrix(Xi_active, Xb)
            ls, var = optimize_hyperparameters_sdd(
                solver.kernel, compute_cov, y_obs, hp_sdd_cfg,
                jitter=cfg['training']['jitter'], verbose=True,
                observation_group_sizes=[Xi_active.shape[0], Xb.shape[0]],
                optimize_variance=False,
            )
            print(f"  Updated HP: ls={format_lengthscale(ls)}, var={var:.4f}")

        C_full = solver.compute_covariance_matrix(Xi_active, Xb)

        pred_mean, pred_var, correction = sdd_posterior_stats(
            solver, X_test, Xi_active, Xb, C_full, y_obs, sdd_cfg,
            cfg['sdd']['epochs_mean'], cfg['sdd']['epochs_var']
        )

        error = np.abs(pred_mean.cpu().numpy().reshape(-1) - u_ref)
        errors.append(np.mean(error))
        max_errors.append(np.max(error))

        print(f"Iter {it}: Points={Xi_active.shape[0]}, MAE={errors[-1]:.6f}, "
              f"MaxErr={max_errors[-1]:.6f}")

        if it < cfg['training']['n_iterations']:
            acq_sdd_cfg = {**sdd_cfg, 'num_epochs': cfg['sdd']['epochs_var']}
            new_pts = adaptive_sampling_poisson_sdd(
                solver, X_pool, Xi_active, Xb, y_obs, C_full, acq_sdd_cfg,
                na=n_per_iter,
                kappa=cfg['active_learning']['kappa'],
                exclusion_radius=cfg['active_learning']['exclusion_radius'],
                acquisition_function=cfg['active_learning']['acquisition_function'],
                ar_ratio=cfg['active_learning']['ar_ratio'],
                use_clustering=use_clustering
            )
            if new_pts.shape[0] > 0:
                Xi_active = torch.cat([Xi_active, new_pts], dim=0)
                n_points.append(Xi_active.shape[0])

    pred_std = np.sqrt(np.clip(pred_var, 1e-12, None))
    return {
        'points': n_points, 'errors': errors, 'max_errors': max_errors,
        'Xi_final': Xi_active, 'pred_mean': pred_mean, 'pred_std': pred_std
    }


def plot_results(clustered, nocluster, X_test, u_ref, Xb, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16,
        'figure.dpi': 300, 'savefig.dpi': 600, 'savefig.bbox': 'tight',
    })

    X_np = X_test.cpu().numpy()
    x1, x2 = X_np[:, 0], X_np[:, 1]
    theta = np.linspace(0, 2 * np.pi, 100)
    circle_x, circle_y = np.cos(theta), np.sin(theta)

    pred_cl = clustered['pred_mean'].cpu().numpy().reshape(-1)
    error_cl = np.abs(pred_cl - u_ref)
    std_cl = clustered['pred_std'].reshape(-1)

    # Row plot: Ground Truth | Mean Prediction | Absolute Error | Std Dev
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))

    im0 = axes[0].tricontourf(x1, x2, u_ref, levels=50, cmap='viridis')
    fig.colorbar(im0, ax=axes[0], shrink=0.8)
    axes[0].plot(circle_x, circle_y, 'k--', alpha=0.5, linewidth=0.8)
    axes[0].set_title('Ground Truth')
    axes[0].set_xlabel('$x_1$'); axes[0].set_ylabel('$x_2$')
    axes[0].set_aspect('equal')

    im1 = axes[1].tricontourf(x1, x2, pred_cl, levels=50, cmap='viridis')
    fig.colorbar(im1, ax=axes[1], shrink=0.8)
    Xi_cl = clustered['Xi_final'].cpu().numpy()
    axes[1].scatter(Xi_cl[:, 0], Xi_cl[:, 1], c='white', s=6, alpha=0.7,
                    edgecolors='black', linewidth=0.3, zorder=5)
    axes[1].plot(circle_x, circle_y, 'k--', alpha=0.5, linewidth=0.8)
    axes[1].set_title(f'AL+SDD Mean (MAE: {clustered["errors"][-1]:.5f})')
    axes[1].set_xlabel('$x_1$')
    axes[1].set_aspect('equal')

    im2 = axes[2].tricontourf(x1, x2, error_cl, levels=50, cmap='coolwarm')
    fig.colorbar(im2, ax=axes[2], shrink=0.8)
    axes[2].plot(circle_x, circle_y, 'k--', alpha=0.5, linewidth=0.8)
    axes[2].set_title('Absolute Error')
    axes[2].set_xlabel('$x_1$')
    axes[2].set_aspect('equal')

    im3 = axes[3].tricontourf(x1, x2, std_cl, levels=50, cmap='plasma')
    fig.colorbar(im3, ax=axes[3], shrink=0.8)
    axes[3].plot(circle_x, circle_y, 'k--', alpha=0.5, linewidth=0.8)
    axes[3].set_title('Standard Deviation')
    axes[3].set_xlabel('$x_1$')
    axes[3].set_aspect('equal')

    plt.tight_layout()
    fig.savefig(output_dir / 'row_al_sdd_cluster_poisson_2d.pdf', format='pdf', bbox_inches='tight')
    fig.savefig(output_dir / 'row_al_sdd_cluster_poisson_2d.png', format='png', dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved row plot to {output_dir}/row_al_sdd_cluster_poisson_2d.[pdf|png]")

    # Convergence plot
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(clustered['points'], clustered['errors'], 'o-', color='#2ca02c', linewidth=2,
             label='AL+SDD + Clustering')
    ax1.plot(nocluster['points'], nocluster['errors'], 's--', color='#1f77b4', linewidth=2,
             label='AL+SDD (no clustering)')
    ax1.set_xlabel('Number of Training Points'); ax1.set_ylabel('Mean Absolute Error')
    ax1.set_title('Error Convergence'); ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(clustered['points'], clustered['max_errors'], 'o-', color='#2ca02c', linewidth=2,
             label='AL+SDD + Clustering')
    ax2.plot(nocluster['points'], nocluster['max_errors'], 's--', color='#1f77b4', linewidth=2,
             label='AL+SDD (no clustering)')
    ax2.set_xlabel('Number of Training Points')
    ax2.set_ylabel('Max Absolute Error')
    ax2.set_title('Worst-Case Error'); ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig2.savefig(output_dir / 'convergence_poisson_2d_sdd.pdf', format='pdf', bbox_inches='tight')
    fig2.savefig(output_dir / 'convergence_poisson_2d_sdd.png', format='png', dpi=600, bbox_inches='tight')
    plt.close(fig2)
    print(f"Saved convergence plot to {output_dir}/convergence_poisson_2d_sdd.[pdf|png]")


def save_results(results, X_test, u_ref, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_dir / 'results_al_sdd.npz',
        clustered_points=results['clustered']['points'],
        clustered_errors=results['clustered']['errors'],
        clustered_max_errors=results['clustered']['max_errors'],
        clustered_Xi=results['clustered']['Xi_final'].cpu().numpy(),
        clustered_pred=results['clustered']['pred_mean'].cpu().numpy(),
        clustered_std=results['clustered']['pred_std'],
        nocluster_points=results['nocluster']['points'],
        nocluster_errors=results['nocluster']['errors'],
        nocluster_max_errors=results['nocluster']['max_errors'],
        nocluster_Xi=results['nocluster']['Xi_final'].cpu().numpy(),
        nocluster_pred=results['nocluster']['pred_mean'].cpu().numpy(),
        nocluster_std=results['nocluster']['pred_std'],
        X_test=X_test.cpu().numpy(),
        u_ref=u_ref
    )
    print(f"Results saved to {output_dir / 'results_al_sdd.npz'}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--optimize-hp', action='store_true',
                        help='Enable SDD-based kernel hyperparameter optimization during AL')
    parser.add_argument('--hp-every', type=int, default=2,
                        help='Optimize hyperparameters every N AL iterations (default: 2)')
    args = parser.parse_args()

    config_path = ROOT_DIR / 'configs' / 'config.yaml'
    cfg = load_config(config_path)
    device = setup_device()
    dtype = torch.float64

    print(f"Using device: {device}")
    print("Inference method: Stochastic Dual Descent (SDD)")
    if args.optimize_hp:
        print(f"Hyperparameter optimization: ON (every {args.hp_every} iterations)")

    theta_b = torch.linspace(0, 2 * torch.pi, cfg['training']['n_boundary'] + 1)[:-1]
    Xb = torch.stack([torch.cos(theta_b), torch.sin(theta_b)], dim=1).to(dtype).to(device)
    g_Xb = torch.zeros(cfg['training']['n_boundary'], 1, dtype=dtype, device=device)

    Xi_test, _, _, _ = generate_disc_domain(
        n_interior=cfg['training']['n_test'], n_boundary=0, dtype=dtype, device=device
    )
    X_test = torch.cat((Xi_test, Xb), dim=0)
    u_ref = exact_solution(X_test.cpu())

    X_pool = sobol_disk_sampling(cfg['training']['n_pool'], seed=123, dtype=dtype, device=device)
    Xi_initial = sobol_disk_sampling(cfg['training']['n_initial'], dtype=dtype, device=device)

    kernel = RBFKernel(
        lengthscale=cfg['kernel']['lengthscale'],
        variance=cfg['kernel']['variance'],
        device=device
    ).to(dtype)
    operators = PoissonOperators(kernel)
    solver = GPPoissonSolver(kernel, operators)

    print("\n=== AL + SDD WITH Clustering ===")
    # Reset kernel to initial values for fair comparison
    kernel.set_lengthscale(cfg['kernel']['lengthscale'])
    kernel.set_variance(cfg['kernel']['variance'])
    clustered_results = run_active_learning_sdd(
        solver, Xi_initial.clone(), Xb, g_Xb, X_pool, X_test, u_ref, cfg, device,
        use_clustering=True, optimize_hp=args.optimize_hp, hp_every=args.hp_every
    )

    print("\n=== AL + SDD WITHOUT Clustering ===")
    # Reset kernel to initial values for fair comparison
    kernel.set_lengthscale(cfg['kernel']['lengthscale'])
    kernel.set_variance(cfg['kernel']['variance'])
    nocluster_results = run_active_learning_sdd(
        solver, Xi_initial.clone(), Xb, g_Xb, X_pool, X_test, u_ref, cfg, device,
        use_clustering=False, optimize_hp=args.optimize_hp, hp_every=args.hp_every
    )

    print("\n=== Results Summary ===")
    print(f"AL+SDD + Clustering:    MAE={clustered_results['errors'][-1]:.6f}, "
          f"MaxErr={clustered_results['max_errors'][-1]:.6f}")
    print(f"AL+SDD no Clustering:   MAE={nocluster_results['errors'][-1]:.6f}, "
          f"MaxErr={nocluster_results['max_errors'][-1]:.6f}")

    if nocluster_results['errors'][-1] > 0:
        improvement = 100 * (nocluster_results['errors'][-1] - clustered_results['errors'][-1]) / \
                      nocluster_results['errors'][-1]
        print(f"Improvement with clustering: {improvement:.2f}%")

    output_dir = ROOT_DIR / cfg['output']['results_dir']

    if cfg['output']['save_results']:
        results = {'clustered': clustered_results, 'nocluster': nocluster_results}
        save_results(results, X_test, u_ref, output_dir)

    plot_results(clustered_results, nocluster_results, X_test, u_ref, Xb, output_dir)


if __name__ == "__main__":
    main()
