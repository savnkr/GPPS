#%%
r"""
Active Learning + SDD for 1D Time-Dependent Heat Equation
PDE: $\partial u/\partial t = \alpha \partial^2 u/\partial x^2$ with $\alpha = 0.01$

Combines active learning (uncertainty-based acquisition with clustering)
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
    RBFKernel, HeatEquationOperators, GPPDESolver,
    generate_spacetime_domain, sobol_sampling,
    adaptive_sampling_sdd, train_sdd, HeatEquationFDM,
    optimize_hyperparameters_sdd
)

EXPERIMENT_SEED = 100


def set_experiment_seed(seed=EXPERIMENT_SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)


set_experiment_seed()


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)['heat_equation']


def setup_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def create_test_grid(n_grid, dtype, device):
    x = torch.linspace(0, 1, n_grid)
    t = torch.linspace(0, 1, n_grid)
    X, T = torch.meshgrid(x, t, indexing='ij')
    return torch.stack([X.flatten(), T.flatten()], dim=1).to(dtype).to(device)


def build_sdd_cfg(cfg):
    """Build SDD config dict from YAML config."""
    return {
        'batch_size': cfg['sdd']['batch_size'],
        'beta': cfg['sdd']['beta'],
        'rho': cfg['sdd']['rho'],
        'jitter': cfg['training']['jitter'],
    }


def build_heat_solver(cfg, device, dtype):
    kernel = RBFKernel(
        lengthscale=cfg['kernel']['lengthscale'],
        variance=cfg['kernel']['variance'],
        device=device
    ).to(dtype)
    operators = HeatEquationOperators(kernel, alpha=cfg['pde']['alpha'])
    return GPPDESolver(kernel, operators)


def format_lengthscale(value):
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size == 1:
        return f"{arr[0]:.4f}"
    return "[" + ", ".join(f"{v:.4f}" for v in arr) + "]"


def sdd_posterior_stats(solver, X_test, Xi, Xd, Xn, C_full, y_obs, sdd_cfg, epochs_mean, epochs_var):
    """Compute posterior mean and diagonal variance using SDD.

    Returns correction (k* C^-1 k*^T) separately since base_var - correction
    suffers from catastrophic cancellation when correction ≈ base_var.
    """
    cfg_mean = {**sdd_cfg, 'num_epochs': epochs_mean}
    A_mean = train_sdd(C_full, y_obs, **cfg_mean, verbose=False)

    cov_vec = solver.compute_covariance_vector(X_test, Xi, Xd, Xn).detach()
    pred_mean = (cov_vec @ A_mean).detach()

    cfg_var = {**sdd_cfg, 'num_epochs': epochs_var}
    A_var = train_sdd(C_full, cov_vec.T, **cfg_var, verbose=False)

    base_var = solver.kernel.variance.detach() ** 2
    correction = torch.sum(cov_vec * A_var.T, dim=1)
    pred_var = torch.clamp(base_var - correction, min=1e-10)

    return pred_mean, pred_var.cpu().numpy(), correction.cpu().numpy()


def run_active_learning_sdd(solver, Xi, Xd, Xn, X_pool, X_test, u_ref, cfg, device,
                            optimize_hp=False, hp_every=2):
    dtype = torch.float64
    n_per_iter = (cfg['training']['n_final'] - cfg['training']['n_initial']) // cfg['training']['n_iterations']
    sdd_cfg = build_sdd_cfg(cfg)
    hp_sdd_cfg = {**sdd_cfg, 'num_epochs': 1000}

    g_Xd = torch.sin(torch.pi * Xd[:, 0]).reshape(-1, 1).to(device)
    g_Xn = torch.zeros(Xn.shape[0], 1, dtype=dtype, device=device)

    Xi_active = Xi.clone()
    errors, max_errors, n_points = [], [], [Xi_active.shape[0]]

    for it in range(cfg['training']['n_iterations'] + 1):
        f_Xi = torch.zeros(Xi_active.shape[0], 1, dtype=dtype, device=device)
        y_obs = torch.cat((f_Xi, g_Xd, g_Xn), dim=0)

        if optimize_hp and it > 0 and it % hp_every == 0:
            Xi_ref = Xi_active
            compute_cov = lambda: solver.compute_covariance_matrix(Xi_ref, Xd, Xn)
            ls, var = optimize_hyperparameters_sdd(
                solver.kernel, compute_cov, y_obs, hp_sdd_cfg,
                jitter=cfg['training']['jitter'], verbose=True,
                observation_group_sizes=[Xi_active.shape[0], Xd.shape[0], Xn.shape[0]],
            )
            print(f"  Updated HP: ls={format_lengthscale(ls)}, var={var:.4f}")

        C_full = solver.compute_covariance_matrix(Xi_active, Xd, Xn)

        # Evaluate at test points using SDD
        pred_mean, pred_var, correction = sdd_posterior_stats(
            solver, X_test, Xi_active, Xd, Xn, C_full, y_obs, sdd_cfg,
            cfg['sdd']['epochs_mean'], cfg['sdd']['epochs_var']
        )

        error = np.abs(pred_mean.cpu().numpy().reshape(-1) - u_ref)
        errors.append(np.mean(error))
        max_errors.append(np.max(error))

        print(f"Iter {it}: Points={Xi_active.shape[0]}, MAE={errors[-1]:.6f}, "
              f"MaxErr={max_errors[-1]:.6f}")

        if it < cfg['training']['n_iterations']:
            acq_sdd_cfg = {**sdd_cfg, 'num_epochs': cfg['sdd']['epochs_var']}
            new_pts = adaptive_sampling_sdd(
                solver, X_pool, Xi_active, Xd, Xn, y_obs, C_full, acq_sdd_cfg,
                na=n_per_iter,
                kappa=cfg['active_learning']['kappa'],
                exclusion_radius=cfg['active_learning']['exclusion_radius'],
                acquisition_function=cfg['active_learning']['acquisition_function'],
                ar_ratio=cfg['active_learning']['ar_ratio']
            )
            if new_pts.shape[0] > 0:
                Xi_active = torch.cat([Xi_active, new_pts], dim=0)
                n_points.append(Xi_active.shape[0])

    pred_std = np.sqrt(np.clip(pred_var, 1e-12, None))
    return {
        'points': n_points, 'errors': errors, 'max_errors': max_errors,
        'Xi_final': Xi_active, 'pred_mean': pred_mean, 'pred_std': pred_std
    }


def run_random_baseline_sdd(solver, Xd, Xn, X_pool, X_test, u_ref, cfg, device,
                            optimize_hp=False, hp_every=2, random_seed=EXPERIMENT_SEED):
    set_experiment_seed(random_seed)
    dtype = torch.float64
    sdd_cfg = build_sdd_cfg(cfg)
    hp_sdd_cfg = {**sdd_cfg, 'num_epochs': 1000}
    n_initial = cfg['training']['n_initial']
    n_final = cfg['training']['n_final']
    n_per_iter = (n_final - n_initial) // cfg['training']['n_iterations']

    perm = torch.randperm(X_pool.size(0))
    Xi_random_full = X_pool[perm[:n_final]]
    Xi_random = Xi_random_full[:n_initial].clone()

    g_Xd = torch.sin(torch.pi * Xd[:, 0]).reshape(-1, 1).to(device)
    g_Xn = torch.zeros(Xn.shape[0], 1, dtype=dtype, device=device)

    for it in range(cfg['training']['n_iterations'] + 1):
        f_Xi = torch.zeros(Xi_random.shape[0], 1, dtype=dtype, device=device)
        y_obs = torch.cat((f_Xi, g_Xd, g_Xn), dim=0)

        if optimize_hp and it > 0 and it % hp_every == 0:
            Xi_ref = Xi_random
            compute_cov = lambda: solver.compute_covariance_matrix(Xi_ref, Xd, Xn)
            optimize_hyperparameters_sdd(
                solver.kernel, compute_cov, y_obs, hp_sdd_cfg,
                jitter=cfg['training']['jitter'], verbose=False,
                observation_group_sizes=[Xi_random.shape[0], Xd.shape[0], Xn.shape[0]],
            )

        if it < cfg['training']['n_iterations']:
            next_size = min(n_initial + (it + 1) * n_per_iter, n_final)
            Xi_random = Xi_random_full[:next_size].clone()

    C_full = solver.compute_covariance_matrix(Xi_random, Xd, Xn)
    pred_mean, pred_var, correction = sdd_posterior_stats(
        solver, X_test, Xi_random, Xd, Xn, C_full, y_obs, sdd_cfg,
        cfg['sdd']['epochs_mean'], cfg['sdd']['epochs_var']
    )
    pred_std = np.sqrt(np.clip(pred_var, 1e-12, None))

    abs_error = np.abs(pred_mean.cpu().numpy().reshape(-1) - u_ref)
    error = np.mean(abs_error)
    max_error = np.max(abs_error)

    return {'error': error, 'max_error': max_error, 'Xi': Xi_random,
            'pred_mean': pred_mean, 'pred_std': pred_std}


def plot_results(al_results, random_results, X_test, u_ref, Xd, Xn, cfg, output_dir, n_grid):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16,
        'figure.dpi': 300, 'savefig.dpi': 600, 'savefig.bbox': 'tight',
    })

    X_np = X_test.cpu().numpy()
    x_grid = X_np[:, 0].reshape(n_grid, n_grid)
    t_grid = X_np[:, 1].reshape(n_grid, n_grid)
    u_ref_grid = u_ref.reshape(n_grid, n_grid)

    al_mean = al_results['pred_mean'].cpu().numpy().reshape(n_grid, n_grid)
    al_error = np.abs(al_mean - u_ref_grid)
    al_std = al_results['pred_std'].reshape(n_grid, n_grid)

    # Row plot: Ground Truth | AL+SDD Mean | Absolute Error | Std Dev
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))

    im0 = axes[0].contourf(x_grid, t_grid, u_ref_grid, levels=50, cmap='viridis')
    fig.colorbar(im0, ax=axes[0], shrink=0.8)
    axes[0].set_title('Ground Truth')
    axes[0].set_xlabel('x'); axes[0].set_ylabel('t')

    im1 = axes[1].contourf(x_grid, t_grid, al_mean, levels=50, cmap='viridis')
    fig.colorbar(im1, ax=axes[1], shrink=0.8)
    Xi_np = al_results['Xi_final'].cpu().numpy()
    axes[1].scatter(Xi_np[:, 0], Xi_np[:, 1], c='red', s=8, alpha=0.6, zorder=5)
    axes[1].set_title(f'AL+SDD Mean (MAE: {al_results["errors"][-1]:.5f})')
    axes[1].set_xlabel('x')

    im2 = axes[2].contourf(x_grid, t_grid, al_error, levels=50, cmap='coolwarm')
    fig.colorbar(im2, ax=axes[2], shrink=0.8)
    axes[2].set_title('Absolute Error')
    axes[2].set_xlabel('x')

    im3 = axes[3].contourf(x_grid, t_grid, al_std, levels=50, cmap='plasma')
    fig.colorbar(im3, ax=axes[3], shrink=0.8)
    axes[3].set_title('Standard Deviation')
    axes[3].set_xlabel('x')

    plt.tight_layout()
    fig.savefig(output_dir / 'row_heat_1d_al_sdd.pdf', format='pdf', bbox_inches='tight')
    fig.savefig(output_dir / 'row_heat_1d_al_sdd.png', format='png', dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved row plot to {output_dir}/row_heat_1d_al_sdd.[pdf|png]")

    # Convergence plot
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(al_results['points'], al_results['errors'], 'o-', color='#2ca02c', linewidth=2, label='AL + SDD')
    ax1.axhline(y=random_results['error'], color='#d62728', linestyle='--', linewidth=1.5,
                label=f'Random+SDD ({cfg["training"]["n_final"]} pts)')
    ax1.set_xlabel('Number of Training Points')
    ax1.set_ylabel('Mean Absolute Error')
    ax1.set_title('Error Convergence')
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(al_results['points'], al_results['max_errors'], 'o-', color='#1f77b4', linewidth=2, label='AL + SDD')
    ax2.axhline(y=random_results['max_error'], color='#d62728', linestyle='--', linewidth=1.5,
                label=f'Random+SDD ({cfg["training"]["n_final"]} pts)')
    ax2.set_xlabel('Number of Training Points')
    ax2.set_ylabel('Max Absolute Error')
    ax2.set_title('Worst-Case Error')
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig2.savefig(output_dir / 'convergence_heat_1d_sdd.pdf', format='pdf', bbox_inches='tight')
    fig2.savefig(output_dir / 'convergence_heat_1d_sdd.png', format='png', dpi=600, bbox_inches='tight')
    plt.close(fig2)
    print(f"Saved convergence plot to {output_dir}/convergence_heat_1d_sdd.[pdf|png]")


def save_results(results, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_dir / 'results_al_sdd.npz',
        al_points=results['active']['points'],
        al_errors=results['active']['errors'],
        al_max_errors=results['active']['max_errors'],
        al_Xi=results['active']['Xi_final'].cpu().numpy(),
        al_pred=results['active']['pred_mean'].cpu().numpy(),
        random_error=results['random']['error'],
        random_max_error=results['random']['max_error'],
        random_Xi=results['random']['Xi'].cpu().numpy(),
        random_pred=results['random']['pred_mean'].cpu().numpy(),
        X_test=results['X_test'].cpu().numpy(),
        u_ref=results['u_ref']
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

    _, Xd, Xn, _, _, _ = generate_spacetime_domain(
        n_samples_per_side=int(np.sqrt(cfg['training']['n_initial'])),
        dtype=dtype, device=device
    )

    Xi_initial = sobol_sampling(cfg['training']['n_initial'], dtype=dtype, device=device)
    X_pool = sobol_sampling(cfg['training']['n_pool'], seed=123, dtype=dtype, device=device)

    n_grid = cfg['training']['n_test_grid']
    X_test = create_test_grid(n_grid, dtype, device)
    ref_solver = HeatEquationFDM(N_x=n_grid, N_t=n_grid, alpha=cfg['pde']['alpha'])
    u_ref = ref_solver.interpolate(X_test.cpu())

    solver = build_heat_solver(cfg, device, dtype)

    print("\n=== Active Learning + SDD ===")
    set_experiment_seed()
    al_results = run_active_learning_sdd(
        solver, Xi_initial, Xd, Xn, X_pool, X_test, u_ref, cfg, device,
        optimize_hp=args.optimize_hp, hp_every=args.hp_every
    )

    print("\n=== Random Baseline + SDD ===")
    random_solver = build_heat_solver(cfg, device, dtype)
    random_results = run_random_baseline_sdd(
        random_solver, Xd, Xn, X_pool, X_test, u_ref,
        cfg=cfg, device=device,
        optimize_hp=args.optimize_hp, hp_every=args.hp_every
    )
    print(f"Random+SDD: Points={cfg['training']['n_final']}, MAE={random_results['error']:.6f}, "
          f"MaxErr={random_results['max_error']:.6f}")

    print("\n=== Results Summary ===")
    print(f"AL + SDD:      MAE={al_results['errors'][-1]:.6f}")
    print(f"Random + SDD:  MAE={random_results['error']:.6f}")
    improvement = 100 * (random_results['error'] - al_results['errors'][-1]) / random_results['error']
    print(f"Improvement:   {improvement:.2f}%")

    output_dir = ROOT_DIR / cfg['output']['results_dir']

    if cfg['output']['save_results']:
        results = {
            'active': al_results,
            'random': random_results,
            'X_test': X_test,
            'u_ref': u_ref
        }
        save_results(results, output_dir)

    plot_results(al_results, random_results, X_test, u_ref, Xd, Xn, cfg, output_dir, n_grid)


if __name__ == "__main__":
    main()
