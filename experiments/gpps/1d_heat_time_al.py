#%%
"""
Active Learning for 1D Time-Dependent Heat Equation
PDE: $\del u/\del t = \alpha \del^2 u/\del x^2 with \alpha = 0.01$
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

import torch
import numpy as np
import yaml

from src import (
    RBFKernel, HeatEquationOperators, GPPDESolver,
    generate_spacetime_domain, sobol_sampling, adaptive_sampling,
    HeatEquationFDM
)


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


def run_active_learning(solver, Xi, Xd, Xn, X_pool, X_test, u_ref, cfg, device):
    dtype = torch.float64
    n_per_iter = (cfg['training']['n_final'] - cfg['training']['n_initial']) // cfg['training']['n_iterations']
    
    g_Xd = torch.sin(torch.pi * Xd[:, 0]).reshape(-1, 1).to(device)
    g_Xn = torch.zeros(Xn.shape[0], 1, dtype=dtype, device=device)
    
    Xi_active = Xi.clone()
    errors, variances, n_points = [], [], [Xi_active.shape[0]]
    
    for it in range(cfg['training']['n_iterations'] + 1):
        f_Xi = torch.zeros(Xi_active.shape[0], 1, dtype=dtype, device=device)
        y_obs = torch.cat((f_Xi, g_Xd, g_Xn), dim=0)
        
        C_full = solver.compute_covariance_matrix(Xi_active, Xd, Xn)
        C_inv = solver.compute_inverse(C_full, cfg['training']['jitter'])
        
        pred_mean = solver.posterior_mean(X_test, Xi_active, Xd, Xn, C_inv, y_obs).detach()
        pred_cov = solver.posterior_covariance(X_test, X_test, Xi_active, Xd, Xn, C_inv).detach()
        pred_var = torch.diag(pred_cov).cpu().numpy()
        
        error = np.abs(pred_mean.cpu().numpy().reshape(-1) - u_ref)
        errors.append(np.mean(error))
        variances.append(np.mean(pred_var))
        
        print(f"Iter {it}: Points={Xi_active.shape[0]}, MAE={errors[-1]:.6f}, Var={variances[-1]:.6f}")
        
        if it < cfg['training']['n_iterations']:
            new_pts = adaptive_sampling(
                solver, X_pool, Xi_active, Xd, Xn, y_obs,
                na=n_per_iter,
                kappa=cfg['active_learning']['kappa'],
                exclusion_radius=cfg['active_learning']['exclusion_radius'],
                acquisition_function=cfg['active_learning']['acquisition_function'],
                ar_ratio=cfg['active_learning']['ar_ratio']
            )
            if new_pts.shape[0] > 0:
                Xi_active = torch.cat([Xi_active, new_pts], dim=0)
                n_points.append(Xi_active.shape[0])
    
    return {'points': n_points, 'errors': errors, 'variances': variances, 'Xi_final': Xi_active, 'pred_mean': pred_mean}


def run_random_baseline(solver, Xd, Xn, X_pool, X_test, u_ref, n_points, cfg, device):
    dtype = torch.float64
    perm = torch.randperm(X_pool.size(0))
    Xi_random = X_pool[perm[:n_points]]
    
    g_Xd = torch.sin(torch.pi * Xd[:, 0]).reshape(-1, 1).to(device)
    g_Xn = torch.zeros(Xn.shape[0], 1, dtype=dtype, device=device)
    f_Xi = torch.zeros(Xi_random.shape[0], 1, dtype=dtype, device=device)
    y_obs = torch.cat((f_Xi, g_Xd, g_Xn), dim=0)
    
    C_full = solver.compute_covariance_matrix(Xi_random, Xd, Xn)
    C_inv = solver.compute_inverse(C_full, cfg['training']['jitter'])
    
    pred_mean = solver.posterior_mean(X_test, Xi_random, Xd, Xn, C_inv, y_obs).detach()
    pred_cov = solver.posterior_covariance(X_test, X_test, Xi_random, Xd, Xn, C_inv).detach()
    
    error = np.mean(np.abs(pred_mean.cpu().numpy().reshape(-1) - u_ref))
    variance = np.mean(torch.diag(pred_cov).cpu().numpy())
    
    return {'error': error, 'variance': variance, 'Xi': Xi_random, 'pred_mean': pred_mean}


def save_results(results, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    np.savez(
        output_dir / 'results.npz',
        al_points=results['active']['points'],
        al_errors=results['active']['errors'],
        al_variances=results['active']['variances'],
        al_Xi=results['active']['Xi_final'].cpu().numpy(),
        al_pred=results['active']['pred_mean'].cpu().numpy(),
        random_error=results['random']['error'],
        random_variance=results['random']['variance'],
        random_Xi=results['random']['Xi'].cpu().numpy(),
        random_pred=results['random']['pred_mean'].cpu().numpy(),
        X_test=results['X_test'].cpu().numpy(),
        u_ref=results['u_ref']
    )
    print(f"Results saved to {output_dir / 'results.npz'}")


def main():
    config_path = ROOT_DIR / 'configs' / 'config.yaml'
    cfg = load_config(config_path)
    device = setup_device()
    dtype = torch.float64
    
    print(f"Using device: {device}")
    
    # Setup domain
    _, Xd, Xn, _, _, _ = generate_spacetime_domain(
        n_samples_per_side=int(np.sqrt(cfg['training']['n_initial'])),
        dtype=dtype, device=device
    )
    
    # Initial points and pool
    Xi_initial = sobol_sampling(cfg['training']['n_initial'], dtype=dtype, device=device)
    X_pool = sobol_sampling(cfg['training']['n_pool'], seed=123, dtype=dtype, device=device)
    
    # Test grid and reference solution
    X_test = create_test_grid(cfg['training']['n_test_grid'], dtype, device)
    ref_solver = HeatEquationFDM(
        N_x=cfg['training']['n_test_grid'],
        N_t=cfg['training']['n_test_grid'],
        alpha=cfg['pde']['alpha']
    )
    u_ref = ref_solver.interpolate(X_test.cpu())
    
    # Setup GP solver
    kernel = RBFKernel(
        lengthscale=cfg['kernel']['lengthscale'],
        variance=cfg['kernel']['variance'],
        device=device
    ).to(dtype)
    operators = HeatEquationOperators(kernel, alpha=cfg['pde']['alpha'])
    solver = GPPDESolver(kernel, operators)
    
    # Run experiments
    print("\n=== Active Learning ===")
    al_results = run_active_learning(solver, Xi_initial, Xd, Xn, X_pool, X_test, u_ref, cfg, device)
    
    print("\n=== Random Baseline ===")
    random_results = run_random_baseline(
        solver, Xd, Xn, X_pool, X_test, u_ref,
        n_points=cfg['training']['n_final'], cfg=cfg, device=device
    )
    print(f"Random: Points={cfg['training']['n_final']}, MAE={random_results['error']:.6f}, Var={random_results['variance']:.6f}")
    
    # Summary
    print("\n=== Results Summary ===")
    print(f"Active Learning: MAE={al_results['errors'][-1]:.6f}")
    print(f"Random Sampling: MAE={random_results['error']:.6f}")
    improvement = 100 * (random_results['error'] - al_results['errors'][-1]) / random_results['error']
    print(f"Improvement: {improvement:.2f}%")
    
    # Save results
    if cfg['output']['save_results']:
        results = {
            'active': al_results,
            'random': random_results,
            'X_test': X_test,
            'u_ref': u_ref
        }
        save_results(results, ROOT_DIR / cfg['output']['results_dir'])


if __name__ == "__main__":
    main()
