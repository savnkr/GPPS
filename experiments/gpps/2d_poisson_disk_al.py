#%%
"""
Active Learning for 2D Poisson Equation on Unit Disk
PDE: \Delta u = -1 with u = 0 on boundary
Exact solution: u(x,y) = (1 - x² - y²) / 4
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

import torch
import numpy as np
import yaml

from src import (
    RBFKernel, PoissonOperators, GPPoissonSolver,
    generate_disc_domain, sobol_disk_sampling, adaptive_sampling_poisson
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


def run_active_learning(solver, Xi, Xb, g_Xb, X_pool, X_test, u_ref, cfg, device, use_clustering=True):
    dtype = torch.float64
    n_per_iter = (cfg['training']['n_final'] - cfg['training']['n_initial']) // cfg['training']['n_iterations']
    
    Xi_active = Xi.clone()
    errors, variances, n_points = [], [], [Xi_active.shape[0]]
    
    for it in range(cfg['training']['n_iterations'] + 1):
        f_Xi = -torch.ones(Xi_active.shape[0], 1, dtype=dtype, device=device)
        y_obs = torch.cat((f_Xi, g_Xb), dim=0)
        
        C_full = solver.compute_covariance_matrix(Xi_active, Xb)
        C_inv = solver.compute_inverse(C_full, cfg['training']['jitter'])
        
        pred_mean = solver.posterior_mean(X_test, Xi_active, Xb, C_inv, y_obs).detach()
        pred_cov = solver.posterior_covariance(X_test, X_test, Xi_active, Xb, C_inv).detach()
        pred_var = torch.clamp(torch.diag(pred_cov), min=1e-10).cpu().numpy()
        
        error = np.abs(pred_mean.cpu().numpy().reshape(-1) - u_ref)
        errors.append(np.mean(error))
        variances.append(np.mean(pred_var))
        
        print(f"Iter {it}: Points={Xi_active.shape[0]}, MAE={errors[-1]:.6f}, Var={variances[-1]:.6f}")
        
        if it < cfg['training']['n_iterations']:
            new_pts = adaptive_sampling_poisson(
                solver, X_pool, Xi_active, Xb, y_obs,
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
    
    return {
        'points': n_points, 
        'errors': errors, 
        'variances': variances, 
        'Xi_final': Xi_active, 
        'pred_mean': pred_mean,
        'pred_var': pred_var
    }


def save_results(results, X_test, u_ref, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    np.savez(
        output_dir / 'results.npz',
        clustered_points=results['clustered']['points'],
        clustered_errors=results['clustered']['errors'],
        clustered_variances=results['clustered']['variances'],
        clustered_Xi=results['clustered']['Xi_final'].cpu().numpy(),
        clustered_pred=results['clustered']['pred_mean'].cpu().numpy(),
        clustered_var=results['clustered']['pred_var'],
        nocluster_points=results['nocluster']['points'],
        nocluster_errors=results['nocluster']['errors'],
        nocluster_variances=results['nocluster']['variances'],
        nocluster_Xi=results['nocluster']['Xi_final'].cpu().numpy(),
        nocluster_pred=results['nocluster']['pred_mean'].cpu().numpy(),
        nocluster_var=results['nocluster']['pred_var'],
        X_test=X_test.cpu().numpy(),
        u_ref=u_ref
    )
    print(f"Results saved to {output_dir / 'results.npz'}")


def main():
    config_path = ROOT_DIR / 'configs' / 'config.yaml'
    cfg = load_config(config_path)
    device = setup_device()
    dtype = torch.float64
    
    print(f"Using device: {device}")
    
    # Setup boundary points
    theta_b = torch.linspace(0, 2 * torch.pi, cfg['training']['n_boundary'] + 1)[:-1]
    Xb = torch.stack([torch.cos(theta_b), torch.sin(theta_b)], dim=1).to(dtype).to(device)
    g_Xb = torch.zeros(cfg['training']['n_boundary'], 1, dtype=dtype, device=device)
    
    # Test points
    Xi_test, _, _, _ = generate_disc_domain(n_interior=cfg['training']['n_test'], n_boundary=0, dtype=dtype, device=device)
    X_test = torch.cat((Xi_test, Xb), dim=0)
    u_ref = exact_solution(X_test.cpu())
    
    # Candidate pool
    X_pool = sobol_disk_sampling(cfg['training']['n_pool'], seed=123, dtype=dtype, device=device)
    
    # Initial points (same for both methods)
    Xi_initial = sobol_disk_sampling(cfg['training']['n_initial'], dtype=dtype, device=device)
    
    # Setup GP solver
    kernel = RBFKernel(
        lengthscale=cfg['kernel']['lengthscale'],
        variance=cfg['kernel']['variance'],
        device=device
    ).to(dtype)
    operators = PoissonOperators(kernel)
    solver = GPPoissonSolver(kernel, operators)
    
    # Run with clustering
    print("\n=== Active Learning WITH Clustering ===")
    clustered_results = run_active_learning(
        solver, Xi_initial.clone(), Xb, g_Xb, X_pool, X_test, u_ref, cfg, device, use_clustering=True
    )
    
    # Run without clustering
    print("\n=== Active Learning WITHOUT Clustering ===")
    nocluster_results = run_active_learning(
        solver, Xi_initial.clone(), Xb, g_Xb, X_pool, X_test, u_ref, cfg, device, use_clustering=False
    )
    
    # Summary
    print("\n=== Results Summary ===")
    print(f"With Clustering:    MAE={clustered_results['errors'][-1]:.6f}, Var={clustered_results['variances'][-1]:.6f}")
    print(f"Without Clustering: MAE={nocluster_results['errors'][-1]:.6f}, Var={nocluster_results['variances'][-1]:.6f}")
    
    if nocluster_results['errors'][-1] > 0:
        improvement = 100 * (nocluster_results['errors'][-1] - clustered_results['errors'][-1]) / nocluster_results['errors'][-1]
        print(f"Improvement with clustering: {improvement:.2f}%")
    
    # Save results
    if cfg['output']['save_results']:
        results = {
            'clustered': clustered_results,
            'nocluster': nocluster_results
        }
        save_results(results, X_test, u_ref, ROOT_DIR / cfg['output']['results_dir'])


if __name__ == "__main__":
    main()
