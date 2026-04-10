"""
3D Poisson equation: GP solver with active learning + SDD (Stochastic Dual Descent).
PDE: nabla^2 u = f on [0,1]^3, u=0 on boundary.
Exact: u(x,y,z) = sin(pi*x)sin(pi*y)sin(pi*z), f = -3*pi^2*sin(pi*x)sin(pi*y)sin(pi*z).

Combines clustering-based active learning with SDD inference to avoid O(N^3) inversion.
Produces row_al_sdd_cluster_poisson_3d plot (4-panel) and convergence comparison.
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

import torch
import numpy as np
import yaml
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src import (
    RBFKernel, PoissonOperators, GPPoissonSolver,
    generate_cuboid_boundary, sobol_cuboid_sampling,
    adaptive_sampling_poisson_sdd, train_sdd,
)


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)['poisson_3d']


def exact_solution(X):
    return np.sin(np.pi * X[:, 0]) * np.sin(np.pi * X[:, 1]) * np.sin(np.pi * X[:, 2])


def source_term_torch(X):
    return -3 * (np.pi ** 2) * torch.sin(np.pi * X[:, 0]) * \
           torch.sin(np.pi * X[:, 1]) * torch.sin(np.pi * X[:, 2])


def build_sdd_cfg(cfg):
    """Build SDD config dict from YAML config."""
    return {
        'batch_size': cfg['sdd']['batch_size'],
        'beta': cfg['sdd']['beta'],
        'rho': cfg['sdd']['rho'],
        'jitter': cfg['training']['jitter'],
    }


def sdd_posterior_stats(solver, X_test, Xi, Xb, C_full, y_obs, sdd_cfg, epochs_mean, epochs_var):
    """Compute posterior mean and diagonal variance using SDD."""
    cfg_mean = {**sdd_cfg, 'num_epochs': epochs_mean}
    A_mean = train_sdd(C_full, y_obs, **cfg_mean, verbose=False)

    cov_vec = solver.compute_covariance_vector(X_test, Xi, Xb).detach()
    pred_mean = (cov_vec @ A_mean).detach()

    cfg_var = {**sdd_cfg, 'num_epochs': epochs_var}
    A_var = train_sdd(C_full, cov_vec.T, **cfg_var, verbose=False)

    base_var = solver.kernel.variance.detach() ** 2
    correction = torch.sum(cov_vec * A_var.T, dim=1)
    pred_var = torch.clamp(base_var - correction, min=1e-10)

    return pred_mean, pred_var.numpy()


def run_gp_prediction_sdd(solver, Xi, Xb, f_Xi, g_Xb, X_test, actual, sdd_cfg,
                          epochs_mean, epochs_var):
    """Run GP prediction using SDD and return stats."""
    y_obs = torch.cat((f_Xi, g_Xb), dim=0)
    C_full = solver.compute_covariance_matrix(Xi, Xb)

    pred_mean, pred_var = sdd_posterior_stats(
        solver, X_test, Xi, Xb, C_full, y_obs, sdd_cfg, epochs_mean, epochs_var
    )
    pred_std = np.sqrt(np.clip(pred_var, 1e-12, None))
    pred_np = pred_mean.numpy().reshape(-1)
    err = np.abs(pred_np - actual)

    return pred_np, err, pred_std, pred_var, C_full, y_obs


def create_row_plot(X_test_np, ground_truth, prediction, error, std_dev,
                    save_path='row_al_sdd_cluster_poisson_3d.html',
                    title_text='3D Poisson — GP + AL + SDD (Clustering)'):
    """Solid cube with a single mid-plane slice visible inside."""
    from scipy.interpolate import griddata

    fig = make_subplots(
        rows=1, cols=4,
        specs=[[{'type': 'scene'}] * 4],
        subplot_titles=['Ground Truth', 'Mean Prediction', 'Absolute Error', 'Standard Deviation'],
        horizontal_spacing=0.02,
    )

    n_grid = 30
    xg = np.linspace(0, 1, n_grid)
    yg = np.linspace(0, 1, n_grid)
    zg = np.linspace(0, 1, n_grid)
    Xg, Yg, Zg = np.meshgrid(xg, yg, zg, indexing='ij')
    coords = X_test_np[:, :3]

    # 2D grids for cube faces
    X2, Y2 = np.meshgrid(xg, yg, indexing='ij')

    datasets = [
        (ground_truth, 'Viridis'),
        (prediction, 'Viridis'),
        (error, 'Blues'),
        (std_dev, 'Plasma'),
    ]

    for col, (vals, cscale) in enumerate(datasets, 1):
        vol = griddata(coords, vals, (Xg, Yg, Zg), method='linear', fill_value=0.0)
        vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
        mid = n_grid // 2

        # 6 cube faces: (x_coords, y_coords, z_coords, surfacecolor)
        faces = [
            (X2, Y2, np.zeros_like(X2), vol[:, :, 0]),       # z=0 (bottom)
            (X2, Y2, np.ones_like(X2), vol[:, :, -1]),        # z=1 (top)
            (X2, np.zeros_like(X2), Y2, vol[:, 0, :]),        # y=0 (front)
            (X2, np.ones_like(X2), Y2, vol[:, -1, :]),        # y=1 (back)
            (np.zeros_like(X2), X2, Y2, vol[0, :, :]),        # x=0 (left)
            (np.ones_like(X2), X2, Y2, vol[-1, :, :]),        # x=1 (right)
        ]

        # Mid-plane slice at z=0.5
        midplane = (X2, Y2, np.full_like(X2, zg[mid]), vol[:, :, mid])

        # Draw cube faces (semi-transparent so mid-plane is visible inside)
        for i, (fx, fy, fz, fval) in enumerate(faces):
            fig.add_trace(go.Surface(
                x=fx, y=fy, z=fz,
                surfacecolor=fval,
                colorscale=cscale,
                cmin=vmin, cmax=vmax,
                opacity=0.55,
                showscale=False,
            ), row=1, col=col)

        # Draw mid-plane slice (fully opaque, visible through semi-transparent faces)
        mx, my, mz, mval = midplane
        fig.add_trace(go.Surface(
            x=mx, y=my, z=mz,
            surfacecolor=mval,
            colorscale=cscale,
            cmin=vmin, cmax=vmax,
            opacity=1.0,
            showscale=True,
            colorbar=dict(len=0.6, thickness=15),
        ), row=1, col=col)

    scene_cfg = dict(
        xaxis=dict(range=[0, 1], title=''),
        yaxis=dict(range=[0, 1], title=''),
        zaxis=dict(range=[0, 1], title=''),
        aspectmode='cube',
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
    )
    fig.update_layout(
        scene=scene_cfg, scene2=scene_cfg, scene3=scene_cfg, scene4=scene_cfg,
        width=1600, height=400,
        margin=dict(l=10, r=10, t=40, b=10),
        title=dict(text=title_text, x=0.5, font=dict(size=18)),
    )
    fig.write_html(save_path)
    fig.write_image(save_path.replace('.html', '.pdf'), format='pdf', width=1600, height=400, scale=2)
    fig.write_image(save_path.replace('.html', '.png'), format='png', width=1600, height=400, scale=3)
    print(f"Saved: {save_path} (+pdf, +png)")
    return fig


def create_convergence_plot(cluster_pts, cluster_errors, cluster_vars,
                            nocluster_pts, nocluster_errors, nocluster_vars,
                            save_path='convergence_3d_sdd.html'):
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=['Mean Absolute Error', 'Mean Posterior Variance'])

    fig.add_trace(go.Scatter(
        x=cluster_pts, y=cluster_errors, mode='lines+markers',
        name='AL+SDD + Clustering', line=dict(color='#2ca02c', width=2),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=nocluster_pts, y=nocluster_errors, mode='lines+markers',
        name='AL+SDD (no clustering)', line=dict(color='#1f77b4', width=2, dash='dash'),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=cluster_pts, y=cluster_vars, mode='lines+markers',
        name='AL+SDD + Clustering', line=dict(color='#2ca02c', width=2), showlegend=False,
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=nocluster_pts, y=nocluster_vars, mode='lines+markers',
        name='AL+SDD (no clustering)', line=dict(color='#1f77b4', width=2, dash='dash'),
        showlegend=False,
    ), row=1, col=2)

    fig.update_xaxes(title_text='Number of Training Points', row=1, col=1)
    fig.update_xaxes(title_text='Number of Training Points', row=1, col=2)
    fig.update_yaxes(title_text='MAE', row=1, col=1)
    fig.update_yaxes(title_text='Variance', row=1, col=2)

    fig.update_layout(
        width=1000, height=400,
        margin=dict(l=60, r=20, t=50, b=50),
        title=dict(text='AL + SDD Convergence — 3D Poisson', x=0.5),
    )
    fig.write_html(save_path)
    fig.write_image(save_path.replace('.html', '.pdf'), format='pdf', width=1000, height=400, scale=2)
    fig.write_image(save_path.replace('.html', '.png'), format='png', width=1000, height=400, scale=3)
    print(f"Saved: {save_path} (+pdf, +png)")
    return fig


def run(cfg):
    save_dir = ROOT_DIR / cfg['output']['results_dir']
    save_dir.mkdir(parents=True, exist_ok=True)

    n_initial = cfg['training']['n_initial']
    n_final = cfg['training']['n_final']
    n_iterations = cfg['training']['n_iterations']
    n_per_iter = max((n_final - n_initial) // n_iterations, 1)
    sdd_cfg = build_sdd_cfg(cfg)

    Xb = generate_cuboid_boundary(cfg['training']['n_boundary_per_side']).to(torch.float64)
    g_Xb = torch.zeros(Xb.shape[0], 1, dtype=torch.float64)
    print(f"Boundary points: {Xb.shape[0]}")

    X_test = sobol_cuboid_sampling(cfg['training']['n_test'], domain_bounds=(0.05, 0.95)).to(torch.float64)
    actual = exact_solution(X_test.numpy())

    X_pool = sobol_cuboid_sampling(cfg['training']['n_pool'], domain_bounds=(0.05, 0.95), seed=123).to(torch.float64)

    kernel = RBFKernel(lengthscale=cfg['kernel']['lengthscale'], variance=cfg['kernel']['variance'])
    operators = PoissonOperators(kernel)
    solver = GPPoissonSolver(kernel, operators)

    Xi_init = sobol_cuboid_sampling(n_initial, domain_bounds=(0.05, 0.95)).to(torch.float64)
    f_Xi_init = source_term_torch(Xi_init).reshape(-1, 1).to(torch.float64)

    epochs_mean = cfg['sdd']['epochs_mean']
    epochs_var = cfg['sdd']['epochs_var']

    # -- AL+SDD with clustering --
    Xi_cl = Xi_init.clone()
    f_cl = f_Xi_init.clone()
    cl_pts, cl_errs, cl_vars = [n_initial], [], []

    pred, err, std, var, C_full_cl, y_obs_cl = run_gp_prediction_sdd(
        solver, Xi_cl, Xb, f_cl, g_Xb, X_test, actual, sdd_cfg,
        epochs_mean, epochs_var
    )
    cl_errs.append(np.mean(err))
    cl_vars.append(np.mean(var))
    print(f"Initial — pts: {n_initial}, MAE: {cl_errs[-1]:.6f}")

    # -- AL+SDD without clustering --
    Xi_nc = Xi_init.clone()
    f_nc = f_Xi_init.clone()
    nc_pts, nc_errs, nc_vars = [n_initial], [], []

    pred_nc, err_nc, _, var_nc, C_full_nc, y_obs_nc = run_gp_prediction_sdd(
        solver, Xi_nc, Xb, f_nc, g_Xb, X_test, actual, sdd_cfg,
        epochs_mean, epochs_var
    )
    nc_errs.append(np.mean(err_nc))
    nc_vars.append(np.mean(var_nc))

    for it in range(n_iterations):
        print(f"\nIteration {it + 1}/{n_iterations}")

        # With clustering
        acq_sdd_cfg = {**sdd_cfg, 'num_epochs': epochs_var}
        new_cl = adaptive_sampling_poisson_sdd(
            solver, X_pool, Xi_cl, Xb, y_obs_cl, C_full_cl, acq_sdd_cfg,
            na=n_per_iter, exclusion_radius=cfg['active_learning']['exclusion_radius'],
            use_clustering=cfg['active_learning'].get('use_clustering', True)
        )
        if new_cl.shape[0] > 0:
            Xi_cl = torch.cat([Xi_cl, new_cl], dim=0)
            f_cl = source_term_torch(Xi_cl).reshape(-1, 1).to(torch.float64)
            _, err, _, var, C_full_cl, y_obs_cl = run_gp_prediction_sdd(
                solver, Xi_cl, Xb, f_cl, g_Xb, X_test, actual, sdd_cfg,
                epochs_mean, epochs_var
            )
            cl_errs.append(np.mean(err))
            cl_vars.append(np.mean(var))
        cl_pts.append(Xi_cl.shape[0])

        # Without clustering
        new_nc = adaptive_sampling_poisson_sdd(
            solver, X_pool, Xi_nc, Xb, y_obs_nc, C_full_nc, acq_sdd_cfg,
            na=n_per_iter, exclusion_radius=cfg['active_learning']['exclusion_radius'],
            use_clustering=False
        )
        if new_nc.shape[0] > 0:
            Xi_nc = torch.cat([Xi_nc, new_nc], dim=0)
            f_nc = source_term_torch(Xi_nc).reshape(-1, 1).to(torch.float64)
            _, err_nc, _, var_nc, C_full_nc, y_obs_nc = run_gp_prediction_sdd(
                solver, Xi_nc, Xb, f_nc, g_Xb, X_test, actual, sdd_cfg,
                epochs_mean, epochs_var
            )
            nc_errs.append(np.mean(err_nc))
            nc_vars.append(np.mean(var_nc))
        nc_pts.append(Xi_nc.shape[0])

        print(f"  Clustering  — pts: {Xi_cl.shape[0]}, MAE: {cl_errs[-1]:.6f}")
        print(f"  No cluster  — pts: {Xi_nc.shape[0]}, MAE: {nc_errs[-1]:.6f}")

    # Final predictions (clustered)
    pred_final, err_final, std_final, _, _, _ = run_gp_prediction_sdd(
        solver, Xi_cl, Xb, f_cl, g_Xb, X_test, actual, sdd_cfg,
        epochs_mean, epochs_var
    )

    print(f"\n--- Final Results ---")
    print(f"AL+SDD + Clustering ({Xi_cl.shape[0]} pts): MAE = {cl_errs[-1]:.6f}")
    print(f"AL+SDD no clustering ({Xi_nc.shape[0]} pts): MAE = {nc_errs[-1]:.6f}")

    # Save results
    np.savez(
        str(save_dir / 'al_sdd_comparison_results.npz'),
        X_test=X_test.numpy(),
        ground_truth=actual,
        prediction_cluster=pred_final,
        error_cluster=err_final,
        std_cluster=std_final,
        cl_pts=cl_pts, cl_errs=cl_errs, cl_vars=cl_vars,
        nc_pts=nc_pts, nc_errs=nc_errs, nc_vars=nc_vars,
        Xi_cluster=Xi_cl.numpy(), Xi_nocluster=Xi_nc.numpy(),
        Xb=Xb.numpy(),
    )

    # Row plot
    create_row_plot(
        X_test.numpy(), actual, pred_final, err_final, std_final,
        save_path=str(save_dir / 'row_al_sdd_cluster_poisson_3d.html'),
    )

    # Convergence plot
    create_convergence_plot(
        cl_pts, cl_errs, cl_vars,
        nc_pts, nc_errs, nc_vars,
        save_path=str(save_dir / 'convergence_3d_sdd.html'),
    )

    print(f"\nAll results saved to {save_dir}/")


if __name__ == '__main__':
    config_path = ROOT_DIR / 'configs' / 'config.yaml'
    cfg = load_config(config_path)
    run(cfg)
