"""
3D Poisson equation: GP solver with clustering-based active learning.
PDE: ∇²u = f on [0,1]³, u=0 on boundary.
Exact: u(x,y,z) = sin(πx)sin(πy)sin(πz), f = -3π²sin(πx)sin(πy)sin(πz).

Produces the row_al_cluster_poisson_3d plot (4-panel: Ground Truth, Mean Prediction, Absolute Error, Std Dev).
"""
import sys
import os
import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src import (
    RBFKernel, PoissonOperators, GPPoissonSolver,
    generate_cuboid_boundary, sobol_cuboid_sampling,
    adaptive_sampling_poisson,
)


def exact_solution(X):
    return np.sin(np.pi * X[:, 0]) * np.sin(np.pi * X[:, 1]) * np.sin(np.pi * X[:, 2])


def source_term_torch(X):
    return -3 * (np.pi ** 2) * torch.sin(np.pi * X[:, 0]) * torch.sin(np.pi * X[:, 1]) * torch.sin(np.pi * X[:, 2])


def _write_plotly_exports(fig, html_path, *, width, height, png_scale=3, pdf_scale=2):
    """Always save HTML; best-effort static export when Chrome/Kaleido works."""
    fig.write_html(html_path)
    try:
        fig.write_image(html_path.replace('.html', '.pdf'), format='pdf',
                        width=width, height=height, scale=pdf_scale)
        fig.write_image(html_path.replace('.html', '.png'), format='png',
                        width=width, height=height, scale=png_scale)
        print(f"Saved: {html_path} (+pdf, +png)")
    except Exception as exc:
        print(f"Saved: {html_path}")
        print(f"Warning: static Plotly export skipped: {exc}")


def create_row_plot(X_test_np, ground_truth, prediction, error, std_dev,
                    save_path='row_al_cluster_poisson_3d.html', title_text='3D Poisson — GP + Active Learning (Clustering)'):
    fig = make_subplots(
        rows=1, cols=4,
        specs=[[{'type': 'scene'}] * 4],
        subplot_titles=['Ground Truth', 'Mean Prediction', 'Absolute Error', 'Standard Deviation'],
        horizontal_spacing=0.02,
    )

    n_grid = 25
    from scipy.interpolate import griddata
    xg = np.linspace(0, 1, n_grid)
    Xg, Yg, Zg = np.meshgrid(xg, xg, xg, indexing='ij')
    grid_pts = (Xg, Yg, Zg)
    coords = X_test_np[:, :3]

    datasets = [
        (ground_truth, 'Viridis'),
        (prediction, 'Viridis'),
        (error, 'Blues'),
        (std_dev, 'Plasma'),
    ]

    for col, (vals, cscale) in enumerate(datasets, 1):
        vol = griddata(coords, vals, grid_pts, method='linear', fill_value=0.0)
        mid = n_grid // 2

        slice_configs = [
            ('x', Yg[mid, :, :], Zg[mid, :, :], vol[mid, :, :]),
            ('y', Xg[:, mid, :], Zg[:, mid, :], vol[:, mid, :]),
            ('z', Xg[:, :, mid], Yg[:, :, mid], vol[:, :, mid]),
        ]

        vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))

        for i, (axis, s1, s2, sval) in enumerate(slice_configs):
            if axis == 'x':
                sx, sy, sz = np.full_like(s1, 0.5), s1, s2
            elif axis == 'y':
                sx, sy, sz = s1, np.full_like(s1, 0.5), s2
            else:
                sx, sy, sz = s1, s2, np.full_like(s1, 0.5)

            fig.add_trace(go.Surface(
                x=sx, y=sy, z=sz,
                surfacecolor=sval,
                colorscale=cscale,
                showscale=(i == 0),
                opacity=0.9,
                cmin=vmin, cmax=vmax,
                colorbar=dict(len=0.6, thickness=15) if i == 0 else None,
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
    _write_plotly_exports(fig, save_path, width=1600, height=400)
    return fig


def create_convergence_plot(cluster_pts, cluster_errors, cluster_vars,
                            nocluster_pts, nocluster_errors, nocluster_vars,
                            save_path='convergence_3d.html'):
    fig = make_subplots(rows=1, cols=2, subplot_titles=['Mean Absolute Error', 'Mean Posterior Variance'])

    fig.add_trace(go.Scatter(
        x=cluster_pts, y=cluster_errors, mode='lines+markers',
        name='AL + Clustering', line=dict(color='#2ca02c', width=2),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=nocluster_pts, y=nocluster_errors, mode='lines+markers',
        name='AL (no clustering)', line=dict(color='#1f77b4', width=2, dash='dash'),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=cluster_pts, y=cluster_vars, mode='lines+markers',
        name='AL + Clustering', line=dict(color='#2ca02c', width=2), showlegend=False,
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=nocluster_pts, y=nocluster_vars, mode='lines+markers',
        name='AL (no clustering)', line=dict(color='#1f77b4', width=2, dash='dash'), showlegend=False,
    ), row=1, col=2)

    fig.update_xaxes(title_text='Number of Training Points', row=1, col=1)
    fig.update_xaxes(title_text='Number of Training Points', row=1, col=2)
    fig.update_yaxes(title_text='MAE', row=1, col=1)
    fig.update_yaxes(title_text='Variance', row=1, col=2)

    fig.update_layout(
        width=1000, height=400,
        margin=dict(l=60, r=20, t=50, b=50),
        title=dict(text='Active Learning Convergence — 3D Poisson', x=0.5),
    )
    _write_plotly_exports(fig, save_path, width=1000, height=400)
    return fig


def run_gp_prediction(solver, Xi, Xb, f_Xi, g_Xb, X_test, actual):
    C_full = solver.compute_covariance_matrix(Xi, Xb)
    C_inv = solver.compute_inverse(C_full)
    y_obs = torch.cat((f_Xi, g_Xb), dim=0)

    post_mean = solver.posterior_mean(X_test, Xi, Xb, C_inv, y_obs).detach()
    post_cov = solver.posterior_covariance(X_test, X_test.clone(), Xi, Xb, C_inv).detach()
    post_var = torch.diag(post_cov).cpu().numpy()
    post_std = np.sqrt(np.clip(post_var, 1e-12, None))

    pred = post_mean.cpu().numpy().reshape(-1)
    err = np.abs(pred - actual)
    return pred, err, post_std, post_var, C_inv, y_obs


def run(n_initial=20, n_final=80, n_iterations=10, n_test=500,
        n_boundary_per_side=8, lengthscale=0.531, variance=0.678,
        save_dir='results/poisson_3d'):

    os.makedirs(save_dir, exist_ok=True)
    n_per_iter = max((n_final - n_initial) // n_iterations, 1)

    Xb = generate_cuboid_boundary(n_boundary_per_side).to(torch.float64)
    g_Xb = torch.zeros(Xb.shape[0], 1, dtype=torch.float64)
    print(f"Boundary points: {Xb.shape[0]}")

    X_test = sobol_cuboid_sampling(n_test, domain_bounds=(0.05, 0.95)).to(torch.float64)
    actual = exact_solution(X_test.numpy())

    X_pool = sobol_cuboid_sampling(2000, domain_bounds=(0.05, 0.95), seed=123).to(torch.float64)

    kernel = RBFKernel(lengthscale=lengthscale, variance=variance)
    operators = PoissonOperators(kernel)
    solver = GPPoissonSolver(kernel, operators)

    Xi_init = sobol_cuboid_sampling(n_initial, domain_bounds=(0.05, 0.95)).to(torch.float64)
    f_Xi_init = source_term_torch(Xi_init).reshape(-1, 1).to(torch.float64)

    # -- AL with clustering --
    Xi_cl = Xi_init.clone()
    f_cl = f_Xi_init.clone()
    cl_pts, cl_errs, cl_vars = [n_initial], [], []

    pred, err, std, var, C_inv, y_obs_cl = run_gp_prediction(solver, Xi_cl, Xb, f_cl, g_Xb, X_test, actual)
    cl_errs.append(np.mean(err))
    cl_vars.append(np.mean(var))

    # -- AL without clustering --
    Xi_nc = Xi_init.clone()
    f_nc = f_Xi_init.clone()
    nc_pts, nc_errs, nc_vars = [n_initial], [], []

    pred_nc, err_nc, _, var_nc, _, y_obs_nc = run_gp_prediction(solver, Xi_nc, Xb, f_nc, g_Xb, X_test, actual)
    nc_errs.append(np.mean(err_nc))
    nc_vars.append(np.mean(var_nc))

    for it in range(n_iterations):
        print(f"\nIteration {it + 1}/{n_iterations}")

        # With clustering
        new_cl = adaptive_sampling_poisson(
            solver, X_pool, Xi_cl, Xb, y_obs_cl,
            na=n_per_iter, exclusion_radius=0.05, use_clustering=True
        )
        if new_cl.shape[0] > 0:
            Xi_cl = torch.cat([Xi_cl, new_cl], dim=0)
            f_cl = source_term_torch(Xi_cl).reshape(-1, 1).to(torch.float64)
            _, err, _, var, _, y_obs_cl = run_gp_prediction(solver, Xi_cl, Xb, f_cl, g_Xb, X_test, actual)
            cl_errs.append(np.mean(err))
            cl_vars.append(np.mean(var))
        cl_pts.append(Xi_cl.shape[0])

        # Without clustering
        new_nc = adaptive_sampling_poisson(
            solver, X_pool, Xi_nc, Xb, y_obs_nc,
            na=n_per_iter, exclusion_radius=0.05, use_clustering=False
        )
        if new_nc.shape[0] > 0:
            Xi_nc = torch.cat([Xi_nc, new_nc], dim=0)
            f_nc = source_term_torch(Xi_nc).reshape(-1, 1).to(torch.float64)
            _, err_nc, _, var_nc, _, y_obs_nc = run_gp_prediction(solver, Xi_nc, Xb, f_nc, g_Xb, X_test, actual)
            nc_errs.append(np.mean(err_nc))
            nc_vars.append(np.mean(var_nc))
        nc_pts.append(Xi_nc.shape[0])

        print(f"  Clustering  — pts: {Xi_cl.shape[0]}, MAE: {cl_errs[-1]:.6f}")
        print(f"  No cluster  — pts: {Xi_nc.shape[0]}, MAE: {nc_errs[-1]:.6f}")

    # Final predictions (clustered)
    pred_final, err_final, std_final, _, _, _ = run_gp_prediction(solver, Xi_cl, Xb, f_cl, g_Xb, X_test, actual)

    print(f"\n--- Final Results ---")
    print(f"AL + Clustering ({Xi_cl.shape[0]} pts): MAE = {cl_errs[-1]:.6f}")
    print(f"AL no clustering ({Xi_nc.shape[0]} pts): MAE = {nc_errs[-1]:.6f}")

    # Save results
    np.savez(
        os.path.join(save_dir, 'al_comparison_results.npz'),
        X_test=X_test.cpu().numpy(),
        ground_truth=actual,
        prediction_cluster=pred_final,
        error_cluster=err_final,
        std_cluster=std_final,
        cl_pts=cl_pts, cl_errs=cl_errs, cl_vars=cl_vars,
        nc_pts=nc_pts, nc_errs=nc_errs, nc_vars=nc_vars,
        Xi_cluster=Xi_cl.cpu().numpy(), Xi_nocluster=Xi_nc.cpu().numpy(),
        Xb=Xb.cpu().numpy(),
    )

    # Row plot (the main plot: row_al_cluster_poisson_3d)
    create_row_plot(
        X_test.cpu().numpy(), actual, pred_final, err_final, std_final,
        save_path=os.path.join(save_dir, 'row_al_cluster_poisson_3d.html'),
    )

    # Convergence plot
    create_convergence_plot(
        cl_pts, cl_errs, cl_vars,
        nc_pts, nc_errs, nc_vars,
        save_path=os.path.join(save_dir, 'convergence_3d.html'),
    )

    print(f"\nAll results saved to {save_dir}/")


if __name__ == '__main__':
    run()
