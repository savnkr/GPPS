"""
3D Poisson equation: GP solver without active learning.
PDE: ∇²u = f on [0,1]³, u=0 on boundary.
Exact: u(x,y,z) = sin(πx)sin(πy)sin(πz), f = -3π²sin(πx)sin(πy)sin(πz).
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
    generate_cuboid_domain, sobol_cuboid_sampling
)


def exact_solution(X):
    return np.sin(np.pi * X[:, 0]) * np.sin(np.pi * X[:, 1]) * np.sin(np.pi * X[:, 2])


def source_term(X):
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


def make_cube_slices(values, coords, title, colorscale='Viridis', n_grid=30):
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

    xg = np.linspace(0, 1, n_grid)
    yg = np.linspace(0, 1, n_grid)
    zg = np.linspace(0, 1, n_grid)
    Xg, Yg, Zg = np.meshgrid(xg, yg, zg, indexing='ij')

    from scipy.interpolate import griddata
    vol = griddata(
        np.column_stack([x, y, z]), values,
        (Xg, Yg, Zg), method='linear', fill_value=0.0
    )

    fig = go.Figure()

    fig.add_trace(go.Volume(
        x=Xg.flatten(), y=Yg.flatten(), z=Zg.flatten(),
        value=vol.flatten(),
        isomin=float(np.nanmin(values)),
        isomax=float(np.nanmax(values)),
        opacity=0.1,
        surface_count=15,
        colorscale=colorscale,
        colorbar=dict(title=dict(text='', font=dict(size=12)), len=0.75),
        caps=dict(x_show=True, y_show=True, z_show=True),
    ))

    slices_data = [
        ('x', 0.5, Yg[n_grid // 2, :, :], Zg[n_grid // 2, :, :], vol[n_grid // 2, :, :]),
        ('y', 0.5, Xg[:, n_grid // 2, :], Zg[:, n_grid // 2, :], vol[:, n_grid // 2, :]),
        ('z', 0.5, Xg[:, :, n_grid // 2], Yg[:, :, n_grid // 2], vol[:, :, n_grid // 2]),
    ]

    for axis, pos, s1, s2, sval in slices_data:
        if axis == 'x':
            sx, sy, sz = np.full_like(s1, pos), s1, s2
        elif axis == 'y':
            sx, sy, sz = s1, np.full_like(s1, pos), s2
        else:
            sx, sy, sz = s1, s2, np.full_like(s1, pos)

        fig.add_trace(go.Surface(
            x=sx, y=sy, z=sz,
            surfacecolor=sval,
            colorscale=colorscale,
            showscale=False,
            opacity=0.9,
            cmin=float(np.nanmin(values)),
            cmax=float(np.nanmax(values)),
        ))

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=16)),
        scene=dict(
            xaxis_title='x', yaxis_title='y', zaxis_title='z',
            xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]), zaxis=dict(range=[0, 1]),
            aspectmode='cube',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        width=450, height=400,
    )
    return fig


def create_row_plot(X_test_np, ground_truth, prediction, error, std_dev, save_path='row_poisson_3d_gp.html'):
    fig = make_subplots(
        rows=1, cols=4,
        specs=[[{'type': 'scene'}, {'type': 'scene'}, {'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=['Ground Truth', 'Mean Prediction', 'Absolute Error', 'Standard Deviation'],
        horizontal_spacing=0.02,
    )

    n_grid = 25
    from scipy.interpolate import griddata
    xg = np.linspace(0, 1, n_grid)
    Xg, Yg, Zg = np.meshgrid(xg, xg, xg, indexing='ij')
    grid_pts = (Xg, Yg, Zg)
    coords = np.column_stack([X_test_np[:, 0], X_test_np[:, 1], X_test_np[:, 2]])

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
            ('x', 0.5, Yg[mid, :, :], Zg[mid, :, :], vol[mid, :, :]),
            ('y', 0.5, Xg[:, mid, :], Zg[:, mid, :], vol[:, mid, :]),
            ('z', 0.5, Xg[:, :, mid], Yg[:, :, mid], vol[:, :, mid]),
        ]

        vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
        show_scale = (col == 1) or (col == 3) or (col == 4)

        for axis, pos, s1, s2, sval in slice_configs:
            if axis == 'x':
                sx, sy, sz = np.full_like(s1, pos), s1, s2
            elif axis == 'y':
                sx, sy, sz = s1, np.full_like(s1, pos), s2
            else:
                sx, sy, sz = s1, s2, np.full_like(s1, pos)

            fig.add_trace(go.Surface(
                x=sx, y=sy, z=sz,
                surfacecolor=sval,
                colorscale=cscale,
                showscale=show_scale and (axis == 'x'),
                opacity=0.9,
                cmin=vmin, cmax=vmax,
                colorbar=dict(len=0.6, thickness=15) if show_scale and axis == 'x' else None,
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
        title=dict(text='3D Poisson — GP (No Active Learning)', x=0.5, font=dict(size=18)),
    )

    _write_plotly_exports(fig, save_path, width=1600, height=400)
    return fig


def run(n_interior_per_side=6, n_boundary_per_side=8, n_test=500,
        lengthscale=0.531, variance=0.678, save_dir='results/poisson_3d'):

    os.makedirs(save_dir, exist_ok=True)

    Xi, Xb, f_Xi, g_Xb = generate_cuboid_domain(
        n_interior_per_side=n_interior_per_side,
        n_boundary_per_side=n_boundary_per_side,
    )
    Xi, Xb = Xi.to(torch.float64), Xb.to(torch.float64)
    f_Xi, g_Xb = f_Xi.to(torch.float64), g_Xb.to(torch.float64)

    print(f"Interior points: {Xi.shape[0]}, Boundary points: {Xb.shape[0]}")

    kernel = RBFKernel(lengthscale=lengthscale, variance=variance)
    operators = PoissonOperators(kernel)
    solver = GPPoissonSolver(kernel, operators)

    print("Computing covariance matrix...")
    C_full = solver.compute_covariance_matrix(Xi, Xb)
    C_inv = solver.compute_inverse(C_full)

    y_obs = torch.cat((f_Xi, g_Xb), dim=0).to(torch.float64)

    X_test = sobol_cuboid_sampling(n_test, domain_bounds=(0.05, 0.95)).to(torch.float64)

    print("Computing posterior predictions...")
    post_mean = solver.posterior_mean(X_test, Xi, Xb, C_inv, y_obs).detach()
    post_cov = solver.posterior_covariance(X_test, X_test.clone(), Xi, Xb, C_inv).detach()
    post_std = torch.sqrt(torch.clamp(torch.diag(post_cov), min=1e-12)).cpu().numpy()

    X_test_np = X_test.cpu().numpy()
    pred_np = post_mean.cpu().numpy().reshape(-1)
    truth = exact_solution(X_test_np)
    error = np.abs(pred_np - truth)

    mae = np.mean(error)
    rel_err = np.mean(error) / np.mean(np.abs(truth))
    print(f"Mean Absolute Error: {mae:.6f}")
    print(f"Relative Error: {rel_err:.6f}")
    print(f"Max Error: {np.max(error):.6f}")
    print(f"Mean Std Dev: {np.mean(post_std):.6f}")

    np.savez(
        os.path.join(save_dir, 'gp_no_al_results.npz'),
        X_test=X_test_np,
        prediction=pred_np,
        ground_truth=truth,
        error=error,
        std_dev=post_std,
        Xi=Xi.cpu().numpy(), Xb=Xb.cpu().numpy(),
    )

    fig = create_row_plot(
        X_test_np, truth, pred_np, error, post_std,
        save_path=os.path.join(save_dir, 'row_gp_poisson_3d.html'),
    )

    for name, vals, cscale in [
        ('ground_truth', truth, 'Viridis'),
        ('mean_prediction', pred_np, 'Viridis'),
        ('absolute_error', error, 'Blues'),
        ('std_deviation', post_std, 'Plasma'),
    ]:
        f = make_cube_slices(vals, X_test_np, name.replace('_', ' ').title(), cscale)
        _write_plotly_exports(
            f,
            os.path.join(save_dir, f'gp_{name}.html'),
            width=450,
            height=400,
        )

    print(f"\nAll results saved to {save_dir}/")
    return fig


if __name__ == '__main__':
    run()
