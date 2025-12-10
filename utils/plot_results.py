#%%
"""Plotting utilities for GP PDE solver results."""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_results(results_path):
    return np.load(results_path)


def plot_convergence_heat(results, save_path=None):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    ax1.plot(results['al_points'], results['al_errors'], 'o-', color='blue', linewidth=2, label='Active Learning')
    ax1.axhline(y=results['random_error'], color='r', linestyle='--', label='Random Sampling')
    ax1.set_xlabel('Number of Training Points')
    ax1.set_ylabel('Mean Absolute Error')
    ax1.set_title('Error Convergence')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(results['al_points'], results['al_variances'], 'o-', color='green', linewidth=2, label='Active Learning')
    ax2.axhline(y=results['random_variance'], color='r', linestyle='--', label='Random Sampling')
    ax2.set_xlabel('Number of Training Points')
    ax2.set_ylabel('Mean Posterior Variance')
    ax2.set_title('Uncertainty Reduction')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_convergence_poisson(results, save_path=None):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    ax1.plot(results['clustered_points'], results['clustered_errors'], 'o-', color='blue', linewidth=2, label='With Clustering')
    ax1.plot(results['nocluster_points'], results['nocluster_errors'], 's--', color='red', linewidth=2, label='Without Clustering')
    ax1.set_xlabel('Number of Training Points')
    ax1.set_ylabel('Mean Absolute Error')
    ax1.set_title('Error Convergence: Clustering vs No Clustering')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(results['clustered_points'], results['clustered_variances'], 'o-', color='blue', linewidth=2, label='With Clustering')
    ax2.plot(results['nocluster_points'], results['nocluster_variances'], 's--', color='red', linewidth=2, label='Without Clustering')
    ax2.set_xlabel('Number of Training Points')
    ax2.set_ylabel('Mean Posterior Variance')
    ax2.set_title('Uncertainty Reduction')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_solution_poisson(results, save_path=None):
    X_test = results['X_test']
    x1, x2 = X_test[:, 0], X_test[:, 1]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: Reference, Clustered, No Cluster predictions
    im0 = axes[0, 0].tricontourf(x1, x2, results['u_ref'], levels=50, cmap='viridis')
    plt.colorbar(im0, ax=axes[0, 0])
    axes[0, 0].set_title('Reference Solution')
    axes[0, 0].set_xlabel('$x_1$')
    axes[0, 0].set_ylabel('$x_2$')
    axes[0, 0].set_aspect('equal')
    
    im1 = axes[0, 1].tricontourf(x1, x2, results['clustered_pred'].reshape(-1), levels=50, cmap='viridis')
    plt.colorbar(im1, ax=axes[0, 1])
    axes[0, 1].scatter(results['clustered_Xi'][:, 0], results['clustered_Xi'][:, 1], c='white', s=5, alpha=0.7)
    axes[0, 1].set_title(f'With Clustering (MAE: {results["clustered_errors"][-1]:.5f})')
    axes[0, 1].set_xlabel('$x_1$')
    axes[0, 1].set_aspect('equal')
    
    im2 = axes[0, 2].tricontourf(x1, x2, results['nocluster_pred'].reshape(-1), levels=50, cmap='viridis')
    plt.colorbar(im2, ax=axes[0, 2])
    axes[0, 2].scatter(results['nocluster_Xi'][:, 0], results['nocluster_Xi'][:, 1], c='white', s=5, alpha=0.7)
    axes[0, 2].set_title(f'Without Clustering (MAE: {results["nocluster_errors"][-1]:.5f})')
    axes[0, 2].set_xlabel('$x_1$')
    axes[0, 2].set_aspect('equal')
    
    # Row 2: Errors and Variances
    error_clustered = np.abs(results['clustered_pred'].reshape(-1) - results['u_ref'])
    error_nocluster = np.abs(results['nocluster_pred'].reshape(-1) - results['u_ref'])
    
    im3 = axes[1, 0].tricontourf(x1, x2, error_clustered, levels=50, cmap='coolwarm')
    plt.colorbar(im3, ax=axes[1, 0])
    axes[1, 0].set_title('Error (With Clustering)')
    axes[1, 0].set_xlabel('$x_1$')
    axes[1, 0].set_ylabel('$x_2$')
    axes[1, 0].set_aspect('equal')
    
    im4 = axes[1, 1].tricontourf(x1, x2, error_nocluster, levels=50, cmap='coolwarm')
    plt.colorbar(im4, ax=axes[1, 1])
    axes[1, 1].set_title('Error (Without Clustering)')
    axes[1, 1].set_xlabel('$x_1$')
    axes[1, 1].set_aspect('equal')
    
    im5 = axes[1, 2].tricontourf(x1, x2, results['clustered_var'], levels=50, cmap='plasma')
    plt.colorbar(im5, ax=axes[1, 2])
    axes[1, 2].set_title('Variance (With Clustering)')
    axes[1, 2].set_xlabel('$x_1$')
    axes[1, 2].set_aspect('equal')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_solution_comparison(results, n_grid=20, save_path=None):
    X_test = results['X_test']
    x = X_test[:, 0].reshape(n_grid, n_grid)
    t = X_test[:, 1].reshape(n_grid, n_grid)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Reference solution
    im0 = axes[0].contourf(x, t, results['u_ref'].reshape(n_grid, n_grid), levels=50, cmap='viridis')
    plt.colorbar(im0, ax=axes[0])
    axes[0].set_title('Reference Solution')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('t')
    
    # Random sampling
    im1 = axes[1].contourf(x, t, results['random_pred'].reshape(n_grid, n_grid), levels=50, cmap='viridis')
    plt.colorbar(im1, ax=axes[1])
    axes[1].scatter(results['random_Xi'][:, 0], results['random_Xi'][:, 1], c='red', s=10, alpha=0.5)
    axes[1].set_title(f'Random (MAE: {results["random_error"]:.4f})')
    axes[1].set_xlabel('x')
    
    # Active learning
    im2 = axes[2].contourf(x, t, results['al_pred'].reshape(n_grid, n_grid), levels=50, cmap='viridis')
    plt.colorbar(im2, ax=axes[2])
    axes[2].scatter(results['al_Xi'][:, 0], results['al_Xi'][:, 1], c='red', s=10, alpha=0.5)
    axes[2].set_title(f'Active Learning (MAE: {results["al_errors"][-1]:.4f})')
    axes[2].set_xlabel('x')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=str, required=True, help='Path to results.npz')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    parser.add_argument('--type', type=str, default='poisson', choices=['poisson', 'heat'], help='PDE type')
    args = parser.parse_args()
    
    results = load_results(args.results)
    output_dir = Path(args.output) if args.output else Path(args.results).parent
    
    if args.type == 'poisson':
        plot_convergence_poisson(results, save_path=output_dir / 'convergence.png')
        plot_solution_poisson(results, save_path=output_dir / 'solution_comparison.png')
    else:
        plot_convergence_heat(results, save_path=output_dir / 'convergence.png')
        plot_solution_comparison(results, save_path=output_dir / 'solution_comparison.png')
    
    plt.show()


if __name__ == "__main__":
    main()