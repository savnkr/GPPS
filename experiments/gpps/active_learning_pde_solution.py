import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import os
from matplotlib import rcParams
from sklearn.cluster import KMeans
from scipy.stats import qmc
import pickle
from scipy.interpolate import griddata

# Import the classes from the original script
from gp_pde_solution import (
    setup_matplotlib_for_publication,
    compute_exact_solution,
    RBFKernel2D, 
    PDEBoundaryOperators2D, 
    PosteriorSolver2D, 
    generate_square_domain_data
)

def sobol_rectangle_sampling(n_points, x_range=(0, 1), t_range=(0, 1), seed=42):
    """
    Generate points using Sobol sequence and map them to a rectangular domain.
    
    Args:
        n_points: Number of points to generate
        x_range: Range for spatial dimension (x)
        t_range: Range for time dimension (t)
        seed: Random seed for reproducibility
        
    Returns:
        torch.Tensor of shape [n_points, 2] with points in rectangular domain
    """
    # Initialize Sobol sequence generator
    sampler = qmc.Sobol(d=2, scramble=True, seed=seed)
    
    # Generate Sobol sequence in [0, 1)^2
    sample = sampler.random(n_points)
    
    # Scale to the desired ranges
    sample[:, 0] = sample[:, 0] * (x_range[1] - x_range[0]) + x_range[0]  # Scale x
    sample[:, 1] = sample[:, 1] * (t_range[1] - t_range[0]) + t_range[0]  # Scale t
    
    return torch.tensor(sample, dtype=torch.float64)

def filter_candidates_2d(X_pool, X_train, exclusion_radius=0.05):
    """
    Remove candidates that are too close to existing training points.
    
    Args:
        X_pool: Tensor of candidate points [N, 2]
        X_train: Tensor of existing training points [M, 2]
        exclusion_radius: Minimum distance between any two points
        
    Returns:
        Filtered candidate pool
    """
    # Calculate distances between candidates and training points
    dists = torch.cdist(X_pool, X_train)
    
    # Keep only points that are far enough from all training points
    mask = (dists.min(dim=1)[0] >= exclusion_radius)
    return X_pool[mask]

def ucb_acquisition(mean, variance, kappa=2.0):
    """
    Compute the Upper Confidence Bound acquisition function.
    
    Args:
        mean: Posterior mean predictions
        variance: Posterior variance predictions
        kappa: Exploration-exploitation trade-off parameter
        
    Returns:
        UCB acquisition values
    """
    return mean + kappa * torch.sqrt(variance)

def adaptive_sampling_2d(
    solver,
    X_pool,
    X_train,
    Xd,
    Xn,
    y_train,
    na=5,
    kappa=2.0,
    exclusion_radius=0.05,
    acquisition_function="variance",
    ar_ratio=0.3
):
    """
    Adaptive sampling using uncertainty-based acquisition functions for time-dependent PDEs.
    
    Args:
        solver: GP solver for the PDE
        X_pool: Candidate pool of points [N, 2]
        X_train: Current training points [M, 2]
        Xd: Dirichlet boundary points [P1, 2]
        Xn: Neumann boundary points [P2, 2]
        y_train: Training targets
        na: Number of points to select
        kappa: Exploration parameter for UCB
        exclusion_radius: Minimum distance between selected points
        acquisition_function: Strategy for selecting points
        ar_ratio: Active ratio for preliminary filtering
        
    Returns:
        Selected points to add to training set
    """
    # Recompute covariance matrix and its inverse for current training data
    C_full = solver.compute_covariance_matrix(X_train, Xd, Xn)
    C_inv = torch.inverse(C_full + 1e-6 * torch.eye(C_full.shape[0]))
    
    # Filter candidate pool - remove points too close to training set
    X_pool_filtered = filter_candidates_2d(X_pool, X_train, exclusion_radius)
    
    if X_pool_filtered.shape[0] == 0:
        print("Warning: No valid candidates remain after filtering.")
        return torch.zeros((0, 2), dtype=torch.float64)
    
    # Compute posterior mean and covariance for candidates
    mean = solver.posterior_mean(X_pool_filtered, X_train, Xd, Xn, C_inv, y_train)
    cov_full = solver.posterior_covariance(X_pool_filtered, X_pool_filtered, X_train, Xd, Xn, C_inv)
    variance = torch.diag(cov_full).unsqueeze(1)
    
    # Apply acquisition function
    if acquisition_function == "variance":
        # Pure uncertainty sampling
        acquisition_values = variance
    elif acquisition_function == "ucb":
        # Upper Confidence Bound
        acquisition_values = ucb_acquisition(mean, variance, kappa)
    else:
        # Default to variance
        acquisition_values = variance
        
    # Sort candidates by acquisition value (descending)
    sorted_indices = torch.argsort(acquisition_values.squeeze(), descending=True)
    sorted_candidates = X_pool_filtered[sorted_indices]
    
    # Select top ar_ratio fraction of candidates to cluster
    ar = max(int(ar_ratio * sorted_candidates.size(0)), na*2)
    retained = sorted_candidates[:ar]
    
    # Use KMeans clustering to select diverse points
    retained_np = retained.cpu().detach().numpy()
    kmeans = KMeans(n_clusters=na, random_state=42).fit(retained_np)
    
    # Get cluster centers and find closest points to each center
    centers = kmeans.cluster_centers_
    selected = []
    
    for i in range(na):
        # Find distances to current center
        dists = np.sum((retained_np - centers[i])**2, axis=1)
        idx = np.argmin(dists)
        selected.append(retained[idx])
    
    return torch.stack(selected) if selected else retained[:na]

def run_active_learning_experiment(
    n_initial=20,
    n_final=60,
    n_iterations=8,
    n_test_per_side=20,
    lengthscale=0.6,
    variance=0.02,
    save_dir='results'
):
    """
    Run active learning experiment with Sobol initialization for the time-dependent heat equation.
    
    Args:
        n_initial: Number of initial training points
        n_final: Final number of training points
        n_iterations: Number of active learning iterations
        n_test_per_side: Number of test points per side for evaluation grid
        lengthscale: Initial lengthscale for RBF kernel
        variance: Initial variance for RBF kernel
        save_dir: Directory to save results
        
    Returns:
        Dictionary containing results
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Setup matplotlib for publication-quality plots
    setup_matplotlib_for_publication()
    
    # Calculate points to add per iteration
    n_per_iter = (n_final - n_initial) // n_iterations
    
    # Generate boundary data
    _, Xd, Xn, _, g_Xd, g_Xn = generate_square_domain_data(n_samples_per_side=int(np.sqrt(n_initial)))
    
    # Reshape boundary condition tensors to have 2 dimensions
    g_Xd = g_Xd.reshape(-1, 1)
    g_Xn = g_Xn.reshape(-1, 1)
    
    # Generate test points on a grid
    x_test = torch.linspace(0, 1, n_test_per_side)
    t_test = torch.linspace(0, 1, n_test_per_side)
    X_test, T_test = torch.meshgrid(x_test, t_test, indexing='ij')
    X_test_flat = torch.stack([X_test.flatten(), T_test.flatten()], dim=1).to(torch.float64)
    
    # Create a large candidate pool within the domain for active learning
    X_pool = sobol_rectangle_sampling(1000)
    
    # Setup kernel and GP solver
    kernel = RBFKernel2D(init_lengthscale=lengthscale, init_variance=variance)
    operators = PDEBoundaryOperators2D(kernel=kernel)
    solver = PosteriorSolver2D(kernel=kernel, operators=operators)
    
    # Compute exact solution for comparison
    x_grid, t_grid, u_exact = compute_exact_solution(N_x=n_test_per_side, N_t=n_test_per_side)
    
    # Interpolate the exact solution to test points
    exact_points = np.column_stack((x_grid.flatten(), t_grid.flatten()))
    exact_values = u_exact.flatten()
    test_points = X_test_flat.numpy()
    u_interp = griddata(exact_points, exact_values, test_points, method='cubic')
    
    # ------- ACTIVE LEARNING EXPERIMENT -------
    # Generate initial points using Sobol sequence
    Xi_active = sobol_rectangle_sampling(n_initial)
    f_Xi_active = torch.zeros((Xi_active.size(0), 1)).to(torch.float64)
    
    # Lists to track metrics over iterations
    active_points = [n_initial]  # Number of points at each iteration
    active_errors = []           # Mean absolute error at each iteration
    active_variances = []        # Mean variance at each iteration
    
    # Initial prediction with Sobol points
    C_full_active = solver.compute_covariance_matrix(Xi_active, Xd, Xn)
    C_inv_active = torch.inverse(C_full_active + 1e-6 * torch.eye(C_full_active.shape[0]))
    y_obs_active = torch.cat((f_Xi_active, g_Xd, g_Xn), dim=0)
    
    posterior_mean_active = solver.posterior_mean(X_test_flat, Xi_active, Xd, Xn, C_inv_active, y_obs_active).detach()
    active_abs_error = np.abs(posterior_mean_active.cpu().numpy().reshape(-1) - u_interp)
    active_errors.append(np.mean(active_abs_error))
    
    # Calculate posterior variance for active learning points
    cov_full_active = solver.posterior_covariance(X_test_flat, X_test_flat, Xi_active, Xd, Xn, C_inv_active).detach()
    variance_active = torch.diag(cov_full_active).cpu().numpy().reshape(-1)
    active_variances.append(np.mean(variance_active))
    
    Xi_collection = [Xi_active.clone()]  # Store points at each iteration for visualization
    mean_collection = [posterior_mean_active.clone()]  # Store predictions at each iteration
    var_collection = [torch.diag(cov_full_active).clone()]  # Store variances at each iteration
    
    # Active learning iterations
    for it in range(n_iterations):
        print(f"Active Learning Iteration {it+1}/{n_iterations}")
        
        # Select new points based on variance
        new_samples = adaptive_sampling_2d(
            solver=solver,
            X_pool=X_pool,
            X_train=Xi_active,
            Xd=Xd,
            Xn=Xn,
            y_train=y_obs_active,
            na=n_per_iter,
            kappa=2.0,
            exclusion_radius=0.05,
            acquisition_function="variance"
        )
        
        if new_samples.shape[0] == 0:
            print("Warning: No new points selected in this iteration.")
            continue
        
        # Add new points to training set
        Xi_active = torch.cat([Xi_active, new_samples], dim=0)
        f_Xi_active = torch.zeros((Xi_active.size(0), 1)).to(torch.float64)
        
        # Update observation vector
        y_obs_active = torch.cat((f_Xi_active, g_Xd, g_Xn), dim=0)
        
        # Recompute GP prediction
        C_full_active = solver.compute_covariance_matrix(Xi_active, Xd, Xn)
        C_inv_active = torch.inverse(C_full_active + 1e-6 * torch.eye(C_full_active.shape[0]))
        
        posterior_mean_active = solver.posterior_mean(X_test_flat, Xi_active, Xd, Xn, C_inv_active, y_obs_active).detach()
        active_abs_error = np.abs(posterior_mean_active.cpu().numpy().reshape(-1) - u_interp)
        active_errors.append(np.mean(active_abs_error))
        
        # Calculate posterior variance
        cov_full_active = solver.posterior_covariance(X_test_flat, X_test_flat, Xi_active, Xd, Xn, C_inv_active).detach()
        variance_active = torch.diag(cov_full_active).cpu().numpy().reshape(-1)
        active_variances.append(np.mean(variance_active))
        
        # Track number of points
        active_points.append(Xi_active.shape[0])
        Xi_collection.append(Xi_active.clone())
        mean_collection.append(posterior_mean_active.clone())
        var_collection.append(torch.diag(cov_full_active).clone())
        
        print(f"Points: {Xi_active.shape[0]}, Mean Error: {active_errors[-1]:.6f}, Mean Variance: {active_variances[-1]:.6f}")
    
    # ------- RANDOM SAMPLING EXPERIMENT (BASELINE) -------
    # Generate random interior points from the pool
    perm = torch.randperm(X_pool.size(0))
    Xi_random = X_pool[perm[:n_final]]
    f_Xi_random = torch.zeros((Xi_random.size(0), 1)).to(torch.float64)
    
    # Compute prediction with random points
    C_full_random = solver.compute_covariance_matrix(Xi_random, Xd, Xn)
    C_inv_random = torch.inverse(C_full_random + 1e-6 * torch.eye(C_full_random.shape[0]))
    y_obs_random = torch.cat((f_Xi_random, g_Xd, g_Xn), dim=0)
    
    posterior_mean_random = solver.posterior_mean(X_test_flat, Xi_random, Xd, Xn, C_inv_random, y_obs_random).detach()
    random_abs_error = np.abs(posterior_mean_random.cpu().numpy().reshape(-1) - u_interp)
    random_mean_error = np.mean(random_abs_error)
    
    # Calculate posterior variance for random points
    cov_full_random = solver.posterior_covariance(X_test_flat, X_test_flat, Xi_random, Xd, Xn, C_inv_random).detach()
    variance_random = torch.diag(cov_full_random).cpu().numpy().reshape(-1)
    mean_variance_random = np.mean(variance_random)
    
    # Display final results
    print("\n--- Final Results ---")
    print(f"Random Sampling ({Xi_random.shape[0]} points): Mean Error = {random_mean_error:.6f}, Mean Variance = {mean_variance_random:.6f}")
    print(f"Active Learning ({Xi_active.shape[0]} points): Mean Error = {active_errors[-1]:.6f}, Mean Variance = {active_variances[-1]:.6f}")
    print(f"Improvement: {100 * (random_mean_error - active_errors[-1]) / random_mean_error:.2f}% reduction in error")
    
    # Save results
    results = {
        'random_points': Xi_random.detach().numpy(),
        'active_points': Xi_active.detach().numpy(),
        'random_error': random_mean_error,
        'active_error': active_errors[-1],
        'random_variance': mean_variance_random,
        'active_variance': active_variances[-1],
        'convergence': {
            'points': active_points,
            'errors': active_errors,
            'variances': active_variances
        },
        'exact_solution': u_interp.reshape(n_test_per_side, n_test_per_side),
        'random_solution': posterior_mean_random.detach().numpy().reshape(n_test_per_side, n_test_per_side),
        'active_solution': posterior_mean_active.detach().numpy().reshape(n_test_per_side, n_test_per_side),
        'random_variance_field': variance_random.reshape(n_test_per_side, n_test_per_side),
        'active_variance_field': variance_active.reshape(n_test_per_side, n_test_per_side),
        'x_grid': x_test.numpy(),
        't_grid': t_test.numpy(),
        'iterations': {
            'points': [x.detach().numpy() for x in Xi_collection],
            'means': [m.detach().numpy().reshape(n_test_per_side, n_test_per_side) for m in mean_collection],
            'variances': [v.detach().numpy().reshape(n_test_per_side, n_test_per_side) for v in var_collection]
        }
    }
    
    # Save results as numpy file
    np.save(os.path.join(save_dir, 'active_learning_results.npy'), results)
    
    # Also save as pickle for easier loading
    with open(os.path.join(save_dir, 'active_learning_results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    # ------- VISUALIZATION OF RESULTS -------
    # Reshape test points for plotting
    x_test_np = X_test_flat[:, 0].numpy().reshape(n_test_per_side, n_test_per_side)
    t_test_np = X_test_flat[:, 1].numpy().reshape(n_test_per_side, n_test_per_side)
    
    # Plot 1: Convergence of error and variance
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Error plot
    ax1.plot(active_points, active_errors, 'o-', color='blue', linewidth=2)
    ax1.set_xlabel(r'Number of Training Points')
    ax1.set_ylabel(r'Mean Absolute Error')
    ax1.set_title(r'Error Convergence with Active Learning')
    ax1.grid(True)
    ax1.axhline(y=random_mean_error, color='r', linestyle='--', label=f'Random ({Xi_random.shape[0]} points)')
    ax1.legend()
    
    # Variance plot
    ax2.plot(active_points, active_variances, 'o-', color='green', linewidth=2)
    ax2.set_xlabel(r'Number of Training Points')
    ax2.set_ylabel(r'Mean Posterior Variance')
    ax2.set_title(r'Uncertainty Reduction with Active Learning')
    ax2.grid(True)
    ax2.axhline(y=mean_variance_random, color='r', linestyle='--', label=f'Random ({Xi_random.shape[0]} points)')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'active_learning_convergence.pdf'), dpi=600, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Final comparison of solutions
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot ground truth solution
    im0 = axes[0, 0].contourf(x_test_np, t_test_np, u_interp.reshape(n_test_per_side, n_test_per_side), 
                            levels=50, cmap='viridis')
    cbar0 = fig.colorbar(im0, ax=axes[0, 0])
    cbar0.set_label(r'$u(x,t)$')
    axes[0, 0].set_title(r'Exact Solution')
    axes[0, 0].set_xlabel(r'Spatial Coordinate $(x)$')
    axes[0, 0].set_ylabel(r'Time $(t)$')
    
    # Plot random sampling solution
    im1 = axes[0, 1].contourf(x_test_np, t_test_np, 
                            posterior_mean_random.detach().numpy().reshape(n_test_per_side, n_test_per_side), 
                            levels=50, cmap='viridis')
    cbar1 = fig.colorbar(im1, ax=axes[0, 1])
    cbar1.set_label(r'$\mu(x,t)$')
    axes[0, 1].scatter(Xi_random[:, 0], Xi_random[:, 1], color='red', s=10, alpha=0.7)
    axes[0, 1].set_title(r'Random Sampling Solution')
    axes[0, 1].set_xlabel(r'Spatial Coordinate $(x)$')
    axes[0, 1].set_ylabel(r'Time $(t)$')
    
    # Plot active learning solution
    im2 = axes[0, 2].contourf(x_test_np, t_test_np, 
                            posterior_mean_active.detach().numpy().reshape(n_test_per_side, n_test_per_side), 
                            levels=50, cmap='viridis')
    cbar2 = fig.colorbar(im2, ax=axes[0, 2])
    cbar2.set_label(r'$\mu(x,t)$')
    axes[0, 2].scatter(Xi_active[:, 0], Xi_active[:, 1], color='red', s=10, alpha=0.7)
    axes[0, 2].set_title(r'Active Learning Solution')
    axes[0, 2].set_xlabel(r'Spatial Coordinate $(x)$')
    axes[0, 2].set_ylabel(r'Time $(t)$')
    
    # Plot error for random sampling
    random_error = np.abs(posterior_mean_random.detach().numpy().reshape(n_test_per_side, n_test_per_side) - 
                        u_interp.reshape(n_test_per_side, n_test_per_side))
    im3 = axes[1, 0].contourf(x_test_np, t_test_np, random_error, levels=50, cmap='coolwarm')
    cbar3 = fig.colorbar(im3, ax=axes[1, 0])
    cbar3.set_label(r'$|u(x,t) - \mu(x,t)|$')
    axes[1, 0].set_title(f'Random Sampling Error (MAE: {random_mean_error:.6f})')
    axes[1, 0].set_xlabel(r'Spatial Coordinate $(x)$')
    axes[1, 0].set_ylabel(r'Time $(t)$')
    
    # Plot error for active learning
    active_error = np.abs(posterior_mean_active.detach().numpy().reshape(n_test_per_side, n_test_per_side) - 
                        u_interp.reshape(n_test_per_side, n_test_per_side))
    im4 = axes[1, 1].contourf(x_test_np, t_test_np, active_error, levels=50, cmap='coolwarm')
    cbar4 = fig.colorbar(im4, ax=axes[1, 1])
    cbar4.set_label(r'$|u(x,t) - \mu(x,t)|$')
    axes[1, 1].set_title(f'Active Learning Error (MAE: {active_errors[-1]:.6f})')
    axes[1, 1].set_xlabel(r'Spatial Coordinate $(x)$')
    axes[1, 1].set_ylabel(r'Time $(t)$')
    
    # Plot variance for active learning
    im5 = axes[1, 2].contourf(x_test_np, t_test_np, 
                            np.sqrt(variance_active.reshape(n_test_per_side, n_test_per_side)), 
                            levels=50, cmap='plasma')
    cbar5 = fig.colorbar(im5, ax=axes[1, 2])
    cbar5.set_label(r'$\sigma(x,t)$')
    axes[1, 2].set_title(r'Active Learning Posterior Standard Deviation')
    axes[1, 2].set_xlabel(r'Spatial Coordinate $(x)$')
    axes[1, 2].set_ylabel(r'Time $(t)$')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'active_learning_comparison.pdf'), dpi=600, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Evolution of points and variance over iterations
    n_vis = min(4, n_iterations+1)  # Visualize up to 4 iterations
    vis_idx = np.linspace(0, len(Xi_collection)-1, n_vis).astype(int)
    
    fig, axes = plt.subplots(2, n_vis, figsize=(4*n_vis, 10))
    if n_vis == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    
    vmin_pred = min(u_interp.min(), min(mean_collection[i].min().item() for i in vis_idx))
    vmax_pred = max(u_interp.max(), max(mean_collection[i].max().item() for i in vis_idx))
    
    for i, idx in enumerate(vis_idx):
        # Top row: Points and predictions
        Xi_iter = Xi_collection[idx]
        pred_iter = mean_collection[idx].detach().numpy().reshape(n_test_per_side, n_test_per_side)
        
        im1 = axes[0, i].contourf(x_test_np, t_test_np, pred_iter, levels=50, 
                                cmap='viridis', vmin=vmin_pred, vmax=vmax_pred)
        axes[0, i].scatter(Xi_iter[:, 0], Xi_iter[:, 1], color='red', s=10, alpha=0.7)
        axes[0, i].set_title(f'Iteration {idx}\n{Xi_iter.shape[0]} points')
        
        # Bottom row: Variance
        var_iter = var_collection[idx].detach().numpy().reshape(n_test_per_side, n_test_per_side)
        im2 = axes[1, i].contourf(x_test_np, t_test_np, np.sqrt(var_iter), levels=50, cmap='plasma')
        axes[1, i].scatter(Xi_iter[:, 0], Xi_iter[:, 1], color='white', s=10, alpha=0.7)
        axes[1, i].set_title(f'Std. Dev. (mean: {np.sqrt(var_collection[idx].mean().item()):.6f})')
    
    # Set labels for the first column
    axes[0, 0].set_ylabel(r'Time $(t)$')
    axes[1, 0].set_ylabel(r'Time $(t)$')
    
    # Set x-labels for the bottom row
    for i in range(n_vis):
        axes[1, i].set_xlabel(r'Spatial Coordinate $(x)$')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'active_learning_evolution.pdf'), dpi=600, bbox_inches='tight')
    plt.close()
    
    return results

if __name__ == "__main__":
    results = run_active_learning_experiment(
        n_initial=20,
        n_final=60,
        n_iterations=8,
        n_test_per_side=50,
        lengthscale=0.6,
        variance=0.02,
        save_dir='results'
    )