#%%
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import os
from matplotlib import rcParams
import matplotlib.font_manager as fm
from matplotlib.ticker import MaxNLocator
import pickle
import os 
import sys
import time
import tracemalloc
sys.path.append(r"../../")
# Import the classes from the original script
from al_time_dep import (
    RBFKernel2D, 
    PDEBoundaryOperators2D, 
    PosteriorSolver2D, 
    generate_square_domain_data
)

def setup_matplotlib_for_publication():
    """Configure matplotlib for publication-quality plots"""
    # Use LaTeX for text rendering
    rcParams['text.usetex'] = False
    # rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    
    # Set font to be similar to LaTeX default
    # rcParams['font.family'] = 'serif'
    # rcParams['font.serif'] = ['Computer Modern Roman']
    rcParams['font.size'] = 11
    rcParams['axes.labelsize'] = 12
    rcParams['axes.titlesize'] = 14
    rcParams['xtick.labelsize'] = 10
    rcParams['ytick.labelsize'] = 10
    rcParams['legend.fontsize'] = 10
    rcParams['figure.titlesize'] = 14
    
    # Set figure properties
    rcParams['figure.figsize'] = (8, 6)
    rcParams['figure.dpi'] = 300
    rcParams['savefig.dpi'] = 600
    rcParams['savefig.bbox'] = 'tight'
    rcParams['savefig.pad_inches'] = 0.1
    
    # Set line properties
    rcParams['lines.linewidth'] = 1.5
    rcParams['lines.markersize'] = 6
    
    # Set grid properties
    rcParams['grid.linestyle'] = '--'
    rcParams['grid.linewidth'] = 0.5
    rcParams['grid.alpha'] = 0.8
    
    # Set axes properties
    rcParams['axes.grid'] = True
    rcParams['axes.axisbelow'] = True
    
    # Set legend properties
    rcParams['legend.frameon'] = True
    rcParams['legend.framealpha'] = 0.8
    rcParams['legend.edgecolor'] = 'k'
    
    # Set figure face and edge color
    rcParams['figure.facecolor'] = 'white'
    rcParams['figure.edgecolor'] = 'white'
    
    # Set axes face and edge color
    rcParams['axes.facecolor'] = 'white'
    rcParams['axes.edgecolor'] = 'black'

def compute_exact_solution(N_x=20, N_t=20, alpha=0.01):
    """
    Compute the exact solution for the heat equation using finite difference method.
    
    Args:
        N_x: Number of spatial points
        N_t: Number of time points
        alpha: Thermal diffusivity
        
    Returns:
        x_grid, t_grid, u_exact: Grids and solution values
    """
    L_x = 1.0
    L_t = 1.0
    dx = L_x / (N_x - 1)
    dt = L_t / (N_t - 1)
    x = np.linspace(0, L_x, N_x)
    u_initial = np.sin(np.pi*x)
    
    # Helper function to map 2D indices (i, j) to 1D index
    def index(i, j):
        return i + j * N_x
    
    # Sparse matrix A and right-hand side vector b
    A = lil_matrix((N_x * N_t, N_x * N_t))
    b = np.zeros(N_x * N_t)
    
    # Construct the matrix A and vector b
    for j in range(N_t):
        for i in range(N_x):
            k = index(i, j)
            
            # Boundary conditions
            if i == 0 or i == N_x - 1:  # Spatial boundaries (Dirichlet)
                A[k, k] = 1
                b[k] = 0
            elif j == 0:  # Initial pseudo-time boundary with sine wave
                A[k, k] = 1
                b[k] = u_initial[i]
            elif j == N_t - 1:  # Neumann boundary at end of pseudo-time
                A[k, k] = 1
                A[k, index(i, j - 1)] = -1
                b[k] = 0
            else:
                # Interior points - finite difference scheme
                A[k, k] = 1 + 2 * alpha * dt / dx**2
                A[k, index(i - 1, j)] = -alpha * dt / dx**2
                A[k, index(i + 1, j)] = -alpha * dt / dx**2
                A[k, index(i, j - 1)] = -1
    
    # Solve the linear system
    u_exact = spsolve(A.tocsr(), b).reshape((N_t, N_x))
    
    # Create meshgrid for plotting
    x_grid, t_grid = np.meshgrid(np.linspace(0, 1, N_x), np.linspace(0, 1, N_t))
    
    return x_grid, t_grid, u_exact

def generate_random_collocation_points(n_points_interior, n_points_boundary, seed=42):
    """
    Generate random collocation points for the PDE problem.
    
    Args:
        n_points_interior: Number of interior points
        n_points_boundary: Number of points on each boundary
        seed: Random seed for reproducibility
        
    Returns:
        Xi, Xd, Xn, f_Xi, g_Xd, g_Xn: Training data points and values
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Generate interior points randomly in (0,1) x (0,1)
    Xi = torch.rand(n_points_interior, 2).to(torch.float64)
    
    # Initial condition (t=0) - random x positions
    n_initial = n_points_boundary
    x_initial = torch.rand(n_initial, 1).to(torch.float64)
    t_initial = torch.zeros(n_initial, 1).to(torch.float64)
    Xd_initial = torch.cat((x_initial, t_initial), dim=1)
    g_Xd_initial = torch.sin(np.pi * x_initial)
    
    # Boundary conditions (x=0 and x=1) - random t positions
    n_boundary_each = n_points_boundary // 2
    
    # x=0 boundary
    t_left = torch.rand(n_boundary_each, 1).to(torch.float64)
    x_left = torch.zeros(n_boundary_each, 1).to(torch.float64)
    Xd_left = torch.cat((x_left, t_left), dim=1)
    g_Xd_left = torch.zeros(n_boundary_each, 1).to(torch.float64)
    
    # x=1 boundary
    t_right = torch.rand(n_boundary_each, 1).to(torch.float64)
    x_right = torch.ones(n_boundary_each, 1).to(torch.float64)
    Xd_right = torch.cat((x_right, t_right), dim=1)
    g_Xd_right = torch.zeros(n_boundary_each, 1).to(torch.float64)
    
    # Combine Dirichlet boundaries
    Xd = torch.cat((Xd_initial, Xd_left, Xd_right), dim=0)
    g_Xd = torch.cat((g_Xd_initial, g_Xd_left, g_Xd_right), dim=0)
    
    # Neumann boundary at t=1 (final time) - random x positions
    n_neumann = n_points_boundary
    x_final = torch.rand(n_neumann, 1).to(torch.float64)
    t_final = torch.ones(n_neumann, 1).to(torch.float64)
    Xn = torch.cat((x_final, t_final), dim=1)
    g_Xn = torch.zeros(n_neumann, 1).to(torch.float64)
    
    # Generate f_Xi (interior values - heat equation)
    f_Xi = torch.zeros(n_points_interior, 1).to(torch.float64)
    
    return Xi, Xd, Xn, f_Xi, g_Xd, g_Xn

def train_SDD(N, du, input_dim, sigma_n, T, B, beta, rho, r, num_epochs, C, y):
    """
    Stochastic Dual Descent algorithm implementation.
    
    Args:
        N: Number of data points
        du: Output dimension
        input_dim: Input dimension  
        sigma_n: Likelihood variance
        T: Number of steps (not used in this implementation)
        B: Batch size
        beta: Step size
        rho: Momentum parameter
        r: Averaging parameter
        num_epochs: Number of training epochs
        C: Covariance matrix
        y: Observations
        
    Returns:
        A_approx: Approximated dual variables
    """
    print(f"Training SDD with N={N}, batch_size={B}")
    
    # Start memory and time tracking
    tracemalloc.start()
    start_time = time.time()
    
    A_t = torch.zeros(N, du, dtype=torch.float64)       # Parameter A_t
    V_t = torch.zeros(N, du, dtype=torch.float64)        # Velocity V_t
    A_bar_t = torch.zeros(N, du, dtype=torch.float64)    # Averaged parameter A_bar_t
    K_full = (C + 1e-6 * torch.eye(C.shape[0]))
    
    for t in range(num_epochs):
        S = A_t + rho * V_t      # Shape: [N, du]

        # Sample random batch indices - ensure they're within bounds
        It = torch.randint(0, N, (min(B, N),))  # Use min(B, N) to avoid index errors

        # Initialize gradient G_t
        G_t = torch.zeros(N, du, dtype=torch.float64)

        # For each index i in the batch
        batch_size_actual = len(It)
        G_t[It] = (N / batch_size_actual) * K_full[It] @ S - y[It]
        V_t = rho * V_t - beta * G_t                  # Update V_t

        # Update parameters A_t
        A_t += V_t

        # Iterative averaging of parameters
        A_bar_t = r * A_t + (1 - r) * A_bar_t

        # (Optional) Compute and print the loss every 1000 steps
        if t % 1000 == 0 or t == num_epochs-1:
            # Compute the predictions
            pred = K_full @ A_t                  # Shape: [N, du]
            loss_term1 = 0.5 * torch.norm(y - pred) ** 2
            At_K_At = torch.sum(A_t * (K_full @ A_t))
            loss_term2 = (sigma_n / 2) * At_K_At
            L_t = loss_term1 + loss_term2
            
            # Get current memory usage
            current, peak = tracemalloc.get_traced_memory()
            elapsed_time = time.time() - start_time
            
            print(f"Step {t}, Loss: {L_t.item():.6e}, Time: {elapsed_time:.2f}s, Memory: {current/1024/1024:.2f}MB")
    
    # Final memory and time measurement
    final_time = time.time() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    print(f"Training completed in {final_time:.2f}s, Peak memory: {peak/1024/1024:.2f}MB")
    
    A_approx = A_bar_t
    return A_approx

# Extend PosteriorSolver2D class to include SDD methods
class ExtendedPosteriorSolver2D(PosteriorSolver2D):
    def posterior_mean_sdd(self, x, Xi, Xd, Xn, A_approx):
        """
        Compute the posterior mean using SDD approximation.
        """
        c_x = self.compute_covariance_vector(x, Xi, Xd, Xn)
        return c_x @ A_approx

    def posterior_covariance_sdd(self, x, x_prime, cov_vec, A_approx):
        """
        Compute the posterior covariance using SDD approximation.
        """
        x_expand = x.unsqueeze(1).repeat(1, x_prime.size(0), 1)
        x_prime_expand = x_prime.unsqueeze(0).repeat(x.size(0), 1, 1)
        base_cov = self.kernel(x_expand, x_prime_expand)
        c_x_cinv_cxprime = cov_vec @ A_approx
        posterior_cov = base_cov - c_x_cinv_cxprime
        return posterior_cov

def plot_exact_gp_results(x_grid, t_grid, exact_solution, posterior_mean, posterior_var, 
                         absolute_error, Xi, Xd, Xn, save_path):
    """
    Create publication-quality plots for Exact GP results.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Ground Truth
    im0 = axes[0, 0].contourf(x_grid, t_grid, exact_solution, levels=50, cmap='viridis')
    cbar0 = fig.colorbar(im0, ax=axes[0, 0], shrink=0.8)
    cbar0.set_label(r'$u(x,t)$', fontsize=12)
    axes[0, 0].set_title(r'Ground Truth', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel(r'$x$', fontsize=12)
    axes[0, 0].set_ylabel(r'$t$', fontsize=12)
    
    # Plot 2: Mean Predicted Solution
    im1 = axes[0, 1].contourf(x_grid, t_grid, posterior_mean, levels=50, cmap='viridis')
    cbar1 = fig.colorbar(im1, ax=axes[0, 1], shrink=0.8)
    cbar1.set_label(r'$\mu_{GP}(x,t)$', fontsize=12)
    axes[0, 1].set_title(r'Mean Predicted Solution', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel(r'$x$', fontsize=12)
    axes[0, 1].set_ylabel(r'$t$', fontsize=12)
    
    # Plot 3: Absolute Error
    im2 = axes[1, 0].contourf(x_grid, t_grid, absolute_error, levels=50, cmap='coolwarm')
    cbar2 = fig.colorbar(im2, ax=axes[1, 0], shrink=0.8)
    cbar2.set_label(r'$|u(x,t) - \mu_{GP}(x,t)|$', fontsize=12)
    axes[1, 0].set_title(r'Absolute Error', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel(r'$x$', fontsize=12)
    axes[1, 0].set_ylabel(r'$t$', fontsize=12)
    
    # Plot 4: Standard Deviation
    im3 = axes[1, 1].contourf(x_grid, t_grid, np.sqrt(posterior_var), levels=50, cmap='plasma')
    cbar3 = fig.colorbar(im3, ax=axes[1, 1], shrink=0.8)
    cbar3.set_label(r'$\sigma_{GP}(x,t)$', fontsize=12)
    axes[1, 1].set_title(r'Standard Deviation', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel(r'$x$', fontsize=12)
    axes[1, 1].set_ylabel(r'$t$', fontsize=12)
    
    # Add training points to all plots with smaller, more subtle markers
    for ax in axes.flatten():
        ax.scatter(Xi[:, 0], Xi[:, 1], color='red', s=8, alpha=0.6, marker='o')
        ax.scatter(Xd[:, 0], Xd[:, 1], color='black', s=8, alpha=0.6, marker='s')
        ax.scatter(Xn[:, 0], Xn[:, 1], color='white', s=8, alpha=0.8, edgecolors='black', 
                  marker='^', linewidth=0.5)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    
    plt.tight_layout(pad=3.0)
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()

def plot_sdd_gp_results(x_grid, t_grid, exact_solution, sdd_mean, sdd_var, 
                       sdd_error, Xi, Xd, Xn, save_path):
    """
    Create publication-quality plots for SDD GP results.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Ground Truth
    im0 = axes[0, 0].contourf(x_grid, t_grid, exact_solution, levels=50, cmap='viridis')
    cbar0 = fig.colorbar(im0, ax=axes[0, 0], shrink=0.8)
    cbar0.set_label(r'$u(x,t)$', fontsize=12)
    axes[0, 0].set_title(r'Ground Truth', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel(r'$x$', fontsize=12)
    axes[0, 0].set_ylabel(r'$t$', fontsize=12)
    
    # Plot 2: Mean Predicted Solution
    im1 = axes[0, 1].contourf(x_grid, t_grid, sdd_mean, levels=50, cmap='viridis')
    cbar1 = fig.colorbar(im1, ax=axes[0, 1], shrink=0.8)
    cbar1.set_label(r'$\mu_{SDD}(x,t)$', fontsize=12)
    axes[0, 1].set_title(r'Mean Predicted Solution', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel(r'$x$', fontsize=12)
    axes[0, 1].set_ylabel(r'$t$', fontsize=12)
    
    # Plot 3: Absolute Error
    im2 = axes[1, 0].contourf(x_grid, t_grid, sdd_error, levels=50, cmap='coolwarm')
    cbar2 = fig.colorbar(im2, ax=axes[1, 0], shrink=0.8)
    cbar2.set_label(r'$|u(x,t) - \mu_{SDD}(x,t)|$', fontsize=12)
    axes[1, 0].set_title(r'Absolute Error', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel(r'$x$', fontsize=12)
    axes[1, 0].set_ylabel(r'$t$', fontsize=12)
    
    # Plot 4: Standard Deviation
    im3 = axes[1, 1].contourf(x_grid, t_grid, np.sqrt(sdd_var), levels=50, cmap='plasma')
    cbar3 = fig.colorbar(im3, ax=axes[1, 1], shrink=0.8)
    cbar3.set_label(r'$\sigma_{SDD}(x,t)$', fontsize=12)
    axes[1, 1].set_title(r'Standard Deviation', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel(r'$x$', fontsize=12)
    axes[1, 1].set_ylabel(r'$t$', fontsize=12)
    
    # Add training points to all plots with smaller, more subtle markers
    for ax in axes.flatten():
        ax.scatter(Xi[:, 0], Xi[:, 1], color='red', s=8, alpha=0.6, marker='o')
        ax.scatter(Xd[:, 0], Xd[:, 1], color='black', s=8, alpha=0.6, marker='s')
        ax.scatter(Xn[:, 0], Xn[:, 1], color='white', s=8, alpha=0.8, edgecolors='black', 
                  marker='^', linewidth=0.5)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    
    plt.tight_layout(pad=3.0)
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()

def generate_gp_solution(n_samples_per_side=5, n_test_per_side=20, lengthscale=0.6, variance=0.02, 
                         use_random_points=True, n_interior_points=40, n_boundary_points=20,
                         sdd_iterations=4000, sdd_lr=5, sdd_batch_size=20, save_dir='results'):
    """
    Generate and save the probabilistic solution of the PDE using GP and SDD.
    
    Args:
        n_samples_per_side: Number of training samples per side (for grid)
        n_test_per_side: Number of test points per side for evaluation
        lengthscale: Initial lengthscale for RBF kernel
        variance: Initial variance for RBF kernel
        use_random_points: Whether to use random collocation points
        n_interior_points: Number of random interior points
        n_boundary_points: Number of random boundary points
        sdd_iterations: Number of iterations for SDD
        sdd_lr: Learning rate for SDD (beta parameter)
        sdd_batch_size: Batch size for SDD
        save_dir: Directory to save results
        
    Returns:
        Dictionary containing results
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Setup matplotlib for publication-quality plots
    setup_matplotlib_for_publication()
    
    # Generate training data - either grid or random points
    if use_random_points:
        Xi, Xd, Xn, f_Xi, g_Xd, g_Xn = generate_random_collocation_points(
            n_interior_points, n_boundary_points)
        print(f"Using {n_interior_points} random interior points and {n_boundary_points} boundary points")
    else:
        Xi, Xd, Xn, f_Xi, g_Xd, g_Xn = generate_square_domain_data(n_samples_per_side=n_samples_per_side)
        print(f"Using uniform grid with {n_samples_per_side} samples per side")
    
    # Reshape boundary condition tensors to have 2 dimensions
    f_Xi = f_Xi.reshape(-1, 1)
    g_Xd = g_Xd.reshape(-1, 1)
    g_Xn = g_Xn.reshape(-1, 1)
    
    # Generate test points on a grid
    x_test = torch.linspace(0, 1, n_test_per_side)
    t_test = torch.linspace(0, 1, n_test_per_side)
    X_test, T_test = torch.meshgrid(x_test, t_test, indexing='ij')
    X_test_flat = torch.stack([X_test.flatten(), T_test.flatten()], dim=1).to(torch.float64)
    
    # Setup kernel and GP solver
    kernel = RBFKernel2D(init_lengthscale=lengthscale, init_variance=variance)
    operators = PDEBoundaryOperators2D(kernel=kernel)
    solver = ExtendedPosteriorSolver2D(kernel=kernel, operators=operators)
    
    # Combine interior and boundary observations
    y_obs = torch.cat((f_Xi, g_Xd, g_Xn), dim=0)
    
    # Compute covariance matrix and its inverse for exact GP
    print("Computing exact GP solution...")
    C_full = solver.compute_covariance_matrix(Xi, Xd, Xn)
    C_inv = torch.inverse(C_full + 1e-6 * torch.eye(C_full.shape[0]))
    
    # Compute posterior mean and covariance for exact GP
    posterior_mean = solver.posterior_mean(X_test_flat, Xi, Xd, Xn, C_inv, y_obs)
    posterior_cov_full = solver.posterior_covariance(X_test_flat, X_test_flat, Xi, Xd, Xn, C_inv)
    posterior_var = torch.diag(posterior_cov_full).reshape(-1, 1)
    
    # Compute SDD solution
    print("Computing SDD solution...")
    
    # SDD algorithm parameters
    du = 1            # Output dimension
    input_dim = 1     # Input dimension
    sigma_n = 0       # Likelihood variance
    T = 1000          # Number of steps (not used)
    B = sdd_batch_size # Batch size
    beta = sdd_lr     # Step size
    rho = 0.98         # Momentum parameter
    r = 0.99           # Averaging parameter
    num_epochs = sdd_iterations
    
    # Get actual data size
    N = C_full.shape[0]
    print(f"Training SDD with N = {N}")
    
    with torch.no_grad():
        C_full_jitter = C_full + 1e-6 * torch.eye(C_full.shape[0])
        A_approx = train_SDD(N, du, input_dim, sigma_n, T, B, beta, rho, r, num_epochs, C_full_jitter, y_obs.reshape(-1, 1))
    
    # Compute SDD posterior mean
    posterior_mean_sdd = solver.posterior_mean_sdd(X_test_flat, Xi, Xd, Xn, A_approx).detach()
    
    # Compute SDD posterior variance
    torch.manual_seed(0)
    with torch.no_grad():
        cov_vec = solver.compute_covariance_vector(X_test_flat, Xi, Xd, Xn).detach()
        A_approx_var = train_SDD(N, cov_vec.shape[0], input_dim, sigma_n, T, B, beta, rho, r, 
                                min(1000, num_epochs), C_full_jitter, cov_vec.T)
    
    posterior_cov_sdd = solver.posterior_covariance_sdd(X_test_flat, X_test_flat.clone(), cov_vec, A_approx_var).detach()
    posterior_var_sdd = torch.sqrt(torch.diag(posterior_cov_sdd)).reshape(-1, 1)
    print(f"SDD predictions computed. NaN count: {torch.sum(torch.isnan(posterior_var_sdd))}")
    
    # Compute exact solution
    x_grid, t_grid, u_exact = compute_exact_solution(N_x=n_test_per_side, N_t=n_test_per_side)
    
    # Reshape for plotting
    x_test_np = X_test_flat[:, 0].detach().numpy().reshape(n_test_per_side, n_test_per_side)
    t_test_np = X_test_flat[:, 1].detach().numpy().reshape(n_test_per_side, n_test_per_side)
    posterior_mean_np = posterior_mean.detach().numpy().reshape(n_test_per_side, n_test_per_side)
    posterior_var_np = posterior_var.detach().numpy().reshape(n_test_per_side, n_test_per_side)
    sdd_mean_np = posterior_mean_sdd.detach().numpy().reshape(n_test_per_side, n_test_per_side)
    sdd_var_np = posterior_var_sdd.detach().numpy().reshape(n_test_per_side, n_test_per_side)
    
    # Compute errors
    exact_gp_error = np.abs(posterior_mean_np - u_exact.T)
    sdd_error = np.abs(sdd_mean_np - u_exact.T)
    
    # Save results as .npz file
    np.savez_compressed(
        os.path.join(save_dir, 'gp_pde_solution_results.npz'),
        x_grid=x_test_np,
        t_grid=t_test_np,
        exact_solution=u_exact.T,
        # Exact GP results
        exact_gp_posterior_mean=posterior_mean_np,
        exact_gp_posterior_var=posterior_var_np,
        exact_gp_absolute_error=exact_gp_error,
        # SDD GP results
        sdd_posterior_mean=sdd_mean_np,
        sdd_posterior_var=sdd_var_np,
        sdd_absolute_error=sdd_error,
        # Training points
        Xi=Xi.detach().numpy(),
        Xd=Xd.detach().numpy(),
        Xn=Xn.detach().numpy(),
        # Kernel parameters
        lengthscale=kernel.lengthscale.item(),
        variance=kernel.variance.item(),
        # Comparison metrics
        gp_vs_sdd_mean_diff=np.abs(posterior_mean_np - sdd_mean_np),
        gp_vs_sdd_error_diff=np.abs(exact_gp_error - sdd_error)
    )
    
    # Create separate plots for Exact GP and SDD GP
    plot_exact_gp_results(
        x_test_np, t_test_np, u_exact.T, posterior_mean_np, posterior_var_np,
        exact_gp_error, Xi.detach().numpy(), Xd.detach().numpy(), Xn.detach().numpy(),
        os.path.join(save_dir, 'exact_gp_pde_solution.pdf')
    )
    
    plot_sdd_gp_results(
        x_test_np, t_test_np, u_exact.T, sdd_mean_np, sdd_var_np,
        sdd_error, Xi.detach().numpy(), Xd.detach().numpy(), Xn.detach().numpy(),
        os.path.join(save_dir, 'sdd_gp_pde_solution.pdf')
    )
    
    # Comparison plot: GP vs SDD
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot GP mean
    im0 = axes[0, 0].contourf(x_test_np, t_test_np, posterior_mean_np, levels=50, cmap='viridis')
    cbar0 = fig.colorbar(im0, ax=axes[0, 0])
    cbar0.set_label(r'$\mu_{GP}(x,t)$')
    axes[0, 0].set_title(r'Exact GP Mean')
    axes[0, 0].set_xlabel(r'Spatial Coordinate $(x)$')
    axes[0, 0].set_ylabel(r'Time $(t)$')
    
    # Plot SDD mean
    im1 = axes[0, 1].contourf(x_test_np, t_test_np, sdd_mean_np, levels=50, cmap='viridis')
    cbar1 = fig.colorbar(im1, ax=axes[0, 1])
    cbar1.set_label(r'$\mu_{SDD}(x,t)$')
    axes[0, 1].set_title(r'SDD Mean')
    axes[0, 1].set_xlabel(r'Spatial Coordinate $(x)$')
    axes[0, 1].set_ylabel(r'Time $(t)$')
    
    # Plot difference between GP and SDD
    diff = np.abs(posterior_mean_np - sdd_mean_np)
    im2 = axes[1, 0].contourf(x_test_np, t_test_np, diff, levels=50, cmap='coolwarm')
    cbar2 = fig.colorbar(im2, ax=axes[1, 0])
    cbar2.set_label(r'$|\mu_{GP}(x,t) - \mu_{SDD}(x,t)|$')
    axes[1, 0].set_title(r'Absolute Difference')
    axes[1, 0].set_xlabel(r'Spatial Coordinate $(x)$')
    axes[1, 0].set_ylabel(r'Time $(t)$')
    
    # Plot relative error difference
    rel_error_diff = np.abs(exact_gp_error - sdd_error)
    im3 = axes[1, 1].contourf(x_test_np, t_test_np, rel_error_diff, levels=50, cmap='coolwarm')
    cbar3 = fig.colorbar(im3, ax=axes[1, 1])
    cbar3.set_label(r'$|Error_{GP} - Error_{SDD}|$')
    axes[1, 1].set_title(r'Error Difference')
    axes[1, 1].set_xlabel(r'Spatial Coordinate $(x)$')
    axes[1, 1].set_ylabel(r'Time $(t)$')
    
    # Add training points to all plots
    for ax in axes.flatten():
        ax.scatter(Xi[:, 0], Xi[:, 1], color='red', s=10, alpha=0.7, label='Interior')
        ax.scatter(Xd[:, 0], Xd[:, 1], color='black', s=10, alpha=0.7, label='Dirichlet')
        ax.scatter(Xn[:, 0], Xn[:, 1], color='white', s=10, alpha=0.7, edgecolors='black', label='Neumann')
    
    # Add legend to the first plot only
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=3)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.suptitle(r'Comparison: Exact GP vs SDD for Heat Equation', fontsize=16)
    
    # Save figure
    plt.savefig(os.path.join(save_dir, 'gp_vs_sdd_comparison.pdf'), dpi=600, bbox_inches='tight')
    plt.show()
    plt.close()
    
    # Return results dictionary for backward compatibility
    results = {
        'x_grid': x_test_np,
        't_grid': t_test_np,
        'exact_solution': u_exact.T,
        'posterior_mean': posterior_mean_np,
        'posterior_var': posterior_var_np,
        'sdd_mean': sdd_mean_np,
        'sdd_var': sdd_var_np,
        'exact_gp_error': exact_gp_error,
        'sdd_error': sdd_error,
        'training_points': {
            'Xi': Xi.detach().numpy(),
            'Xd': Xd.detach().numpy(),
            'Xn': Xn.detach().numpy()
        },
        'kernel_params': {
            'lengthscale': kernel.lengthscale.item(),
            'variance': kernel.variance.item()
        }
    }
    
    # Print summary
    print(f"GP and SDD PDE solutions generated and saved to {save_dir}")
    print(f"Results saved as .npz file: gp_pde_solution_results.npz")
    print(f"Exact GP - Mean absolute error: {np.mean(exact_gp_error):.6f}")
    print(f"Exact GP - Maximum absolute error: {np.max(exact_gp_error):.6f}")
    print(f"Exact GP - Mean posterior std: {np.mean(np.sqrt(posterior_var_np)):.6f}")
    print(f"SDD - Mean absolute error: {np.mean(sdd_error):.6f}")
    print(f"SDD - Maximum absolute error: {np.max(sdd_error):.6f}")
    print(f"SDD - Mean posterior std: {np.mean(np.sqrt(sdd_var_np)):.6f}")
    
    return results

if __name__ == "__main__":
    results = generate_gp_solution(
        use_random_points=False,
        n_interior_points=100,
        n_boundary_points=30,
        n_test_per_side=50,
        sdd_iterations=4000,
        sdd_lr=5,
        sdd_batch_size=50
    )
#%%