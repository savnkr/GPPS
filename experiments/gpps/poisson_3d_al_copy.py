# %%
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import pdb
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import qmc  # For Sobol sequence
from sklearn.cluster import KMeans

# %%

# Step 1: Define the 2D RBF Kernel for Gaussian Process
class RBFKernel2D(nn.Module):
    def __init__(self, init_lengthscale=1.0, init_variance=1.0):
        super(RBFKernel2D, self).__init__()
        self.lengthscale = torch.nn.Parameter(torch.tensor(init_lengthscale))
        self.variance = torch.nn.Parameter(torch.tensor(init_variance))

    def forward(self, X1, X2):
        """
        Compute the RBF kernel between two sets of 2D inputs.
        :param X1: Tensor of shape [N1, 2] for 2D points.
        :param X2: Tensor of shape [N2, 2] for 2D points.
        :return: Kernel matrix of shape [N1, N2].
        """
        diff = X1 - X2  # Shape: (N1, N2, 2)
        dist_sq = diff.pow(2).sum(-1)  # Sum over the last dimension (for 2D)
        return (self.variance**2) * torch.exp(-0.5 * dist_sq / (self.lengthscale**2))

# Step 2: PDE and Boundary Operators for 2D
class PDEBoundaryOperators2D(torch.nn.Module):
    def __init__(self, kernel):
        super(PDEBoundaryOperators2D, self).__init__()
        self.kernel = kernel
        self.hess_diag_fnx2 = lambda x1,x2: torch.diag(torch.func.hessian(self.kernel, argnums=1)(x1,x2))
        self.grad_fnx2 =lambda x1,x2: torch.vmap(torch.vmap(torch.func.grad(kernel, argnums=1), in_dims=(0, 0)), in_dims=(0, 0))(x1, x2)
        self.hess_fnx2 = lambda x1,x2: torch.vmap(torch.vmap(self.hess_diag_fnx2, in_dims=(0, 0)), in_dims=(0, 0))(x1, x2)
        self.grad_fnx1 =lambda x1,x2: torch.vmap(torch.vmap(torch.func.grad(kernel, argnums=0), in_dims=(0, 0)), in_dims=(0, 0))(x1, x2)
        self.hess_diag_fnx1 = lambda x1,x2: torch.diag(torch.func.hessian(self.kernel, argnums=0)(x1,x2))
        self.hess_fnx1 = lambda x1,x2: torch.vmap(torch.vmap(self.hess_diag_fnx1, in_dims=(0, 0)), in_dims=(0, 0))(x1, x2)

    def apply_pde_operator(self, X1, X2):
        """
        Apply the PDE operator L to X1 and L^T to X2 in 2D.
        Compute second derivatives (Hessian) in both x and y directions.
        """
        X1_expand = X1.unsqueeze(1).repeat(1, X2.size(0),1).requires_grad_(True)
        X2_expand = X2.unsqueeze(0).repeat(X1.size(0), 1,1).requires_grad_(True)
        # K = self.kernel(X1_expand, X2_expand)

        return self.operator_lx1lx2(X1_expand, X2_expand).detach()

    def apply_lb_operator(self, X1, Xb):
        """
        Apply the PDE operator L to X1 and boundary operator B to Xb in 2D.
        """
        X1_expand = X1.unsqueeze(1).repeat(1, Xb.size(0),1)
        Xb_expand = Xb.unsqueeze(0).repeat(X1.size(0), 1,1)
        hessian_K_x1 = self.hess_fnx1(X1_expand, Xb_expand)
        L_K_x1 = hessian_K_x1[:, :, 0] + hessian_K_x1[:, :, 1] + hessian_K_x1[:, :, 2]

        # Boundary condition: Dirichlet (use kernel directly)
        K_xb_dirichlet = L_K_x1.detach()

        return K_xb_dirichlet

    def apply_boundary_operator(self, X1, Xb):
        """
        Apply boundary operator B in 2D for boundary points.
        """
        X1_expand = X1.unsqueeze(1).repeat(1, Xb.size(0),1)
        Xb_expand = Xb.unsqueeze(0).repeat(X1.size(0), 1,1)
        K = self.kernel(X1_expand, Xb_expand)

        # Dirichlet boundary condition: Use the kernel directly
        return K

    def operator_lx2(self, X1, X2):
        hessian_K_x2 = self.hess_fnx2(X1, X2)
        return hessian_K_x2[:, :, 0] + hessian_K_x2[:, :, 1] + hessian_K_x2[:, :, 2] # Hessian trace (Laplacian)

    def operator_lx1lx2(self, X1, X2):
        hess_diag_lx1lx2 = lambda x1,x2: torch.diag(torch.func.hessian(self.hess_funcx2, argnums=0)(x1,x2))
        hess = torch.vmap(torch.vmap(hess_diag_lx1lx2, in_dims=(0, 0)), in_dims=(0, 0))(X1, X2)
        return hess[:, :, 0] + hess[:, :, 1] + hess[:, :, 2]

    def hess_funcx2(self, X1, X2):
        hess_op = self.hess_diag_fnx2(X1, X2)
        return torch.sum(hess_op)
# Step 3: Posterior Mean and Covariance with Operators (2D version)
class PosteriorSolver2D(nn.Module):
    def __init__(self, kernel, operators, noise_variance=1e-4):
        super(PosteriorSolver2D, self).__init__()
        self.kernel = kernel
        self.operators = operators
        self.noise_variance = noise_variance

    def compute_covariance_matrix(self, Xi, Xb):
        """
        Construct the covariance matrix C in 2D.
        """
        Xi_clone = Xi.clone().detach()
        Xb_clone = Xb.clone().detach()

        # Apply the PDE operator to interior points
        C_ii = self.operators.apply_pde_operator(Xi, Xi_clone)
        # Apply the boundary operator for interactions
        C_ib_dirichlet = self.operators.apply_lb_operator(Xi, Xb)
        C_bb_dirichlet = self.operators.apply_boundary_operator(Xb, Xb_clone)

        # Combine into the full covariance matrix
        C_full = torch.cat([torch.cat([C_ii, C_ib_dirichlet], dim=1),
                            torch.cat([C_ib_dirichlet.T, C_bb_dirichlet], dim=1)], dim=0)

        return C_full

    def compute_covariance_vector(self, x, Xi, Xb):
        """
        Compute the covariance vector c(x) for a 2D point x.
        """
        x_expand = x.unsqueeze(1).repeat(1, Xi.size(0),1).requires_grad_(True)
        xb_expand = x.unsqueeze(1).repeat(1,Xb.size(0),1)
        Xi_expand = Xi.unsqueeze(0).repeat(x.size(0), 1,1).requires_grad_(True)
        Xb_expand = Xb.unsqueeze(0).repeat(x.size(0), 1,1)
        L_K_x= self.operators.operator_lx2(x_expand, Xi_expand)

        B_K_x_dirichlet = self.kernel(xb_expand, Xb_expand)

        return torch.cat((L_K_x.detach(), B_K_x_dirichlet.detach()), dim=1)

    def posterior_mean(self, x, Xi, Xb, C_inv, y):
        """
        Compute the posterior mean in 2D at point x.
        """
        c_x = self.compute_covariance_vector(x, Xi, Xb)
        return c_x @ C_inv @ y

    def posterior_covariance(self, x, x_prime, Xi, Xb, C_inv):
        """
        Compute the posterior covariance in 2D between x and x_prime.
        """
        c_x = self.compute_covariance_vector(x, Xi, Xb)
        # c_x_prime = self.compute_covariance_vector(x_prime, Xi, Xb)

        x_expand = x.unsqueeze(1).repeat(1, x_prime.size(0),1)
        x_prime_expand = x_prime.unsqueeze(0).repeat(x.size(0), 1,1)
        base_cov = self.kernel(x_expand, x_prime_expand)
        c_x_cinv_cxprime = c_x @ C_inv @ c_x.T
        posterior_cov = base_cov - c_x_cinv_cxprime
        return posterior_cov#torch.clamp(posterior_cov, min=1e-10)


# %%
def generate_uniform_points_cuboid(n_samples_per_side=10, n_samples_per_bnd=8):
    """
    Generate points uniformly within the unit cuboid [0, 1]^3 and separate boundary and interior points.

    :param n_samples_per_side: Number of samples along each side of the cube.
    :return: Xi (interior points), Xb (boundary points), f_Xi (interior values), g_Xb (boundary values)
    """
    # Generate points uniformly in the cuboid domain [0, 1] x [0, 1] x [0, 1]
    x = torch.linspace(0, 1, n_samples_per_side)
    X, Y, Z = torch.meshgrid(x, x, x, indexing='ij')

    # Flatten the grids to create a list of points in the cuboid
    points = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=1)

    # Determine boundary points (pointsa where any coordinate is 0 or 1)
    boundary_mask = (points[:, 0] == 0) | (points[:, 0] == 1) | (points[:, 1] == 0) | (points[:, 1] == 1) | (points[:, 2] == 0) | (points[:, 2] == 1)
    #Xb = points[boundary_mask]  # Boundary points
    Xi = points[~boundary_mask]  # Interior points

    xb = torch.linspace(0, 1, n_samples_per_bnd)
    Xb,Yb,Zb = torch.meshgrid(xb, xb, xb, indexing='ij')
    pts_bnd = torch.stack([Xb.flatten(), Yb.flatten(), Zb.flatten()], dim=1)
    bnd_mask = (pts_bnd[:, 0] == 0) | (pts_bnd[:, 0] == 1) | (pts_bnd[:, 1] == 0) | (pts_bnd[:, 1] == 1) | (pts_bnd[:, 2] == 0) | (pts_bnd[:, 2] == 1)
    Xb = pts_bnd[bnd_mask]  # Boundary points
    # Xi = pts_bnd[~boundary_mask]  # Interior points


    # Compute the source term (f_Xi) for the Poisson equation at the interior points
    f_Xi = (-3 * (np.pi ** 2) * torch.sin(np.pi * Xi[:, 0]) * torch.sin(np.pi * Xi[:, 1]) * torch.sin(np.pi * Xi[:, 2])).reshape(-1, 1)

    # Compute the boundary values (g_Xb) using the sinusoidal solution
    g_Xb = torch.zeros_like(Xb)[:,:1]#torch.sin(np.pi * Xb[:, 0]) * torch.sin(np.pi * Xb[:, 1]) * torch.sin(np.pi * Xb[:, 2])

    return Xi, Xb, f_Xi, g_Xb

# Example usage
n_samples_per_side = 6  # Number of samples along each side
n_samples_per_bnd = 4
Xi, Xb, f_Xi, g_Xb = generate_uniform_points_cuboid(n_samples_per_side=n_samples_per_side,n_samples_per_bnd=n_samples_per_bnd)

# Plotting the points in the cuboid
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot interior points
ax.scatter(Xi[:, 0].numpy(), Xi[:, 1].numpy(), Xi[:, 2].numpy(), color='blue', marker='o', label='Interior Points')

# Plot boundary points
ax.scatter(Xb[:, 0].numpy(), Xb[:, 1].numpy(), Xb[:, 2].numpy(), color='red', marker='^', label='Boundary Points')

# Set labels and title
ax.set_title('Uniformly Distributed Points in the Cuboid [0, 1]^3')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$x_3$')
ax.legend()
ax.grid(True)

# Show plot
plt.show()

# %%
kernel = RBFKernel2D(init_lengthscale=0.531,init_variance = 0.678)
# kernel = RBFKernel2D(init_lengthscale=8.66,init_variance = 5.5)
operators = PDEBoundaryOperators2D(kernel=kernel)
solver = PosteriorSolver2D(kernel=kernel, operators=operators)

# Full covariance matrix and inverse for the observations
C_full = solver.compute_covariance_matrix(Xi.to(torch.float64), Xb.to(torch.float64))

C_inv = torch.inverse(C_full + 1e-6 * torch.eye(C_full.shape[0]))  # Add jitter for stability

# Combine interior and boundary observations
y_obs = torch.cat((f_Xi, g_Xb), dim=0).to(torch.float64)

X_1,X_2,_,_ = generate_uniform_points_cuboid(n_samples_per_side=12, n_samples_per_bnd=6)
X_test = torch.cat((X_1,X_2),dim=0).to(torch.float64)
# X_test = torch.cat((Xi,Xb),dim=0)
# Make predictions at a test point
# x_test = torch.tensor([[0.5, 0.5]], requires_grad=True)
posterior_mean = solver.posterior_mean(X_test, Xi, Xb, C_inv, y_obs).detach()
posterior_cov_full = solver.posterior_covariance(X_test, X_test.clone(), Xi, Xb, C_inv).detach()
posterior_cov = torch.sqrt(torch.diag(posterior_cov_full)).reshape(-1,1)
print(torch.sum(torch.isnan(posterior_cov)))

# %% Active Learning Components for 3D

def sobol_cuboid_sampling(n_points, domain_bounds=(0, 1), dim=3, seed=42):
    """
    Generate points using Sobol sequence and map them to a unit cuboid.
    
    Args:
        n_points: Number of points to generate
        domain_bounds: Tuple of (min, max) for all dimensions
        dim: Dimension (3 for 3D cuboid)
        seed: Random seed for reproducibility
        
    Returns:
        torch.Tensor of shape [n_points, 3] with points in unit cuboid
    """
    # Initialize Sobol sequence generator
    sampler = qmc.Sobol(d=dim, scramble=True, seed=seed)
    
    # Generate Sobol sequence in [0, 1)^3
    sample = sampler.random(n_points)
    
    # Scale to domain bounds
    min_val, max_val = domain_bounds
    sample = min_val + (max_val - min_val) * sample
    
    return torch.tensor(sample, dtype=torch.float64)

def filter_candidates_3d(X_pool, X_train, exclusion_radius=0.05):
    """
    Remove candidates that are too close to existing training points.
    
    Args:
        X_pool: Tensor of candidate points [N, 3]
        X_train: Tensor of existing training points [M, 3]
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

def adaptive_sampling_3d(
    solver,
    X_pool,
    X_train,
    Xb,
    y_train,
    na=5,
    kappa=2.0,
    exclusion_radius=0.05,
    acquisition_function="variance",
    ar_ratio=0.3
):
    """
    Adaptive sampling using uncertainty-based acquisition functions for 3D.
    
    Args:
        solver: GP solver for the PDE
        X_pool: Candidate pool of points [N, 3]
        X_train: Current training points [M, 3]
        Xb: Boundary points [P, 3]
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
    C_full = solver.compute_covariance_matrix(X_train, Xb)
    C_inv = torch.inverse(C_full + 1e-6 * torch.eye(C_full.shape[0]))
    
    # Filter candidate pool - remove points too close to training set
    X_pool_filtered = filter_candidates_3d(X_pool, X_train, exclusion_radius)
    
    if X_pool_filtered.shape[0] == 0:
        print("Warning: No valid candidates remain after filtering.")
        return torch.zeros((0, 3), dtype=torch.float64)
    
    # Compute posterior mean and covariance for candidates
    mean = solver.posterior_mean(X_pool_filtered, X_train, Xb, C_inv, y_train)
    cov_full = solver.posterior_covariance(X_pool_filtered, X_pool_filtered, X_train, Xb, C_inv).detach()
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

def run_3d_active_learning_experiment(n_initial=20, n_final=200, n_iterations=10, n_test=500):
    """
    Run experiments comparing random points vs. active learning for 3D PDE.
    Tracks uncertainty reduction and error over iterations.
    """
    # Parameters
    n_per_iter = (n_final - n_initial) // n_iterations
    boundary_points = 8 * 6  # Number of boundary points on unit cuboid (8 points per face * 6 faces)
    
    # Generate boundary points for the unit cuboid
    # Generate uniform points on the 6 faces of the cuboid
    Xb_list = []
    
    # Face points for x=0 and x=1
    for x in [0, 1]:
        for y in np.linspace(0, 1, 8):
            for z in np.linspace(0, 1, 8):
                Xb_list.append([x, y, z])
    
    # Face points for y=0 and y=1 (skipping corners to avoid duplicates)
    for y in [0, 1]:
        for x in np.linspace(0, 1, 8)[1:-1]:
            for z in np.linspace(0, 1, 8):
                Xb_list.append([x, y, z])
    
    # Face points for z=0 and z=1 (skipping edges to avoid duplicates)
    for z in [0, 1]:
        for x in np.linspace(0, 1, 8)[1:-1]:
            for y in np.linspace(0, 1, 8)[1:-1]:
                Xb_list.append([x, y, z])
    
    Xb = torch.tensor(Xb_list, dtype=torch.float64)
    g_Xb = torch.zeros((Xb.shape[0], 1), dtype=torch.float64)  # Boundary values
    
    # Generate test points within unit cuboid for evaluation
    X_test = sobol_cuboid_sampling(n_test, domain_bounds=(0.05, 0.95))
    
    # Calculate exact solution for test points
    x1_test = X_test[:, 0].detach().numpy()
    x2_test = X_test[:, 1].detach().numpy()
    x3_test = X_test[:, 2].detach().numpy()
    actual_solution = np.sin(np.pi * x1_test) * np.sin(np.pi * x2_test) * np.sin(np.pi * x3_test)
    
    # Create a large candidate pool within the unit cuboid for active learning
    X_pool = sobol_cuboid_sampling(2000, domain_bounds=(0.05, 0.95))
    
    # Setup kernel and GP solver
    kernel = RBFKernel2D(init_lengthscale=0.531, init_variance=0.678)
    operators = PDEBoundaryOperators2D(kernel=kernel)
    solver = PosteriorSolver2D(kernel=kernel, operators=operators)

    # ------- STOCHASTIC DUAL DESCENT WITH FIXED 80 POINTS -------
    # Generate exactly 80 interior points for fair comparison
    Xi_sdd = sobol_cuboid_sampling(80, domain_bounds=(0.05, 0.95))
    f_Xi_sdd = -3 * (np.pi ** 2) * torch.sin(np.pi * Xi_sdd[:, 0]) * torch.sin(np.pi * Xi_sdd[:, 1]) * torch.sin(np.pi * Xi_sdd[:, 2])
    f_Xi_sdd = f_Xi_sdd.reshape(-1, 1).to(torch.float64)
    
    print(f"Training SDD model with exactly {Xi_sdd.shape[0]} collocation points...")
    sdd_kernel, sdd_solver, sdd_losses = sdd_poisson_solver_3d(
        X_interior=Xi_sdd,
        Xb=Xb,
        f_interior=f_Xi_sdd,
        g_boundary=g_Xb,
        n_epochs=500,
        lr=0.01,
        batch_size=16
    )
    
    # Compute SDD predictions
    C_full_sdd = sdd_solver.compute_covariance_matrix(Xi_sdd, Xb)
    C_inv_sdd = torch.inverse(C_full_sdd + 1e-6 * torch.eye(C_full_sdd.shape[0]))
    y_obs_sdd = torch.cat((f_Xi_sdd, g_Xb), dim=0)
    
    posterior_mean_sdd = sdd_solver.posterior_mean(X_test, Xi_sdd, Xb, C_inv_sdd, y_obs_sdd).detach()
    sdd_abs_error = np.abs(posterior_mean_sdd.cpu().numpy().reshape(-1) - actual_solution)
    sdd_mean_error = np.mean(sdd_abs_error)
    
    print(f"SDD with {Xi_sdd.shape[0]} points: Mean Error = {sdd_mean_error:.6f}")
    
    # Save comprehensive results
    results = {
        'no_cluster_points': no_cluster_points,
        'with_cluster_points': with_cluster_points,
        'no_cluster_errors': no_cluster_errors,
        'with_cluster_errors': with_cluster_errors,
        'no_cluster_variances': no_cluster_variances,
        'with_cluster_variances': with_cluster_variances,
        'sdd_mean_error': sdd_mean_error,
        'sdd_points': Xi_sdd.detach().numpy(),
        'test_points': X_test.detach().numpy(),
        'actual_solution': actual_solution,
        'final_active_points': {
            'no_cluster': Xi_no_cluster.detach().numpy(),
            'with_cluster': Xi_with_cluster.detach().numpy()
        }
    }
    
    # Save as both npz and pkl formats
    import pickle
    np.savez('poisson_3d_comparison_results.npz', **results)
    with open('poisson_3d_comparison_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print("Results saved to both .npz and .pkl formats")
    
    return results
# Add a simple run statement at the end of the file
if __name__ == "__main__":
    run_3d_active_learning_experiment(n_initial=20, n_final=150, n_iterations=8)

# %%
# Assuming X_test and posterior_mean are already defined
# X_test.shape = (60, 3), posterior_mean.shape = (60, 1), posterior_cov is a (60, 60) matrix

# Detach the tensors from the computation graph if requires_grad=True
x1 = X_test[:, 0].detach().numpy()  # Extract the x1 (first column)
x2 = X_test[:, 1].detach().numpy()  # Extract the x2 (second column)
x3 = X_test[:, 2].detach().numpy()  # Extract the x3 (third column)
posterior_mean_np = posterior_mean.detach().numpy().reshape(-1)  # Flatten the posterior mean
posterior_cov_np = posterior_cov.detach().numpy().reshape(-1)  # Flatten the posterior covariance (diagonal only)

# Step 1: Compute the actual solution u(x1, x2, x3) = (1 - x1^2 - x2^2 - x3^2) / 6
actual_solution = np.sin(np.pi * x1) * np.sin(np.pi * x2) * np.sin(np.pi * x3)

# Step 2: Compute the error between the posterior mean and actual solution
error = np.abs(posterior_mean_np - actual_solution)

# Step 3: Create subplots for posterior mean, actual solution, error, and posterior covariance
fig, axes = plt.subplots(2, 2, figsize=(15, 12), subplot_kw={'projection': '3d'})  # Create 4 subplots for 3D scatter plots

# Plot the posterior mean
scat_posterior = axes[0, 0].scatter(x1, x2, x3, c=posterior_mean_np, cmap='viridis')
fig.colorbar(scat_posterior, ax=axes[0, 0])
axes[0, 0].set_title('Posterior Mean Scatter Plot')
axes[0, 0].set_xlabel('$x_1$')
axes[0, 0].set_ylabel('$x_2$')
axes[0, 0].set_zlabel('$x_3$')

# Plot the actual solution
scat_actual = axes[0, 1].scatter(x1, x2, x3, c=actual_solution, cmap='viridis')
fig.colorbar(scat_actual, ax=axes[0, 1])
axes[0, 1].set_title('Actual Solution Scatter Plot')
axes[0, 1].set_xlabel('$x_1$')
axes[0, 1].set_ylabel('$x_2$')
axes[0, 1].set_zlabel('$x_3$')

# Plot the error (posterior mean - actual solution)
scat_error = axes[1, 0].scatter(x1, x2, x3, c=error, cmap='coolwarm')
fig.colorbar(scat_error, ax=axes[1, 0])
axes[1, 0].set_title('Error (Posterior Mean - Actual Solution)')
axes[1, 0].set_xlabel('$x_1$')
axes[1, 0].set_ylabel('$x_2$')
axes[1, 0].set_zlabel('$x_3$')

# Plot the posterior variance (diagonal of posterior covariance)
scat_variance = axes[1, 1].scatter(x1, x2, x3, c=2 * posterior_cov_np, cmap='coolwarm')
fig.colorbar(scat_variance, ax=axes[1, 1])
axes[1, 1].set_title('Posterior Variance Scatter Plot (Covariance Diagonal)')
axes[1, 1].set_xlabel('$x_1$')
axes[1, 1].set_ylabel('$x_2$')
axes[1, 1].set_zlabel('$x_3$')

# Adjust layout to prevent overlap
plt.tight_layout()

# Show plot

# %%
np.mean(error)/np.mean(np.abs(actual_solution))


