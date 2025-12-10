# %%
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import pdb
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rcParams
import matplotlib.font_manager as fm
from matplotlib.ticker import MaxNLocator
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
        L_K_x1 = hessian_K_x1[:, :, 0] + hessian_K_x1[:, :, 1]

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
        return hessian_K_x2[:, :, 0] + hessian_K_x2[:, :, 1] # Hessian trace (Laplacian)

    def operator_lx1lx2(self, X1, X2):
        hess_diag_lx1lx2 = lambda x1,x2: torch.diag(torch.func.hessian(self.hess_funcx2, argnums=0)(x1,x2))
        hess = torch.vmap(torch.vmap(hess_diag_lx1lx2, in_dims=(0, 0)), in_dims=(0, 0))(X1, X2)
        return hess[:, :, 0] + hess[:, :, 1]

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

    def posterior_mean_sdd(self, x, Xi, Xb, A_approx):
        """
        Compute the posterior mean in 2D at point x using SDD approximation.
        """
        c_x = self.compute_covariance_vector(x, Xi, Xb)
        return c_x @ A_approx

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

    def posterior_covariance_sdd(self, x, x_prime, cov_vec, A_approx):
        """
        Compute the posterior covariance in 2D between x and x_prime using SDD approximation.
        """
        # c_x = self.compute_covariance_vector(x, Xi, Xb)
        # c_x_prime = self.compute_covariance_vector(x_prime, Xi, Xb)

        x_expand = x.unsqueeze(1).repeat(1, x_prime.size(0),1)
        x_prime_expand = x_prime.unsqueeze(0).repeat(x.size(0), 1,1)
        base_cov = self.kernel(x_expand, x_prime_expand)
        c_x_cinv_cxprime = cov_vec@A_approx
        posterior_cov = base_cov - c_x_cinv_cxprime
        return posterior_cov  #torch.clamp(posterior_cov, min=1e-10)

    def run_sdd(self, Xi, Xb, y, n_samples=500, n_iters=1000, batch_size=50, lr=0.01):
        """
        Run Stochastic Dual Descent to approximate C^{-1}y without explicitly computing C^{-1}.
        
        Args:
            Xi: Interior points
            Xb: Boundary points
            y: Observations/targets
            n_samples: Number of random samples for approximation
            n_iters: Number of SGD iterations
            batch_size: Batch size for SGD
            lr: Learning rate
            
        Returns:
            A_approx: Approximation of C^{-1}y
        """
        # Generate random Gaussian samples for approximation
        device = y.device
        d = y.size(0)
        
        # Initialize the approximation
        A_approx = torch.zeros_like(y)
        
        # Generate random samples for the stochastic objective
        Z = torch.randn(n_samples, d, device=device)
        
        # Run SGD iterations
        for iter in range(n_iters):
            # Sample random batch
            batch_indices = torch.randperm(n_samples)[:batch_size]
            Z_batch = Z[batch_indices]
            
            # Compute full covariance matrix (could be cached)
            C_full = self.compute_covariance_matrix(Xi, Xb)
            
            # Compute gradient estimate
            grad = torch.zeros_like(A_approx)
            for z in Z_batch:
                Cz = C_full @ z.unsqueeze(1)
                Az = A_approx.t() @ z
                residual = Az - y.t() @ z
                grad += Cz * residual
            
            grad /= batch_size
            
            # Update approximation with SGD step
            A_approx = A_approx - lr * grad
            
            # Optional: Print progress
            if (iter + 1) % 100 == 0:
                with torch.no_grad():
                    error = torch.norm(C_full @ A_approx - y) / torch.norm(y)
                    print(f"Iteration {iter+1}/{n_iters}, Relative Error: {error.item():.6f}")
        
        return A_approx
    
    def compute_posterior_stats_sdd(self, X_test, Xi, Xb, y_obs, n_samples=500, n_iters=1000):
        """
        Compute posterior mean and covariance using SDD approximation.
        
        Args:
            X_test: Test points
            Xi: Interior points
            Xb: Boundary points 
            y_obs: Observations
            n_samples: Number of random samples for SDD
            n_iters: Number of SDD iterations
            
        Returns:
            posterior_mean: Mean predictions at test points
            posterior_std: Standard deviations at test points
            A_approx: SDD approximation of C^{-1}y
        """
        # Run SDD to get approximation of C^{-1}y
        A_approx = self.run_sdd(Xi, Xb, y_obs, n_samples, n_iters)
        
        # Compute posterior mean using approximation
        posterior_mean = self.posterior_mean_sdd(X_test, Xi, Xb, A_approx)
        
        # Compute covariance vectors for each test point
        cov_vecs = self.compute_covariance_vector(X_test, Xi, Xb)
        
        # Compute posterior covariance 
        test_pts = X_test.shape[0]
        posterior_var = torch.zeros(test_pts, device=X_test.device)
        
        for i in range(test_pts):
            x_i = X_test[i:i+1]
            x_expand = x_i.unsqueeze(1).repeat(1, x_i.size(0), 1)
            x_prime_expand = x_i.unsqueeze(0).repeat(x_i.size(0), 1, 1)
            base_cov = self.kernel(x_expand, x_prime_expand)[0, 0]
            cov_i = cov_vecs[i:i+1]
            c_x_approx = (cov_i @ A_approx)[0, 0]
            posterior_var[i] = max(base_cov - c_x_approx, 1e-8)  # Ensure positive variance
        
        posterior_std = torch.sqrt(posterior_var).reshape(-1, 1)
        
        return posterior_mean, posterior_std, A_approx

# %%
# Step 4: Generate strictly interior 2D points for the Poisson equation on a circular disc
def generate_2d_disc_data(nb = 10, n=100, r_interior=0.98):
    """
    Generate data for the Poisson equation on a circular disc with radius 1.
    The source term is f(x1, x2) = -1 and the exact solution is u(x1, x2) = (1 - x1^2 - x2^2) / 4.

    :param n: Number of interior points to sample inside the circular disc.
    :param r_interior: Maximum radius for interior points to avoid points on the boundary (default 0.99).
    :return: Xi (interior points), Xb (boundary points), f_Xi (source term at interior points), g_Xb (boundary values).
    """

    # Generate interior points uniformly within the unit disc, avoiding the boundary
    def sample_disc_points(n, r_max):
        theta = torch.rand(n) * 2 * torch.pi  # Random angles
        r = r_max * torch.sqrt(torch.rand(n))  # Radius, scaled to ensure uniform distribution in polar coordinates
        x1 = r * torch.cos(theta)
        x2 = r * torch.sin(theta)
        return torch.stack([x1, x2], dim=1)

    Xi = sample_disc_points(n, r_interior)  # Interior points, avoiding points too close to the boundary

    # Define the source term (f(x1, x2) = -1) at the interior points
    f_Xi = -torch.ones(Xi.shape[0],1)  # The source term is constant (-1)

    # Define the boundary points on the unit circle (discretized)
    num_boundary_points = nb
    theta_b = torch.linspace(0, 2 * torch.pi, num_boundary_points)
    Xb = torch.stack([torch.cos(theta_b), torch.sin(theta_b)], dim=1)  # Boundary points on the unit circle

    # Boundary condition (u = 0 on the boundary for a circular disc)
    g_Xb = torch.zeros(Xb.shape[0], 1)

    return Xi, Xb, f_Xi, g_Xb

# %%
def generate_rays_data(n_rays=8, n_samples_per_ray=10, include_center=True):
    """
    Generate data sampled uniformly along rays till the boundary of a unit circle.

    :param n_rays: Number of rays along which to sample points.
    :param n_samples_per_ray: Number of samples along each ray (including boundary).
    :param include_center: Whether to include the center point (0, 0) in one of the rays.
    :return: Xi (interior points), Xb (boundary points), f_Xi (interior values), g_Xb (boundary values)
    """
    # Define angles for the rays in radians, evenly spaced
    angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)

    # Prepare lists to store interior and boundary points
    interior_points = []
    boundary_points = []
    center_point_included = False  # To track if the center point has been included

    # Sample points along each ray
    for i, theta in enumerate(angles):
        # Sample points uniformly along the ray, including the boundary
        radii = torch.linspace(0, 1, n_samples_per_ray)  # radii from 0 to 1 (inclusive)

        # Convert polar coordinates to Cartesian (x1, x2)
        x1 = radii * torch.cos(torch.tensor(theta))
        x2 = radii * torch.sin(torch.tensor(theta))

        # Stack x1 and x2 into a set of points along the line
        points = torch.stack([x1, x2], dim=1)

        if include_center and not center_point_included:
            # Include the center point (0, 0) in the first ray only
            interior_points.append(points[:-1])  # Include all except the boundary point
            center_point_included = True
        else:
            # Exclude the center point for all other rays
            interior_points.append(points[1:-1])  # Exclude center point and boundary point

        # Add the boundary point (r = 1)
        boundary_points.append(points[-1:])  # The last point is the boundary point (r = 1)

    # Concatenate all interior and boundary points
    Xi = torch.cat(interior_points, dim=0)
    Xb = torch.cat(boundary_points, dim=0)

    # Compute the source term (f(Xi)) for the Poisson equation at the interior points
    # For Poisson equation: f(x1, x2) = -1
    f_Xi = -torch.ones(Xi.shape[0])

    # Compute the boundary values (g(Xb)), which are typically 0 for Dirichlet BC
    # Use the actual solution u(x1, x2) = (1 - x1^2 - x2^2) / 4 on the boundary
    g_Xb = (1 - Xb[:, 0]**2 - Xb[:, 1]**2) / 4

    return Xi, Xb, f_Xi, g_Xb

# Example usage
n_rays = 5  # Number of rays (you can change this)
n_samples_per_ray = 5  # Number of samples along each ray (including boundary)

Xi, Xb, f_Xi, g_Xb = generate_rays_data(n_rays=n_rays, n_samples_per_ray=n_samples_per_ray)

# Step 5: Plot the points on the boundary and interior
plt.figure(figsize=(6, 6))

# Plot interior points
plt.scatter(Xi[:, 0].numpy(), Xi[:, 1].numpy(), color='blue', label='Interior Points')

# Plot boundary points
plt.scatter(Xb[:, 0].numpy(), Xb[:, 1].numpy(), color='red', label='Boundary Points')

# Plot the boundary of the domain (unit circle) for reference
theta_circle = np.linspace(0, 2 * np.pi, 100)
x_circle = np.cos(theta_circle)
y_circle = np.sin(theta_circle)
plt.plot(x_circle, y_circle, color='black', linestyle='--', label='Domain Boundary')

# Formatting the plot
plt.title(f'Interior and Boundary Points Along {n_rays} Rays')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.axis('equal')  # Ensure the aspect ratio is equal for x and y
plt.legend()
plt.grid(True)

# Show plot
plt.show()

# %%
kernel = RBFKernel2D(init_lengthscale=2.5,init_variance = 0.25)
# kernel = RBFKernel2D(init_lengthscale=8.66,init_variance = 5.5)
operators = PDEBoundaryOperators2D(kernel=kernel)
solver = PosteriorSolver2D(kernel=kernel, operators=operators)

# Full covariance matrix and inverse for the observations
C_full = solver.compute_covariance_matrix(Xi.to(torch.float64), Xb.to(torch.float64))

C_inv = torch.inverse(C_full + 1e-6 * torch.eye(C_full.shape[0]))  # Add jitter for stability

# Combine interior and boundary observations
y_obs = torch.cat((f_Xi, g_Xb), dim=0).to(torch.float64)

X_1,X_2,_,_ = generate_2d_disc_data(nb=50,n=600,r_interior=0.99)
X_test = torch.cat((X_1,X_2),dim=0).to(torch.float64)
# X_test = torch.cat((Xi,Xb),dim=0)
# Make predictions at a test point
# x_test = torch.tensor([[0.5, 0.5]], requires_grad=True)
posterior_mean = solver.posterior_mean(X_test, Xi, Xb, C_inv, y_obs).detach()
posterior_cov_full = solver.posterior_covariance(X_test, X_test.clone(), Xi, Xb, C_inv).detach()
posterior_cov = torch.sqrt(torch.diag(posterior_cov_full)).reshape(-1,1)
print(torch.sum(torch.isnan(posterior_cov)))

# %%
# SDD algorithm
torch.manual_seed(1)
N = 21          # Number of data points
du = 1            # Output dimension
input_dim = 1    # Input dimension

sigma_n = 0          # Likelihood variance
T = 1000          # Number of steps
B = 21          # Batch size
beta = 5          # Step size
rho = 0.9           # Momentum parameter
r = 0.9           # Averaging parameter
num_epochs = 4000
# Initialize parameters
def train_SDD(N,du,input_dim,sigma_n, T,B,beta,rho,r,num_epochs,C,y):
  A_t = torch.zeros(N, du,dtype=torch.float64)       # Parameter A_t
  V_t = torch.zeros(N, du,dtype=torch.float64)        # Velocity V_t
  A_bar_t = torch.zeros(N, du,dtype=torch.float64)    # Averaged parameter A_bar_t
  K_full = (C + 1e-6 * torch.eye(C.shape[0]))
  for t in range(num_epochs):
      S = A_t + rho * V_t      # Shape: [N, du]

      # Sample random batch indices
      It = torch.randint(0, N, (B,))

      # Initialize gradient G_t
      G_t = torch.zeros(N, du,dtype=torch.float64)

      # For each index i in the batch
      G_t[It] = (N / B)*K_full[It]@S - y[It]
      V_t = rho * V_t - beta * G_t                  # Update V_t

      # Update parameters A_t
      A_t += V_t

      # Iterative averaging of parameters
      A_bar_t = r * A_t + (1 - r) * A_bar_t

      #(Optional) Compute and print the loss every 100 steps

      if t % 100 == 0 or t == T:
          # Compute the predictions
          pred = K_full @ A_t                  # Shape: [N, du]
          loss_term1 = 0.5 * torch.norm(y - pred) ** 2
          At_K_At = torch.sum(A_t * (K_full @ A_t))
          loss_term2 = (sigma_n / 2) * At_K_At
          L_t = loss_term1 + loss_term2
          print(f"SDD Step  {t}, Loss: {L_t.item():.6e}")
      A_approx = A_bar_t
  return A_approx
with torch.no_grad():
  C_full_jitter = C_full + 1e-6 * torch.eye(C_full.shape[0])
  A_approx = train_SDD(N,du,input_dim,sigma_n, T,B,beta,rho,r,num_epochs,C_full_jitter,y_obs.reshape(-1,1))

# %%
torch.manual_seed(0)
with torch.no_grad():
    C_full_jitter = C_full + 1e-6 * torch.eye(C_full.shape[0])
    cov_vec = solver.compute_covariance_vector(X_test, Xi, Xb).detach()
    A_approx_var = train_SDD(N,cov_vec.shape[0],input_dim,sigma_n, T,B,beta,rho,r,num_epochs,C_full_jitter,cov_vec.T)


# %%
posterior_mean_sdd = solver.posterior_mean_sdd(X_test, Xi, Xb,A_approx).detach()
posterior_cov_sdd = solver.posterior_covariance_sdd(X_test, X_test.clone(), cov_vec, A_approx_var).detach()
posterior_cov_sdd_sqrt = torch.sqrt(torch.diag(posterior_cov_sdd)).reshape(-1,1)
print(torch.sum(torch.isnan(posterior_cov_sdd_sqrt)))


# %%
# Assuming X_test and posterior_mean are already defined
# X_test.shape = (60, 2), posterior_mean.shape = (60, 1)

# Detach the tensors from the computation graph if requires_grad=True
x1 = X_test[:, 0].detach().numpy()  # Extract the x1 (first column)
x2 = X_test[:, 1].detach().numpy()  # Extract the x2 (second column)
posterior_mean_np = posterior_mean_sdd.detach().numpy().reshape(-1)  # Flatten the posterior mean
posterior_cov_np = posterior_cov_sdd_sqrt.detach().numpy().reshape(-1)  # Flatten the posterior covariance

# Create a grid of points for contour plots
# Step 1: Compute the actual solution u(x1, x2) = (1 - x1^2 - x2^2) / 4
actual_solution = (1 - x1**2 - x2**2) / 4

# Step 2: Compute the error between the posterior mean and actual solution
error = np.abs(posterior_mean_np - actual_solution)


#%%
######### plots####################
def setup_matplotlib_for_publication():
    # Use LaTeX for text rendering
    rcParams['text.usetex'] = False
    # rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    
    # Set font to be similar to LaTeX default
    rcParams['font.family'] = 'serif'
    # rcParams['font.serif'] = ['Computer Modern Roman']
    rcParams['font.size'] = 20
    rcParams['axes.labelsize'] = 12
    rcParams['axes.titlesize'] = 14
    rcParams['xtick.labelsize'] = 10
    rcParams['ytick.labelsize'] = 10
    rcParams['legend.fontsize'] = 14
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
    
setup_matplotlib_for_publication()


# Step 5: Create subplots for posterior mean, actual solution, error, and posterior covariance
fig, axes = plt.subplots(2, 2, figsize=(12, 10))  # Create 4 subplots

# Plot the posterior mean
contour_posterior = axes[0, 0].tricontourf(x1, x2, posterior_mean_np, levels=50, cmap='viridis')
fig.colorbar(contour_posterior, ax=axes[0, 0])
axes[0, 0].set_title('Ground truth')
axes[0, 0].set_xlabel('$x_1$')
axes[0, 0].set_ylabel('$x_2$')

# Plot the actual solution
contour_actual = axes[0, 1].tricontourf(x1, x2, actual_solution, levels=50, cmap='viridis')
fig.colorbar(contour_actual, ax=axes[0, 1])
axes[0, 1].set_title('Mean prediction (with SDD)')
axes[0, 1].set_xlabel('$x_1$')
axes[0, 1].set_ylabel('$x_2$')

# Plot the error (posterior mean - actual solution)
contour_error = axes[1, 0].tricontourf(x1, x2, error, levels=50, cmap='coolwarm')
fig.colorbar(contour_error, ax=axes[1, 0])
axes[1, 0].set_title('Absolute Error')
axes[1, 0].set_xlabel('$x_1$')
axes[1, 0].set_ylabel('$x_2$')

# Plot the posterior variance (diagonal of posterior covariance) as a contour plot
contour_variance = axes[1, 1].tricontourf(x1, x2, 2*posterior_cov_np, levels=50, cmap='plasma')
fig.colorbar(contour_variance, ax=axes[1, 1])
axes[1, 1].set_title('Standard deviation')
axes[1, 1].set_xlabel('$x_1$')
axes[1, 1].set_ylabel('$x_2$')

plt.tight_layout()
plt.savefig('sdd_poisson_2d_plot.pdf', format='pdf', bbox_inches='tight')
# %%
np.mean(error)/np.mean(np.abs(actual_solution))

#%%

def save_results(output_dir, x1, x2, posterior_mean, actual_solution, error, posterior_std):
    """
    Save computational results to a NumPy file for later use.
    """
    import os
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results to NumPy file
    np.savez(
        os.path.join(output_dir, 'poisson_2d_results.npz'),
        x1=x1,
        x2=x2,
        posterior_mean=posterior_mean,
        actual_solution=actual_solution,
        error=error,
        posterior_std=posterior_std
    )
    
    print(f"Results saved to {os.path.join(output_dir, 'poisson_2d_results.npz')}")

def create_publication_plot(x1, x2, posterior_mean, actual_solution, error, posterior_std, output_dir=None):
    """
    Create publication-quality plot for the 2D Poisson equation results.
    """
    import os
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    
    # Set up matplotlib for publication quality
    rcParams['font.family'] = 'serif'
    rcParams['font.size'] = 12
    rcParams['axes.labelsize'] = 14
    rcParams['axes.titlesize'] = 16
    rcParams['xtick.labelsize'] = 12
    rcParams['ytick.labelsize'] = 12
    rcParams['legend.fontsize'] = 12
    rcParams['figure.figsize'] = (12, 12)
    rcParams['figure.dpi'] = 300
    rcParams['savefig.dpi'] = 600
    rcParams['savefig.bbox'] = 'tight'
    
    # Create figure and subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Plot the ground truth solution
    contour_actual = axes[0, 0].tricontourf(x1, x2, actual_solution, levels=50, cmap='viridis')
    fig.colorbar(contour_actual, ax=axes[0, 0])
    axes[0, 0].set_title('Ground Truth')
    axes[0, 0].set_xlabel('$x_1$')
    axes[0, 0].set_ylabel('$x_2$')
    
    # Plot the posterior mean (predictions)
    contour_posterior = axes[0, 1].tricontourf(x1, x2, posterior_mean, levels=50, cmap='viridis')
    fig.colorbar(contour_posterior, ax=axes[0, 1])
    axes[0, 1].set_title('Mean Predictions')
    axes[0, 1].set_xlabel('$x_1$')
    axes[0, 1].set_ylabel('$x_2$')
    
    # Plot the absolute error
    contour_error = axes[1, 0].tricontourf(x1, x2, error, levels=50, cmap='coolwarm')
    fig.colorbar(contour_error, ax=axes[1, 0])
    axes[1, 0].set_title('Absolute Error')
    axes[1, 0].set_xlabel('$x_1$')
    axes[1, 0].set_ylabel('$x_2$')
    
    # Plot the posterior standard deviation
    contour_variance = axes[1, 1].tricontourf(x1, x2, 2*posterior_std, levels=50, cmap='plasma')
    fig.colorbar(contour_variance, ax=axes[1, 1])
    axes[1, 1].set_title('Standard Deviation (95% CI)')
    axes[1, 1].set_xlabel('$x_1$')
    axes[1, 1].set_ylabel('$x_2$')
    
    # Add unit circle to all plots
    theta = np.linspace(0, 2*np.pi, 100)
    circle_x = np.cos(theta)
    circle_y = np.sin(theta)
    for ax in axes.flat:
        ax.plot(circle_x, circle_y, 'k--', alpha=0.7, linewidth=1)
        ax.set_aspect('equal')
        ax.set_xlim([-1.1, 1.1])
        ax.set_ylim([-1.1, 1.1])
    
    # Add overall title with relative error
    rel_error = np.mean(error) / np.mean(np.abs(actual_solution))
    fig.suptitle(f'2D Poisson Equation Solution\nRelative Error: {rel_error:.6f}', fontsize=18)
    
    plt.tight_layout()
    fig.subplots_adjust(top=0.92)
    
    # Save figure if output directory is provided
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'poisson_2d_plot.pdf')
        plt.savefig(output_path, format='pdf', bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    
    plt.show()
    
    return fig, axes

# Save the results for later use
output_dir = 'c:\\Users\\sawan\\OneDrive - IIT Delhi\\Code\\adaptive_GP\\results'
save_results(
    output_dir, 
    x1, 
    x2, 
    posterior_mean_np, 
    actual_solution, 
    error, 
    posterior_cov_np
)

# Create and save publication-quality plot
create_publication_plot(
    x1, 
    x2, 
    posterior_mean_np, 
    actual_solution, 
    error, 
    posterior_cov_np,
    output_dir
)

# Display relative error for reference
rel_error = np.mean(error) / np.mean(np.abs(actual_solution))
print(f"Relative Error: {rel_error:.6f}")
#%%