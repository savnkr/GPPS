import torch
import torch.nn as nn


class GPPoissonSolver(nn.Module):
    """GP solver for Poisson equation with Dirichlet boundary conditions only."""
    def __init__(self, kernel, operators, noise_variance=1e-4):
        super().__init__()
        self.kernel = kernel
        self.operators = operators
        self.noise_variance = noise_variance
        self.device = getattr(kernel, 'device', torch.device('cpu'))

    def compute_covariance_matrix(self, Xi, Xb):
        Xi_clone = Xi.clone().detach()
        Xb_clone = Xb.clone().detach()
        C_ii = self.operators.apply_pde_operator(Xi, Xi_clone)
        C_ib = self.operators.apply_lb_operator(Xi, Xb)
        C_bb = self.operators.apply_boundary_operator(Xb, Xb_clone)
        C_full = torch.cat([
            torch.cat([C_ii, C_ib], dim=1),
            torch.cat([C_ib.T, C_bb], dim=1)
        ], dim=0)
        return C_full

    def compute_covariance_vector(self, x, Xi, Xb):
        x_expand = x.unsqueeze(1).repeat(1, Xi.size(0), 1).requires_grad_(True)
        xb_expand = x.unsqueeze(1).repeat(1, Xb.size(0), 1)
        Xi_expand = Xi.unsqueeze(0).repeat(x.size(0), 1, 1).requires_grad_(True)
        Xb_expand = Xb.unsqueeze(0).repeat(x.size(0), 1, 1)
        L_K_x = self.operators.operator_lx2(x_expand, Xi_expand)
        B_K_x = self.kernel(xb_expand, Xb_expand)
        return torch.cat((L_K_x.detach(), B_K_x.detach()), dim=1)

    def posterior_mean(self, x, Xi, Xb, C_inv, y):
        c_x = self.compute_covariance_vector(x, Xi, Xb)
        return c_x @ C_inv @ y

    def posterior_covariance(self, x, x_prime, Xi, Xb, C_inv):
        c_x = self.compute_covariance_vector(x, Xi, Xb)
        x_expand = x.unsqueeze(1).repeat(1, x_prime.size(0), 1)
        x_prime_expand = x_prime.unsqueeze(0).repeat(x.size(0), 1, 1)
        base_cov = self.kernel(x_expand, x_prime_expand)
        c_x_cinv_cxprime = c_x @ C_inv @ c_x.T
        return base_cov - c_x_cinv_cxprime

    def compute_inverse(self, C_full, jitter=1e-6):
        C_reg = C_full + jitter * torch.eye(C_full.shape[0], device=C_full.device, dtype=C_full.dtype)
        return torch.inverse(C_reg)


class GPPDESolver(nn.Module):
    """GP solver for PDEs with mixed Dirichlet/Neumann boundary conditions (e.g., heat equation)."""
    def __init__(self, kernel, operators, noise_variance=1e-4):
        super().__init__()
        self.kernel = kernel
        self.operators = operators
        self.noise_variance = noise_variance
        self.device = getattr(kernel, 'device', torch.device('cpu'))

    def compute_covariance_matrix(self, Xi, Xd, Xn):
        Xi_clone = Xi.clone().detach()
        Xd_clone = Xd.clone().detach()
        Xn_clone = Xn.clone().detach()

        C_ii = self.operators.apply_pde_operator(Xi, Xi_clone)
        C_ib = self.operators.apply_lb_operator(Xi, Xd, Xn)
        C_bb = self.operators.apply_boundary_operator(Xd, Xd_clone, Xn, Xn_clone)

        C_full = torch.cat([
            torch.cat([C_ii, C_ib], dim=1),
            torch.cat([C_ib.T, C_bb], dim=1)
        ], dim=0)
        return C_full

    def compute_covariance_vector(self, x, Xi, Xd, Xn):
        x_expand = x.unsqueeze(1).repeat(1, Xi.size(0), 1).requires_grad_(True)
        xd_expand = x.unsqueeze(1).repeat(1, Xd.size(0), 1)
        Xi_expand = Xi.unsqueeze(0).repeat(x.size(0), 1, 1).requires_grad_(True)
        Xd_expand = Xd.unsqueeze(0).repeat(x.size(0), 1, 1)
        Xn_expand = Xn.unsqueeze(0).repeat(x.size(0), 1, 1).requires_grad_(True)
        xn_expand = x.unsqueeze(1).repeat(1, Xn.size(0), 1)

        L_K_x = self.operators.operator_lx2(x_expand, Xi_expand)
        B_K_x_dir = self.kernel(xd_expand, Xd_expand)
        B_K_x_neu = self.operators.grad_fnx2(xn_expand, Xn_expand)[:, :, 1]

        return torch.cat((L_K_x.detach(), B_K_x_dir.detach(), B_K_x_neu.detach()), dim=1)

    def posterior_mean(self, x, Xi, Xd, Xn, C_inv, y):
        c_x = self.compute_covariance_vector(x, Xi, Xd, Xn)
        return c_x @ C_inv @ y

    def posterior_covariance(self, x, x_prime, Xi, Xd, Xn, C_inv):
        c_x = self.compute_covariance_vector(x, Xi, Xd, Xn)
        x_expand = x.unsqueeze(1).repeat(1, x_prime.size(0), 1)
        x_prime_expand = x_prime.unsqueeze(0).repeat(x.size(0), 1, 1)
        base_cov = self.kernel(x_expand, x_prime_expand)
        c_x_cinv_cxprime = c_x @ C_inv @ c_x.T
        return base_cov - c_x_cinv_cxprime

    def compute_inverse(self, C_full, jitter=1e-6):
        C_reg = C_full + jitter * torch.eye(C_full.shape[0], device=C_full.device, dtype=C_full.dtype)
        return torch.inverse(C_reg)

