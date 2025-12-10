import torch
import torch.nn as nn


class PoissonOperators(nn.Module):
    """Operators for Poisson equation: ∇²u = f (Laplacian)"""
    def __init__(self, kernel):
        super().__init__()
        self.kernel = kernel
        self._setup_operators()
        
    def _setup_operators(self):
        self.hess_diag_fnx2 = lambda x1, x2: torch.diag(torch.func.hessian(self.kernel, argnums=1)(x1, x2))
        self.grad_fnx2 = lambda x1, x2: torch.vmap(torch.vmap(torch.func.grad(self.kernel, argnums=1), in_dims=(0, 0)), in_dims=(0, 0))(x1, x2)
        self.hess_fnx2 = lambda x1, x2: torch.vmap(torch.vmap(self.hess_diag_fnx2, in_dims=(0, 0)), in_dims=(0, 0))(x1, x2)
        self.hess_diag_fnx1 = lambda x1, x2: torch.diag(torch.func.hessian(self.kernel, argnums=0)(x1, x2))
        self.hess_fnx1 = lambda x1, x2: torch.vmap(torch.vmap(self.hess_diag_fnx1, in_dims=(0, 0)), in_dims=(0, 0))(x1, x2)

    def apply_pde_operator(self, X1, X2):
        X1_expand = X1.unsqueeze(1).repeat(1, X2.size(0), 1).requires_grad_(True)
        X2_expand = X2.unsqueeze(0).repeat(X1.size(0), 1, 1).requires_grad_(True)
        return self._operator_lx1lx2(X1_expand, X2_expand).detach()

    def apply_lb_operator(self, X1, Xb):
        X1_expand = X1.unsqueeze(1).repeat(1, Xb.size(0), 1)
        Xb_expand = Xb.unsqueeze(0).repeat(X1.size(0), 1, 1)
        hessian_K_x1 = self.hess_fnx1(X1_expand, Xb_expand)
        return (hessian_K_x1[:, :, 0] + hessian_K_x1[:, :, 1]).detach()

    def apply_boundary_operator(self, X1, Xb):
        X1_expand = X1.unsqueeze(1).repeat(1, Xb.size(0), 1)
        Xb_expand = Xb.unsqueeze(0).repeat(X1.size(0), 1, 1)
        return self.kernel(X1_expand, Xb_expand)

    def operator_lx2(self, X1, X2):
        hessian_K_x2 = self.hess_fnx2(X1, X2)
        return hessian_K_x2[:, :, 0] + hessian_K_x2[:, :, 1]

    def _operator_lx1lx2(self, X1, X2):
        hess_diag_lx1lx2 = lambda x1, x2: torch.diag(torch.func.hessian(self._hess_funcx2, argnums=0)(x1, x2))
        hess = torch.vmap(torch.vmap(hess_diag_lx1lx2, in_dims=(0, 0)), in_dims=(0, 0))(X1, X2)
        return hess[:, :, 0] + hess[:, :, 1]

    def _hess_funcx2(self, X1, X2):
        hess_op = self.hess_diag_fnx2(X1, X2)
        return torch.sum(hess_op)


class HeatEquationOperators(nn.Module):
    def __init__(self, kernel, alpha=0.01):
        super().__init__()
        self.kernel = kernel
        self.alpha = alpha
        self._setup_operators()
        
    def _setup_operators(self):
        self.hess_diag_fnx2 = lambda x1, x2: torch.diag(torch.func.hessian(self.kernel, argnums=1)(x1, x2))
        self.grad_fnx2 = lambda x1, x2: torch.vmap(torch.vmap(torch.func.grad(self.kernel, argnums=1), in_dims=(0, 0)), in_dims=(0, 0))(x1, x2)
        self.hess_fnx2 = lambda x1, x2: torch.vmap(torch.vmap(self.hess_diag_fnx2, in_dims=(0, 0)), in_dims=(0, 0))(x1, x2)
        self.grad_fnx1 = lambda x1, x2: torch.vmap(torch.vmap(torch.func.grad(self.kernel, argnums=0), in_dims=(0, 0)), in_dims=(0, 0))(x1, x2)
        self.hess_diag_fnx1 = lambda x1, x2: torch.diag(torch.func.hessian(self.kernel, argnums=0)(x1, x2))
        self.hess_fnx1 = lambda x1, x2: torch.vmap(torch.vmap(self.hess_diag_fnx1, in_dims=(0, 0)), in_dims=(0, 0))(x1, x2)

    def apply_pde_operator(self, X1, X2):
        X1_expand = X1.unsqueeze(1).repeat(1, X2.size(0), 1).requires_grad_(True)
        X2_expand = X2.unsqueeze(0).repeat(X1.size(0), 1, 1).requires_grad_(True)
        return self._operator_lx1lx2(X1_expand, X2_expand).detach()

    def apply_lb_operator(self, X1, Xd, Xn):
        X1d_expand = X1.unsqueeze(1).repeat(1, Xd.size(0), 1)
        Xd_expand = Xd.unsqueeze(0).repeat(X1.size(0), 1, 1)
        hessian_K_x1 = self.hess_fnx1(X1d_expand, Xd_expand)
        grad_K_x1 = self.grad_fnx1(X1d_expand, Xd_expand)
        K_xb_dir = grad_K_x1[:, :, 1] - self.alpha * hessian_K_x1[:, :, 0]

        X1n_expand = X1.unsqueeze(1).repeat(1, Xn.size(0), 1)
        Xn_expand = Xn.unsqueeze(0).repeat(X1.size(0), 1, 1)
        grad_lb = lambda x1, x2: torch.func.grad(self._hessgrad_funcx1, argnums=1)(x1, x2)
        K_xb_neu = torch.vmap(torch.vmap(grad_lb, in_dims=(0, 0)), in_dims=(0, 0))(X1n_expand, Xn_expand)[:, :, 1]

        return torch.cat((K_xb_dir, K_xb_neu), dim=1).detach()

    def apply_boundary_operator(self, Xd, Xd_clone, Xn, Xn_clone):
        Xd_expand = Xd.unsqueeze(1).repeat(1, Xd_clone.size(0), 1)
        Xdc_expand = Xd_clone.unsqueeze(0).repeat(Xd.size(0), 1, 1)
        K_dir = self.kernel(Xd_expand, Xdc_expand)
        
        Xn_expand = Xn.unsqueeze(1).repeat(1, Xn_clone.size(0), 1)
        Xnc_expand = Xn_clone.unsqueeze(0).repeat(Xn.size(0), 1, 1)
        K_neu = self._operator_gradx1x2(Xn_expand, Xnc_expand)

        Xdn_expand = Xd.unsqueeze(1).repeat(1, Xn.size(0), 1)
        Xdnc_expand = Xn.unsqueeze(0).repeat(Xd.size(0), 1, 1)
        K_dn = self.grad_fnx2(Xdn_expand, Xdnc_expand)[:, :, 1]
        
        return torch.cat((torch.cat((K_dir, K_dn), dim=1), torch.cat((K_dn.T, K_neu), dim=1)), dim=0).detach()

    def operator_lx2(self, X1, X2):
        hessian_K_x2 = self.hess_fnx2(X1, X2)
        grad_K_x2 = self.grad_fnx2(X1, X2)
        return -self.alpha * hessian_K_x2[:, :, 0] + grad_K_x2[:, :, 1]

    def _operator_gradx1x2(self, X1, X2):
        grad_op = lambda x1, x2: torch.func.grad(self.kernel, argnums=1)(x1, x2)[1]
        grad_op2 = lambda x1, x2: torch.func.grad(grad_op, argnums=0)(x1, x2)
        return torch.vmap(torch.vmap(grad_op2, in_dims=(0, 0)), in_dims=(0, 0))(X1, X2)[:, :, 1]

    def _operator_lx1lx2(self, X1, X2):
        hess_diag_lx1lx2 = lambda x1, x2: torch.diag(torch.func.hessian(self._hessgrad_funcx2, argnums=0)(x1, x2))
        hess = torch.vmap(torch.vmap(hess_diag_lx1lx2, in_dims=(0, 0)), in_dims=(0, 0))(X1, X2)
        grad_op = lambda x1, x2: torch.func.grad(self._hessgrad_funcx2, argnums=0)(x1, x2)
        grad = torch.vmap(torch.vmap(grad_op, in_dims=(0, 0)), in_dims=(0, 0))(X1, X2)
        return grad[:, :, 1] - self.alpha * hess[:, :, 0]

    def _hessgrad_funcx2(self, X1, X2):
        grad_op = torch.func.grad(self.kernel, argnums=1)(X1, X2)
        hess_op = self.hess_diag_fnx2(X1, X2)
        return grad_op[1] - self.alpha * hess_op[0]

    def _hessgrad_funcx1(self, X1, X2):
        grad_op = torch.func.grad(self.kernel, argnums=0)(X1, X2)
        hess_op = self.hess_diag_fnx1(X1, X2)
        return grad_op[1] - self.alpha * hess_op[0]
