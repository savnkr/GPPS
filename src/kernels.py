import torch
import torch.nn as nn


class RBFKernel(nn.Module):
    def __init__(self, lengthscale=1.0, variance=1.0, device=None):
        super().__init__()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lengthscale = nn.Parameter(torch.as_tensor(lengthscale, dtype=torch.float64, device=self.device))
        self.variance = nn.Parameter(torch.as_tensor(variance, dtype=torch.float64, device=self.device))

    def forward(self, X1, X2):
        lengthscale = torch.clamp(self.lengthscale, min=1e-6)
        diff = (X1 - X2) / lengthscale
        dist_sq = diff.pow(2).sum(-1)
        return (self.variance**2) * torch.exp(-0.5 * dist_sq)

    @staticmethod
    def _coerce_value(value, reference):
        tensor = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
        if tensor.numel() == 1 and reference.numel() > 1:
            tensor = tensor.repeat(reference.numel())
        return tensor.reshape_as(reference)

    def set_lengthscale(self, value):
        with torch.no_grad():
            self.lengthscale.copy_(self._coerce_value(value, self.lengthscale))

    def set_variance(self, value):
        with torch.no_grad():
            self.variance.copy_(self._coerce_value(value, self.variance))
    
    def to(self, device):
        self.device = device
        return super().to(device)
