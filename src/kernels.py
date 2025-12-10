import torch
import torch.nn as nn


class RBFKernel(nn.Module):
    def __init__(self, lengthscale=1.0, variance=1.0, device=None):
        super().__init__()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lengthscale = nn.Parameter(torch.tensor(lengthscale, device=self.device))
        self.variance = nn.Parameter(torch.tensor(variance, device=self.device))
        
    def forward(self, X1, X2):
        diff = X1 - X2
        dist_sq = diff.pow(2).sum(-1)
        return (self.variance**2) * torch.exp(-0.5 * dist_sq / (self.lengthscale**2))
    
    def to(self, device):
        self.device = device
        return super().to(device)
