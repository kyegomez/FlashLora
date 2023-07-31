import torch
from torch import nn


#helper exists(
def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


class Lora(nn.Module):
    def __init__(
            self,
            dim,
            dim_out,
            r=8,
            alpha=None,
    ):
        super().__init__()
        self.linear = nn.Linear(dim, dim_out)
        alpha = default(alpha, r)
        self.scale = alpha / r


        self.A = nn.Parameter(torch.randn(dim, r))
        self.B = nn.Parameter(torch.randn(r, dim_out))
    
    @property
    def weight(self):
        return (self.A @ self.B) * self.scale
    
    def forward(self, x):
        x = self.linear(x) #apply the linear layer
        return x @ self.weight
    
