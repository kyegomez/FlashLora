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
            r=8,
            alpha=None,
    ):
        super().__init__()
        alpha = default(alpha, r)
        self.scale = alpha / r

        self.A = None
        self.B = None
    
    @property
    def weight(self):
        return (self.A @ self.B) * self.scale
    
    def forward(self, x):
        dim = x.shape[-1]
        r = int(self.scale)  # Convert r to an integer
        if self.A is None or self.A.shape[0] != dim:
            self.A = nn.Parameter(torch.randn(dim, r))
            self.B = nn.Parameter(torch.randn(r, dim))
        x = self.linear(x) #apply the linear layer
        return x @ self.weight