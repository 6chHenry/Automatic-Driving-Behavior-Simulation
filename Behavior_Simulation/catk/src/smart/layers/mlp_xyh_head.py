import torch
import torch.nn as nn
from src.smart.utils import weight_init

class MLPHead(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(MLPHead, self).__init__()
        self.mlp=nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim, output_dim)
        )
        self.apply(weight_init)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)