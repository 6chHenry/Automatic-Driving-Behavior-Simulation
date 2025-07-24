import math
from typing import List, Optional

import torch
import torch.nn as nn

from src.smart.utils import weight_init


class FourierEmbedding(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, num_freq_bands: int) -> None:
        super(FourierEmbedding, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.freqs = nn.Embedding(input_dim, num_freq_bands) if input_dim != 0 else None   # 为每个连续特征学习独立的频率参数 [input_dim, num_freq_bands]
        self.mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(num_freq_bands * 2 + 1, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(input_dim)
            ]
        )  # 每个特征有自己的mlp
        self.to_out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.apply(weight_init)

    def forward(
        self,
        continuous_inputs: Optional[torch.Tensor] = None,
        categorical_embs: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        if continuous_inputs is None:
            if categorical_embs is not None:
                x = torch.stack(categorical_embs).sum(dim=0)
            else:
                raise ValueError("Both continuous_inputs and categorical_embs are None")
            
        else: 
            # self.freqs.weight [input_dim, num_freqs_bands]      broadcast      [batch_size, input_dim, 1] * [input_dim, num_freqs_bands] = [batch_size, input_dim, num_freqs_bands]
            x = continuous_inputs.unsqueeze(-1) * self.freqs.weight * 2 * math.pi         
            # Warning: if your data are noisy, don't use learnable sinusoidal embedding
            x = torch.cat([x.cos(), x.sin(), continuous_inputs.unsqueeze(-1)], dim=-1)   # [batch_size, input_dim, 2*num_freqs_bands+1]
            continuous_embs: List[Optional[torch.Tensor]] = [None] * self.input_dim

            for i in range(self.input_dim):
                continuous_embs[i] = self.mlps[i](x[:, i])     # input_dim 个 [batch_size, hidden_dim]
            x = torch.stack(continuous_embs).sum(dim=0)        # 特征融合得到  [batch_size, hidden_dim]

            if categorical_embs is not None:
                x = x + torch.stack(categorical_embs).sum(dim=0)
        return self.to_out(x)   # [batch_size, hidden_dim]


class MLPEmbedding(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super(MLPEmbedding, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.apply(weight_init)

    def forward(
        self,
        continuous_inputs: Optional[torch.Tensor] = None,   # Optional[X]：表示该变量可以是类型 X 或 None     连续特征
        categorical_embs: Optional[List[torch.Tensor]] = None,   # 分类特征
    ) -> torch.Tensor:
        if continuous_inputs is None:
            if categorical_embs is not None:
                x = torch.stack(categorical_embs).sum(dim=0)    # [batch_size, input_dim]
            else:
                raise ValueError("Both continuous_inputs and categorical_embs are None")
            
        else:
            x = self.mlp(continuous_inputs)
            if categorical_embs is not None:
                x = x + torch.stack(categorical_embs).sum(dim=0)   # 保证两类shape相同才可以相加
        return x   # [batch_size, hidden_dim]
