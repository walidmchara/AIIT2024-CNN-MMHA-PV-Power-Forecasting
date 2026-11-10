"""Spatial-transformation CNN with causally masked multi-head attention."""
from __future__ import annotations
import math
import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 4096):
        super().__init__()
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
        enc = torch.zeros(max_len, dim)
        enc[:, 0::2] = torch.sin(pos * div)
        enc[:, 1::2] = torch.cos(pos * div[:enc[:, 1::2].shape[1]])
        self.register_buffer('encoding', enc, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) > self.encoding.size(0):
            raise ValueError('sequence exceeds supported positional encoding length')
        return x + self.encoding[:x.size(1)].to(dtype=x.dtype)


class CNNMaskedAttention(nn.Module):
    """The 9→49 spatial projection, 3 CNN blocks, 100-D 20-head MMHA of Table I.

    An independent spatial map is generated at each historical time step.
    Causal attention is applied across those time steps, not spatial pixels.
    The last *historical* token predicts one subsequent PV-power sample.
    """
    def __init__(self, n_features: int = 9, spatial_dim: int = 49,
                 conv_channels: tuple[int,int,int] = (128,256,10),
                 model_dim: int = 100, heads: int = 20, ff_hidden: int = 50):
        super().__init__()
        if spatial_dim != 49: raise ValueError('spatial dimension must be 49 (7x7) in this implementation')
        if model_dim % heads: raise ValueError('model_dim must be divisible by heads')
        self.spatial = nn.Linear(n_features, spatial_dim)
        c1,c2,c3 = conv_channels
        self.cnn = nn.Sequential(
            nn.Conv2d(1,c1,kernel_size=3,stride=2,padding=1), nn.ReLU(),
            nn.Conv2d(c1,c2,kernel_size=3,stride=2,padding=1), nn.ReLU(),
            nn.Conv2d(c2,c3,kernel_size=1,stride=1), nn.ReLU(),
        )
        # 7x7 -> 4x4 -> 2x2; final 1x1 convolution retains 2x2.
        self.to_token = nn.Linear(c3*2*2, model_dim)
        self.position = PositionalEncoding(model_dim)
        self.attention = nn.MultiheadAttention(model_dim, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(model_dim)
        self.ff = nn.Sequential(nn.Linear(model_dim,ff_hidden),nn.ReLU(),nn.Linear(ff_hidden,model_dim))
        self.norm2 = nn.LayerNorm(model_dim)
        self.readout = nn.Linear(model_dim,1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim!=3: raise ValueError('input shape must be (batch, time, 9 weather features)')
        b,t,_ = x.shape
        s = self.spatial(x).reshape(b*t,1,7,7)
        z = self.cnn(s).flatten(1).reshape(b,t,-1)
        z = self.position(self.to_token(z))
        # PyTorch bool attn_mask: True means attention is forbidden.
        mask = torch.ones(t,t,device=x.device,dtype=torch.bool).triu(1)
        a,_ = self.attention(z,z,z,attn_mask=mask,need_weights=False)
        u = self.norm1(z+a)
        return self.norm2(u+self.ff(u))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.readout(self.encode(x)[:,-1]).squeeze(-1)
