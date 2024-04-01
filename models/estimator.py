import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import pack

from models.dit import DiTConVBlock

class DitWrapper(nn.Module):
    def __init__(self, hidden_channels, filter_channels, num_heads, kernel_size=3, p_dropout=0.1, gin_channels=0, time_channels=0):
        super().__init__()
        self.block = DiTConVBlock(hidden_channels, filter_channels, num_heads, kernel_size, p_dropout, gin_channels)
        self.time_fusion = FiLMLayer(hidden_channels, time_channels)
            
    def forward(self, x, c, t, x_mask):
        x = self.time_fusion(x, t)
        x = self.block(x, c, x_mask)
        return x

class FiLMLayer(nn.Module):
    def __init__(self, in_channels, cond_channels):

        super(FiLMLayer, self).__init__()
        self.in_channels = in_channels
        self.film = nn.Conv1d(cond_channels, in_channels * 2, 1)

    def forward(self, x, c):
        film_params = self.film(c.unsqueeze(2))
        gamma, beta = torch.chunk(film_params, chunks=2, dim=1)
        
        return gamma * x + beta

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        assert self.dim % 2 == 0, "SinusoidalPosEmb requires dim to be even"

    def forward(self, x, scale=1000):
        if x.ndim < 1:
            x = x.unsqueeze(0)
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb



class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels, filter_channels):
        super().__init__()

        self.layer = nn.Sequential(
            nn.Linear(in_channels, filter_channels),
            nn.SiLU(inplace=True),
            nn.Linear(filter_channels, out_channels)
        )

    def forward(self, x):
        return self.layer(x)


class Decoder(nn.Module):
    def __init__(self, hidden_channels, out_channels, filter_channels, dropout=0.05, n_layers=1, n_heads=4, kernel_size=3, gin_channels=0):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels

        self.time_embeddings = SinusoidalPosEmb(hidden_channels)
        self.time_mlp = TimestepEmbedding(hidden_channels, hidden_channels, filter_channels)

        
        self.blocks = nn.ModuleList([DitWrapper(hidden_channels, filter_channels, n_heads, kernel_size, dropout, gin_channels, hidden_channels) for _ in range(n_layers)])
        self.final_proj = nn.Conv1d(hidden_channels, out_channels, 1)

        self.initialize_weights()

    def initialize_weights(self):
        for block in self.blocks:
            nn.init.constant_(block.block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.block.adaLN_modulation[-1].bias, 0)

    def forward(self, x, mask, mu, t, c):
        """Forward pass of the UNet1DConditional model.

        Args:
            x (torch.Tensor): shape (batch_size, in_channels, time)
            mask (_type_): shape (batch_size, 1, time)
            t (_type_): shape (batch_size)
            spks (_type_, optional): shape: (batch_size, condition_channels). Defaults to None.
            cond (_type_, optional): placeholder for future use. Defaults to None.

        Raises:
            ValueError: _description_
            ValueError: _description_

        Returns:
            _type_: _description_
        """

        t = self.time_embeddings(t)
        t = self.time_mlp(t)

        x = pack([x, mu], "b * t")[0]

        for block in self.blocks:
            x = block(x, c, t, mask)

        output = self.final_proj(x *mask)

        return output * mask