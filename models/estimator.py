import math

import torch
import torch.nn as nn

from models.diffusion_transformer import DiTConVBlock

class DitWrapper(nn.Module):
    """ add FiLM layer to condition time embedding to DiT """
    def __init__(self, hidden_channels, filter_channels, num_heads, kernel_size=3, p_dropout=0.1, gin_channels=0, time_channels=0):
        super().__init__()
        self.time_fusion = FiLMLayer(hidden_channels, time_channels)
        self.block = DiTConVBlock(hidden_channels, filter_channels, num_heads, kernel_size, p_dropout, gin_channels)
            
    def forward(self, x, c, t, x_mask):
        x = self.time_fusion(x, t) * x_mask
        x = self.block(x, c, x_mask)
        return x

class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) layer
    Reference: https://arxiv.org/abs/1709.07871
    """
    def __init__(self, in_channels, cond_channels):

        super(FiLMLayer, self).__init__()
        self.in_channels = in_channels
        self.film = nn.Conv1d(cond_channels, in_channels * 2, 1)

    def forward(self, x, c):
        gamma, beta = torch.chunk(self.film(c.unsqueeze(2)), chunks=2, dim=1)
        return gamma * x + beta
    
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        assert self.dim % 2 == 0, "SinusoidalPosEmb requires dim to be even"

    def forward(self, x, scale=1000):
        if x.ndim < 1:
            x = x.unsqueeze(0)
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device).float() * -emb)
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

# reference: https://github.com/shivammehta25/Matcha-TTS/blob/main/matcha/models/components/decoder.py
class Decoder(nn.Module):
    def __init__(self, noise_channels, cond_channels, hidden_channels, out_channels, filter_channels, dropout=0.1, n_layers=1, n_heads=4, kernel_size=3, gin_channels=0, use_lsc=True):
        super().__init__()
        self.noise_channels = noise_channels
        self.cond_channels = cond_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.use_lsc = use_lsc # whether to use unet-like long skip connection

        self.time_embeddings = SinusoidalPosEmb(hidden_channels)
        self.time_mlp = TimestepEmbedding(hidden_channels, hidden_channels, filter_channels)

        self.in_proj = nn.Conv1d(hidden_channels + noise_channels, hidden_channels, 1) # cat noise and encoder output as input
        self.blocks = nn.ModuleList([DitWrapper(hidden_channels, filter_channels, n_heads, kernel_size, dropout, gin_channels, hidden_channels) for _ in range(n_layers)])
        self.final_proj = nn.Conv1d(hidden_channels, out_channels, 1)
        
        # prenet for encoder output
        self.cond_proj = nn.Sequential(
            nn.Conv1d(cond_channels, filter_channels, kernel_size, padding=kernel_size//2),
            nn.SiLU(inplace=True),
            nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size//2), # add about 3M params
            nn.SiLU(inplace=True),
            nn.Conv1d(filter_channels, hidden_channels, kernel_size, padding=kernel_size//2)
        )
        
        if use_lsc:
            assert n_layers % 2 == 0
            self.n_lsc_layers = n_layers // 2
            self.lsc_layers = nn.ModuleList([nn.Conv1d(hidden_channels + hidden_channels, hidden_channels, kernel_size, padding = kernel_size // 2) for _ in range(self.n_lsc_layers)])
            
        self.initialize_weights()

    def initialize_weights(self):
        for block in self.blocks:
            nn.init.constant_(block.block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.block.adaLN_modulation[-1].bias, 0)

    def forward(self, t, x, mask, mu, c):
        """Forward pass of the DiT model.

        Args:
            t (torch.Tensor): timestep, shape (batch_size)
            x (torch.Tensor): noise, shape (batch_size, in_channels, time)
            mask (torch.Tensor): shape (batch_size, 1, time)
            mu (torch.Tensor): output of encoder, shape (batch_size, in_channels, time)
            c (torch.Tensor): shape (batch_size, gin_channels)

        Returns:
            _type_: _description_
        """

        t = self.time_mlp(self.time_embeddings(t))
        mu = self.cond_proj(mu)
        
        x = torch.cat((x, mu), dim=1)
        x = self.in_proj(x)
        
        lsc_outputs = [] if self.use_lsc else None

        for idx, block in enumerate(self.blocks):
            # add long skip connection, see https://arxiv.org/pdf/2209.12152 for more details
            if self.use_lsc:
                if idx < self.n_lsc_layers:
                    lsc_outputs.append(x)
                else:
                    x = torch.cat((x, lsc_outputs.pop()), dim=1)
                    x = self.lsc_layers[idx - self.n_lsc_layers](x)
                    
            x = block(x, c, t, mask)

        output = self.final_proj(x * mask)

        return output * mask