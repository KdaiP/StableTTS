import torch
import torch.nn as nn

from models.diffusion_transformer import DiTConVBlock
from utils.mask import sequence_mask

# modified from https://github.com/jaywalnut310/vits/blob/main/models.py
class TextEncoder(nn.Module):
    def __init__(self, n_vocab, out_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout, gin_channels):
        super().__init__()
        self.n_vocab = n_vocab
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        
        self.scale = self.hidden_channels ** 0.5

        self.emb = nn.Embedding(n_vocab, hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)

        self.encoder = nn.ModuleList([DiTConVBlock(hidden_channels, filter_channels, n_heads, kernel_size, p_dropout, gin_channels) for _ in range(n_layers)])
        self.proj = nn.Conv1d(hidden_channels, out_channels, 1)
        
        self.initialize_weights()
        
    def initialize_weights(self):
        for block in self.encoder:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    def forward(self, x: torch.Tensor, c: torch.Tensor, x_lengths: torch.Tensor):
        x = self.emb(x) * self.scale  # [b, t, h]
        x = x.transpose(1, -1)  # [b, h, t]
        x_mask = sequence_mask(x_lengths, x.size(2)).unsqueeze(1).to(x.dtype)

        for layer in self.encoder:
            x = layer(x, c, x_mask)
        mu_x = self.proj(x) * x_mask

        return x, mu_x, x_mask
