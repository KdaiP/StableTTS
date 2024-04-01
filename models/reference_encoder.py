import torch
import torch.nn as nn
    
class Conv1dGLU(nn.Module):
    """
    Conv1d + GLU(Gated Linear Unit) with residual connection.
    For GLU refer to https://arxiv.org/abs/1612.08083 paper.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dropout):
        super(Conv1dGLU, self).__init__()
        self.out_channels = out_channels
        self.conv1 = nn.Conv1d(in_channels, 2 * out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x1, x2 = torch.split(x, self.out_channels, dim=1)
        x = x1 * torch.sigmoid(x2)
        x = residual + self.dropout(x)
        return x
    
class MelStyleEncoder(nn.Module):
    """MelStyleEncoder"""

    def __init__(
        self,
        n_mel_channels=80,
        style_hidden=128,
        style_vector_dim=256,
        style_kernel_size=5,
        style_head=2,
        dropout=0.1,
    ):
        super(MelStyleEncoder, self).__init__()
        self.in_dim = n_mel_channels
        self.hidden_dim = style_hidden
        self.out_dim = style_vector_dim
        self.kernel_size = style_kernel_size
        self.n_head = style_head
        self.dropout = dropout

        self.spectral = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.Mish(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Mish(inplace=True),
            nn.Dropout(self.dropout),
        )

        self.temporal = nn.Sequential(
            Conv1dGLU(self.hidden_dim, self.hidden_dim, self.kernel_size, self.dropout),
            Conv1dGLU(self.hidden_dim, self.hidden_dim, self.kernel_size, self.dropout),
        )

        self.slf_attn = nn.MultiheadAttention(
            self.hidden_dim,
            self.n_head,
            self.dropout,
            batch_first=True
        )

        self.fc = nn.Linear(self.hidden_dim, self.out_dim)

    def temporal_avg_pool(self, x, mask=None):
        if mask is None:
            return torch.mean(x, dim=1)
        else:
            len_ = (~mask).sum(dim=1).unsqueeze(1).type_as(x)
            return torch.sum(x * ~mask.unsqueeze(-1), dim=1) / len_

    def forward(self, x, x_mask=None):
        x = x.transpose(1, 2)

        # spectral
        x = self.spectral(x)
        # temporal
        x = x.transpose(1, 2)
        x = self.temporal(x)
        x = x.transpose(1, 2)
        # self-attention
        if x_mask is not None:
            x_mask = ~x_mask.squeeze(1).to(torch.bool)   
        x, _ = self.slf_attn(x, x, x, key_padding_mask=x_mask)
        # fc
        x = self.fc(x)
        # temoral average pooling
        w = self.temporal_avg_pool(x, mask=x_mask)

        return w