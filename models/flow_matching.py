import torch
import torch.nn as nn
import torch.nn.functional as F

import functools
from torchdiffeq import odeint

from models.estimator import Decoder

# modified from https://github.com/shivammehta25/Matcha-TTS/blob/main/matcha/models/components/flow_matching.py
class CFMDecoder(torch.nn.Module):
    def __init__(self, noise_channels, cond_channels, hidden_channels, out_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout, gin_channels):
        super().__init__()
        self.noise_channels = noise_channels
        self.cond_channels = cond_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.gin_channels = gin_channels
        self.sigma_min = 1e-4

        self.estimator = Decoder(noise_channels, cond_channels, hidden_channels, out_channels, filter_channels, p_dropout, n_layers, n_heads, kernel_size, gin_channels)

    @torch.inference_mode()
    def forward(self, mu, mask, n_timesteps, temperature=1.0, c=None, solver=None, cfg_kwargs=None):
        """Forward diffusion

        Args:
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            n_timesteps (int): number of diffusion steps
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.
            c (torch.Tensor, optional): speaker embedding
                shape: (batch_size, gin_channels)
            solver: see https://github.com/rtqichen/torchdiffeq for supported solvers
            cfg_kwargs: used for cfg inference

        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, n_feats, mel_timesteps)
        """
        
        z = torch.randn_like(mu) * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device)
        
        # cfg control
        if cfg_kwargs is None:
            estimator = functools.partial(self.estimator, mask=mask, mu=mu, c=c)
        else:
            estimator = functools.partial(self.cfg_wrapper, mask=mask, mu=mu, c=c, cfg_kwargs=cfg_kwargs)
            
        trajectory = odeint(estimator, z, t_span, method=solver, rtol=1e-5, atol=1e-5)
        return trajectory[-1]
    
    # cfg inference
    def cfg_wrapper(self, t, x, mask, mu, c, cfg_kwargs):
        fake_speaker = cfg_kwargs['fake_speaker'].repeat(x.size(0), 1)
        fake_content = cfg_kwargs['fake_content'].repeat(x.size(0), 1, x.size(-1))
        cfg_strength = cfg_kwargs['cfg_strength']
        
        cond_output = self.estimator(t, x, mask, mu, c)
        uncond_output = self.estimator(t, x, mask, fake_content, fake_speaker)
        
        output = uncond_output + cfg_strength * (cond_output - uncond_output)
        return output

    def compute_loss(self, x1, mask, mu, c):
        """Computes diffusion loss

        Args:
            x1 (torch.Tensor): Target
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): target mask
                shape: (batch_size, 1, mel_timesteps)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            c (torch.Tensor, optional): speaker condition.

        Returns:
            loss: conditional flow matching loss
            y: conditional flow
                shape: (batch_size, n_feats, mel_timesteps)
        """
        b, _, t = mu.shape

        # random timestep
        # use cosine timestep scheduler from cosyvoice: https://github.com/FunAudioLLM/CosyVoice/blob/main/cosyvoice/flow/flow_matching.py
        t = torch.rand([b, 1, 1], device=mu.device, dtype=mu.dtype)
        t = 1 - torch.cos(t * 0.5 * torch.pi)
        
        # sample noise p(x_0)
        z = torch.randn_like(x1)

        y = (1 - (1 - self.sigma_min) * t) * z + t * x1
        u = x1 - (1 - self.sigma_min) * z

        loss = F.mse_loss(self.estimator(t.squeeze(), y, mask, mu, c), u, reduction="sum") / (torch.sum(mask) * u.size(1))
        return loss, y
