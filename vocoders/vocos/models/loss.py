import torch
import torch.nn as nn
from typing import List
from dataclasses import asdict

from utils.audio import LogMelSpectrogram
from config import MelConfig

# Adapted from https://github.com/descriptinc/descript-audio-codec/blob/main/dac/nn/loss.py under the MIT license.
class MultiScaleMelSpectrogramLoss(nn.Module):
    def __init__(self, n_mels: List[int] = [5, 10, 20, 40, 80, 160, 320], window_lengths: List[int] = [32, 64, 128, 256, 512, 1024, 2048]):
        super().__init__()
        assert len(n_mels) == len(window_lengths), "n_mels and window_lengths must have the same length"
        self.mel_transforms = nn.ModuleList(self._get_transforms(n_mels, window_lengths))
        self.loss_fn = nn.L1Loss()

    def _get_transforms(self, n_mels, window_lengths):
        transforms = []
        for n_mel, win_length in zip(n_mels, window_lengths):
            transform = LogMelSpectrogram(**asdict(MelConfig(n_mels=n_mel, n_fft=win_length, win_length=win_length, hop_length=win_length//4)))
            transforms.append(transform)
        return transforms
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return sum(self.loss_fn(mel_transform(x), mel_transform(y)) for mel_transform in self.mel_transforms)
    
class SingleScaleMelSpectrogramLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel_transform = LogMelSpectrogram(**asdict(MelConfig()))
        self.loss_fn = nn.L1Loss()
        print('using single mel loss')
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(self.mel_transform(x), self.mel_transform(y))

def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))

    return loss*2

def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1-dr)**2)
        g_loss = torch.mean(dg**2)
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses

def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1-dg)**2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses