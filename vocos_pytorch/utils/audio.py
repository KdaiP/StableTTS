from dataclasses import dataclass, asdict
import torch
from torch import Tensor
import torch.nn as nn
import torchaudio
import torchaudio.transforms

from config import MelConfig

class LogMelSpectrogram(nn.Module):
    def __init__(self, config: MelConfig):
        super().__init__()
        self.spec = torchaudio.transforms.MelSpectrogram(**asdict(config))
        
    def forward(self, x: Tensor) -> Tensor:
        return self.compress(self.spec(x))
        
    def compress(self, x: Tensor) -> Tensor:
        return torch.log(torch.clamp(x, min=1e-5))

    def decompress(self, x: Tensor) -> Tensor:
        return torch.exp(x)