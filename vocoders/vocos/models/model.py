from dataclasses import dataclass, asdict

import torch
from torch import nn
from torch import Tensor

from .head import ISTFTHead
from .backbone import VocosBackbone
from config import MelConfig, VocosConfig
    
class Vocos(nn.Module):
    def __init__(self, vocos_config: VocosConfig, mel_config: MelConfig):
        super().__init__()
        self.backbone = VocosBackbone(**asdict(vocos_config))
        self.head = ISTFTHead(vocos_config.dim, mel_config.n_fft, mel_config.hop_length)
        
    def forward(self, x: Tensor) -> Tensor:
        x = self.backbone(x)
        x = self.head(x)
        return x
