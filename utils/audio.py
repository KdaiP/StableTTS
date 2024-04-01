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

    
def load_and_resample_audio(audio_path, target_sr, device='cpu') -> Tensor:
    try:
        y, sr = torchaudio.load(audio_path)
    except Exception as e:
        print(str(e))
        return None
    
    y.to(device)
    # Convert to mono
    if y.size(0) > 1:
        y = y[0, :].unsqueeze(0) # shape: [2, time] -> [time] -> [1, time]
        
    # resample audio to target sample_rate
    if sr != target_sr:
        y = torchaudio.functional.resample(y, sr, target_sr)
    return y