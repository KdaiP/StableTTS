from dataclasses import dataclass

@dataclass
class MelConfig:
    sample_rate: int = 44100
    n_fft: int = 2048
    win_length: int = 2048
    hop_length: int = 512
    f_min: float = 0.0
    f_max: float = None
    pad: int = 0
    n_mels: int = 128
    center: bool = False
    pad_mode: str = "reflect"
    mel_scale: str = "slaney"
    
    def __post_init__(self):
        if self.pad == 0:
            self.pad = (self.n_fft - self.hop_length) // 2
            
@dataclass
class VocosConfig:
    input_channels: int = 128
    dim: int = 768
    intermediate_dim: int = 2048
    num_layers: int = 12
            
@dataclass
class TrainConfig:
    train_dataset_path: str = './filelists/filelist.txt'
    test_dataset_path: str = './filelists/filelist.txt'
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 10000
    model_save_path: str = './checkpoints'
    log_dir: str = './runs'
    log_interval: int = 64
    warmup_steps: int = 200
    
    segment_size = 20480
    mel_loss_factor = 15