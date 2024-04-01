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
    power: float = 1.0
    normalized: bool = False
    center: bool = False
    pad_mode: str = "reflect"
    mel_scale: str = "htk"
    
    def __post_init__(self):
        if self.pad == 0:
            self.pad = (self.n_fft - self.hop_length) // 2
            
@dataclass
class VocosConfig:
    input_channels: int = 128
    dim: int = 512
    intermediate_dim: int = 1536
    num_layers: int = 8
            
@dataclass
class TrainConfig:
    train_dataset_path: str = './filelists/filelist.txt'
    test_dataset_path: str = './filelists/filelist.txt'
    batch_size: int = 22
    learning_rate: float = 1e-4
    num_epochs: int = 10000
    model_save_path: str = './checkpoints'
    log_dir: str = './runs'
    log_interval: int = 128
    warmup_steps: int = 200
    
    segment_size = 20480