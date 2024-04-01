import os
import torch
import torchaudio
from torch.utils.data import Dataset

from utils.audio import LogMelSpectrogram
from config import MelConfig
    
class VocosDataset(Dataset):
    def __init__(self, filelist_path, segment_size: int, mel_config: MelConfig):
        self.filelist_path = filelist_path     
        self.segment_size = segment_size
        self.mel_extractor = LogMelSpectrogram(mel_config)
        
        self.filelist = self._load_filelist(filelist_path)

    def _load_filelist(self, filelist_path):
        with open(filelist_path, 'r', encoding='utf-8') as f:
            filelist = [line.strip() for line in f if os.path.exists(line.strip())]
        return filelist
    
    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        audio, _ = torchaudio.load(self.filelist[idx])
        
        # select a random segment from the audio file
        # audio is validated in the preprocess stage, so we skip checking sample_rate and padding short audio
        start_index = torch.randint(0, audio.size(-1) - self.segment_size + 1, (1,)).item()
        audio = audio[:, start_index:start_index + self.segment_size] # shape: [1, segment_size]
        mel = self.mel_extractor(audio).squeeze(0) # shape: [n_mels, segment_size // hop_length]
        return audio, mel