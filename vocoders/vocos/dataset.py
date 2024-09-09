import os
from dataclasses import asdict
import torch
import torchaudio
from torch.utils.data import Dataset

from utils.audio import LogMelSpectrogram
from config import MelConfig
    
class VocosDataset(Dataset):
    def __init__(self, filelist_path, segment_size: int, mel_config: MelConfig):
        self.filelist_path = filelist_path     
        self.segment_size = segment_size
        self.sample_rate = mel_config.sample_rate
        self.mel_extractor = LogMelSpectrogram(**asdict(mel_config))
        
        self.filelist = self._load_filelist(filelist_path)

    def _load_filelist(self, filelist_path):
        if os.path.isdir(filelist_path):
            print('scanning dir to get audio files')
            filelist = find_audio_files(filelist_path)
        else:
            with open(filelist_path, 'r', encoding='utf-8') as f:
                filelist = [line.strip() for line in f if os.path.exists(line.strip())]
        return filelist
    
    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        audio = load_and_pad_audio(self.filelist[idx], self.sample_rate, self.segment_size)
        start_index = torch.randint(0, audio.size(-1) - self.segment_size + 1, (1,)).item()
        audio = audio[:, start_index:start_index + self.segment_size] # shape: [1, segment_size]
        mel = self.mel_extractor(audio).squeeze(0) # shape: [n_mels, segment_size // hop_length]
        return audio, mel
    
def load_and_pad_audio(audio_path, target_sr, segment_size):
    y, sr = torchaudio.load(audio_path)
    if y.size(0) > 1:
        y = y[0, :].unsqueeze(0)
    if sr != target_sr:
        y = torchaudio.functional.resample(y, sr, target_sr)
    if y.size(-1) < segment_size:
        y = torch.nn.functional.pad(y, (0, segment_size - y.size(-1)), "constant", 0)
    return y

def find_audio_files(directory):
    audio_files = []
    valid_extensions = ('.wav', '.ogg', '.opus', '.mp3', '.flac')
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(valid_extensions):
                audio_files.append(os.path.join(root, file))
    
    return audio_files