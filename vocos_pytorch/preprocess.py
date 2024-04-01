import glob
import os
from tqdm import tqdm
from dataclasses import dataclass
import torch
from torch import Tensor
from torch.multiprocessing import Pool, set_start_method
import torchaudio
from config import MelConfig,  TrainConfig

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

@dataclass
class DataConfig:
    audio_dir = './audios' # path to audios
    output_dir = './vocos_datasets' # path to save processed audios
    filelist_path = './filelists/filelist.txt' # path to save filelist
            
data_config = DataConfig()
train_config = TrainConfig()
mel_config = MelConfig()
audio_dir = data_config.audio_dir
output_dir = data_config.output_dir
filelist_path = data_config.filelist_path
segment_size = train_config.segment_size

output_audio_dir = os.path.join(output_dir, 'audios')

# Ensure output directories exist
os.makedirs(output_audio_dir, exist_ok=True)
os.makedirs(os.path.dirname(filelist_path), exist_ok=True)

def load_and_resample_audio(audio_path, target_sr, segment_size, device='cpu') -> Tensor:
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
        
    if y.size(-1) < segment_size:
        y = torch.nn.functional.pad(y, (0, segment_size - y.size(-1)), "constant", 0)
        
    return y
    
def find_audio_files(directory) -> list:
    extensions = ['wav', 'mp3', 'flac']
    files_found = []
    for extension in extensions:
        pattern = os.path.join(directory, '**', f'*.{extension}')
        files_found.extend(glob.glob(pattern, recursive=True))
    return files_found

@ torch.inference_mode()
def process_audio(audio_path):
    audio = load_and_resample_audio(audio_path, mel_config.sample_rate, segment_size, device=device) # shape: [1, time]
    if audio is not None:
        
        # get output path
        audio_name, _ = os.path.splitext(os.path.basename(audio_path))
        output_audio_path = os.path.join(output_audio_dir, audio_name + '.wav')
        
        # save resampled audio and mel features
        torchaudio.save(output_audio_path, audio.cpu(), mel_config.sample_rate)
        
        return output_audio_path

def main():
    set_start_method('spawn') # CUDA must use spawn method
    audio_files = find_audio_files(audio_dir)
    results = []
    
    with Pool(processes=8) as pool:
        for result in tqdm(pool.imap(process_audio, audio_files), total=len(audio_files)):
            if result is not None:
                results.append(f'{result}\n') 
            
    # save filelist
    with open(filelist_path, 'w', encoding='utf-8') as f:
        f.writelines(results)
    print(f"filelist file has been saved to {filelist_path}")

# faster and use much less CPU
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
    
if __name__ == '__main__':
    main()