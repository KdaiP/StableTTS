import os
import json
from dataclasses import dataclass
from tqdm import tqdm
import torch
from torch.multiprocessing import Pool, set_start_method
import torchaudio

from config import MelConfig, TrainConfig
from text.mandarin import chinese_to_cnm3
from text.english import english_to_ipa2
from text.japanese import japanese_to_ipa2
from utils.audio import LogMelSpectrogram, load_and_resample_audio

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

@dataclass
class DataConfig:
    input_filelist_path = './filelists/filelist.txt' # a filelist contains 'audiopath | text'
    output_filelist_path = './filelists/filelist.json' # path to save filelist
    output_feature_path = './stableTTS_datasets' # path to save resampled audios and mel features
    language = 'japanese' # chinese, japanese, english
    resample = False # waveform is not used in training. However, it is used to calculate length for DistributedBucketSampler in training. Different samplerate or format may cause wrong bucket.

g2p_mapping = {
    'chinese': chinese_to_cnm3,
    'japanese': japanese_to_ipa2,
    'english': english_to_ipa2,
}
            
data_config = DataConfig()
train_config = TrainConfig()
mel_config = MelConfig()

input_filelist_path = data_config.input_filelist_path
output_filelist_path = data_config.output_filelist_path
output_feature_path = data_config.output_feature_path

# Ensure output directories exist
output_mel_dir = os.path.join(output_feature_path, 'mels')
os.makedirs(output_mel_dir, exist_ok=True)
if data_config.resample:
    output_wav_dir = os.path.join(output_feature_path, 'waves')
    os.makedirs(output_wav_dir, exist_ok=True)
os.makedirs(os.path.dirname(output_filelist_path), exist_ok=True)

mel_extractor = LogMelSpectrogram(mel_config).to(device)

# 使用字典来获取相应的函数
g2p = g2p_mapping.get(data_config.language)
    
def load_filelist(path) -> list:
    file_list = []
    with open(path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            audio_path, text = line.strip().split('|', maxsplit=1)
            file_list.append((str(idx), audio_path, text))
    return file_list

@ torch.inference_mode()
def process_filelist(line) -> str:
    idx, audio_path, text = line
    audio = load_and_resample_audio(audio_path, mel_config.sample_rate, device=device) # shape: [1, time]
    if audio is not None: 
        # get output path
        audio_name, _ = os.path.splitext(os.path.basename(audio_path))
        
        phone = g2p(text)
        if len(phone) > 0:
            mel = mel_extractor(audio.to(device)).cpu().squeeze(0) # shape: [n_mels, time // hop_length]
            output_mel_path = os.path.join(output_mel_dir, f'{idx}_{audio_name}.pt')
            torch.save(mel, output_mel_path)
            
            if data_config.resample:
                audio_path = os.path.join(output_wav_dir, f'{idx}_{audio_name}.wav')
                torchaudio.save(audio_path, audio.cpu(), mel_config.sample_rate)
            return json.dumps({'mel_path': output_mel_path, 'phone': phone, 'audio_path': audio_path, 'text': text}, ensure_ascii=False, allow_nan=False)
            

def main():
    set_start_method('spawn') # CUDA must use spawn method
    input_filelist = load_filelist(input_filelist_path)
    results = []
    
    with Pool(processes=2) as pool:
        for result in tqdm(pool.imap(process_filelist, input_filelist), total=len(input_filelist)):
            if result is not None:
                results.append(f'{result}\n') 
            
    # save filelist
    with open(output_filelist_path, 'w', encoding='utf-8') as f:
        f.writelines(results)
    print(f"filelist file has been saved to {output_filelist_path}")

# faster and use much less CPU
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
    
if __name__ == '__main__':
    main()