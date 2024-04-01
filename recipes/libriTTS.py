import os
from pathlib import Path
from dataclasses import dataclass
import concurrent.futures

from tqdm.auto import tqdm

# download_link: https://openslr.org/60/
@dataclass
class DataConfig:
    dataset_path = './raw_datasets/LibriTTS/train-other-500'
    output_filelist_path = './filelists/libri_tts.txt'

data_config = DataConfig()
    
def process_filelist(wav_path: Path):
    text_path = wav_path.with_suffix('.normalized.txt')
    if text_path.exists():
        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        return f'{wav_path.as_posix()}|{text}\n'

if __name__ == '__main__':
    filelist = []   
    results = []
    
    dataset_path = Path(data_config.dataset_path)
    waves = list(dataset_path.rglob('*.wav'))
           
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_filelist, wav_path) for wav_path in waves]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(waves)):
            result = future.result()
            if result is not None:
                results.append(result)
                                 
    # make sure that the parent dir exists, raising error at the last step is quite terrible OVO
    os.makedirs(os.path.dirname(data_config.output_filelist_path), exist_ok=True)
    with open(data_config.output_filelist_path, 'w', encoding='utf-8') as f:
        f.writelines(results)