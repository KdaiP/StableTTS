import os
import io
from pathlib import Path
from dataclasses import dataclass
import concurrent.futures

from tqdm.auto import tqdm
import pandas as pd
import torchaudio

# download_link: https://huggingface.co/datasets/CSTR-Edinburgh/vctk/tree/063f48e28abda80b2fdc4d4433af8a99e71bfe16
# other huggingface TTS parquet datasets could use the same script
@dataclass
class DataConfig:
    dataset_path = './raw_datasets/VCTK'
    output_filelist_path = './filelists/VCTK.txt'
    output_audio_path = './raw_datasets/VCTK_audios' # to extract audios from parquet files

data_config = DataConfig()
    
def process_parquet(parquet_path: Path):
    df = pd.read_parquet(parquet_path)
    filelist = []
    for idx, data in tqdm(df.iterrows(), total=len(df)):
        audio = io.BytesIO(data['audio']['bytes'])
        audio, sample_rate = torchaudio.load(audio)
        text = data['text']
        
        path = os.path.abspath(os.path.join(data_config.output_audio_path, data['audio']['path']))
        torchaudio.save(path, audio, sample_rate)

        filelist.append(f'{path}|{text}\n')
        
    return filelist

if __name__ == '__main__':
    filelist = []   
    results = []
    
    dataset_path = Path(data_config.dataset_path)
    parquets = list(dataset_path.rglob('*.parquet'))
           
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_parquet, parquet_path) for parquet_path in parquets]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(parquets)):
            result = future.result()
            if result is not None:
                results.extend(result)
                                 
    # make sure that the parent dir exists, raising error at the last step is quite terrible OVO
    os.makedirs(os.path.dirname(data_config.output_filelist_path), exist_ok=True)
    with open(data_config.output_filelist_path, 'w', encoding='utf-8') as f:
        f.writelines(results)
