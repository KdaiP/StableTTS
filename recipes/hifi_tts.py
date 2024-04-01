import os
import json
from pathlib import Path
from dataclasses import dataclass
import concurrent.futures

from tqdm.auto import tqdm

# download_link: https://www.openslr.org/109/
@dataclass
class DataConfig:
    dataset_path = './raw_datasets/hi_fi_tts_v0'
    output_filelist_path = './filelists/hifi_tts.txt'

data_config = DataConfig()
    
def process_filelist(speaker):
    filelist = []
    with open(speaker, 'r', encoding='utf-8') as f:
        for line in f:
            line = json.loads(line.strip())
            audio_path = os.path.abspath(os.path.join(data_config.dataset_path, line['audio_filepath']))
            text = line['text_normalized']
            if os.path.exists(audio_path):
                filelist.append(f'{audio_path}|{text}\n')
    return filelist

if __name__ == '__main__':
    filelist = []   
    results = []
    
    dataset_path = Path(data_config.dataset_path)
    speakers = list(dataset_path.rglob('*.json'))
           
    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
        futures = [executor.submit(process_filelist, speaker) for speaker in speakers]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(speakers)):
            result = future.result()
            if result is not None:
                results.extend(result)
                                 
    # make sure that the parent dir exists, raising error at the last step is quite terrible OVO
    os.makedirs(os.path.dirname(data_config.output_filelist_path), exist_ok=True)
    with open(data_config.output_filelist_path, 'w', encoding='utf-8') as f:
        f.writelines(results)