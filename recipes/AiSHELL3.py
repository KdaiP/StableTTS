import os

import re
from dataclasses import dataclass
import concurrent.futures

from tqdm.auto import tqdm

# download_link: https://www.openslr.org/93/
@dataclass
class DataConfig:
    dataset_path = './raw_datasets/Aishell3/train/wav'
    txt_path = './raw_datasets/Aishell3/train/content.txt'
    output_filelist_path = './filelists/aishell3.txt'

data_config = DataConfig()
    
def process_filelist(line):
    dir_name, audio_path, text = line
    input_audio_path = os.path.abspath(os.path.join(data_config.dataset_path, dir_name, audio_path))
    if os.path.exists(input_audio_path):
        return f'{input_audio_path}|{text}\n'

if __name__ == '__main__':
    filelist = []   
    results = []
    
    with open(data_config.txt_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            audio_path, text = line.strip().split(maxsplit=1)
            dir_name = audio_path[:7]
            text = re.sub(r'[a-zA-Z0-9\s]', '', text) # remove pinyin and tone
            filelist.append((dir_name, audio_path, text))
           
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(process_filelist, line) for line in filelist]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(filelist)):
            result = future.result()
            if result is not None:
                results.append(result)
                                 
    # make sure that the parent dir exists, raising error at the last step is quite terrible OVO
    os.makedirs(os.path.dirname(data_config.output_filelist_path), exist_ok=True)
    with open(data_config.output_filelist_path, 'w', encoding='utf-8') as f:
        f.writelines(results)