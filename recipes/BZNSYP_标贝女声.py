import os

import re
from dataclasses import dataclass
import concurrent.futures

from tqdm.auto import tqdm

# submit the form on: https://www.data-baker.com/data/index/TNtts/
# then you will get the download link
@dataclass
class DataConfig:
    dataset_path = './raw_datasets/BZNSYP/Wave'
    txt_path = './raw_datasets/BZNSYP/ProsodyLabeling/000001-010000.txt'
    output_filelist_path = './filelists/bznsyp.txt'

data_config = DataConfig()
    
def process_filelist(line):
    audio_name, text = line.split('\t')
    text = re.sub('[#\d]+', '', text) # remove '#' and numbers
    input_audio_path = os.path.abspath(os.path.join(data_config.dataset_path, f'{audio_name}.wav'))
    if os.path.exists(input_audio_path):
        return f'{input_audio_path}|{text}\n'

if __name__ == '__main__':
    filelist = []   
    results = []
    
    with open(data_config.txt_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if idx % 2 == 0:
                filelist.append(line.strip()) 
           
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