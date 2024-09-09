import os
import concurrent.futures

from tqdm import tqdm
from dataclasses import dataclass

@dataclass
class DataConfig:
    audio_dirs = ['./datasets'] # paths to audios
    filelist_path = './filelists/filelist.txt' # path to save filelist
    audio_formats = ('.wav', '.ogg', '.opus', '.mp3', '.flac')
            
data_config = DataConfig()

filelist_path = data_config.filelist_path

os.makedirs(os.path.dirname(filelist_path), exist_ok=True)
    
def find_audio_files(directory) -> list:
    audio_files = []
    valid_extensions = data_config.audio_formats
    
    for root, dirs, files in tqdm(os.walk(directory)):
        audio_files.extend(os.path.join(root, file) for file in files if file.endswith(valid_extensions))
        
    return audio_files


def main():
    results = []
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(find_audio_files, audio_dir) for audio_dir in data_config.audio_dirs]
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            results.extend(future.result())
            
    # save filelist
    with open(filelist_path, 'w', encoding='utf-8') as f:
        f.writelines(f"{result}\n" for result in results)
        
    print(f"filelist has been saved to {filelist_path}")
    
if __name__ == '__main__':
    main()