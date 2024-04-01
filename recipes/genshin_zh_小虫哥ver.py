import os
import re
from dataclasses import dataclass
import concurrent.futures

from tqdm.auto import tqdm
import openpyxl # use to open excel. run ! pip install openpyxl 

# download_link: https://www.bilibili.com/read/cv23965717
@dataclass
class DataConfig:
    dataset_path = './raw_datasets/Genshin_chinese4.5/原神语音包4.5（中）'
    excel_path = './raw_datasets/Genshin_chinese4.5/原神4.5语音包对应文本（中）.xlsx'
    output_filelist_path = './filelists/genshin_zh.txt'

# 若文本中出现以下字符，基本和语音对不上
FORBIDDEN_TEXTS = ["……", "{NICKNAME}", "#", "(", ")", "♪", "test", "{0}", "█", "*", "█", "+", "Gohus"]
REPLACEMENTS = {"$UNRELEASED": ""}
escaped_forbidden_texts = [re.escape(text) for text in FORBIDDEN_TEXTS]
pattern = re.compile("|".join(escaped_forbidden_texts))

data_config = DataConfig()

def clean_text(text):
    cleaned_text = text
    # 删去所有包含英文的台词
    if re.search(r'[A-Za-z0-9]', cleaned_text):
        return None
    if pattern.search(cleaned_text):
        return None
    for old, new in REPLACEMENTS.items():
        cleaned_text = cleaned_text.replace(old, new)
    return text

def read_excel(excel):
    wb = openpyxl.load_workbook(excel)
    sheet_names = wb.sheetnames
    main_sheet = wb[sheet_names[0]]
    npc_names = [cell.value for cell in main_sheet['B'] if cell.value][1:]
    npc_audio_number = [cell.value for cell in main_sheet['C'] if cell.value][1:]
    return wb, npc_names, npc_audio_number
    
def process_filelist(data):
    audio_path, text, npc_path = data
    input_audio_path = os.path.abspath(os.path.join(npc_path, audio_path))
    if os.path.exists(input_audio_path):
        text = clean_text(text)
        if text is not None:
            return f'{input_audio_path}|{text}\n'

if __name__ == '__main__':   
    wb, npc_names, npc_audio_number = read_excel(data_config.excel_path)
    datas_list = []
    results = []
    
    for index, npc_name in enumerate(tqdm(npc_names)):
        sheet = wb[npc_name]
        audio_names = [cell.value for cell in sheet['C'] if cell.value][1:]
        texts = [cell.value for cell in sheet['D'] if cell.value][1:]
        npc_path = os.path.join(data_config.dataset_path,  npc_name)
        datas_list.extend([(audio_name, text, npc_path) for audio_name, text in zip(audio_names, texts)]) 
           
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(process_filelist, data) for data in datas_list]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(datas_list)):
            result = future.result()
            if result is not None:
                results.append(result)
                                 
    # make sure that the parent dir exists, raising error at the last step is quite terrible OVO
    os.makedirs(os.path.dirname(data_config.output_filelist_path), exist_ok=True)
    with open(data_config.output_filelist_path, 'w', encoding='utf-8') as f:
        f.writelines(results)