import os

import json
import torch
from torch.utils.data import Dataset

from text import cleaned_text_to_sequence

def intersperse(lst, item):
  result = [item] * (len(lst) * 2 + 1)
  result[1::2] = lst
  return result
    
class StableDataset(Dataset):
    def __init__(self, filelist_path, hop_length):
        self.filelist_path = filelist_path     
        self.hop_length = hop_length  
        
        self._load_filelist(filelist_path)

    def _load_filelist(self, filelist_path):
        filelist, lengths = [], []
        with open(filelist_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = json.loads(line.strip())
                filelist.append((line['mel_path'], line['phone']))
                lengths.append(os.path.getsize(line['audio_path']) // (2 * self.hop_length))
            
        self.filelist = filelist
        self.lengths = lengths
    
    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        mel_path, phone = self.filelist[idx]
        mel = torch.load(mel_path, map_location='cpu')
        phone = torch.tensor(intersperse(cleaned_text_to_sequence(phone), 0), dtype=torch.long)
        return mel, phone
    
def collate_fn(batch):
    texts = [item[1] for item in batch]
    mels = [item[0] for item in batch]
    
    text_lengths = torch.tensor([text.size(-1) for text in texts], dtype=torch.long)
    mel_lengths = torch.tensor([mel.size(-1) for mel in mels], dtype=torch.long)
    
    # pad to the same length
    texts_padded = torch.nested.to_padded_tensor(torch.nested.nested_tensor(texts), padding=0)
    mels_padded = torch.nested.to_padded_tensor(torch.nested.nested_tensor(mels), padding=0)

    return texts_padded, text_lengths, mels_padded, mel_lengths