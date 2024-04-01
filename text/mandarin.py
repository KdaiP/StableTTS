import re
from pypinyin import lazy_pinyin, Style
from .custom_pypinyin_dict import phrase_pinyin_data
phrase_pinyin_data.load()
import jieba
from .cn2an import an2cn

# 标点符号正则
punc_map = {
    "：": ",",
    "；": ",",
    "，": ",",
    "。": ".",
    "！": "!",
    "？": "?",
    "\n": ".",
    "·": ",",
    "、": ",",
    "$": ".",
    "“": "'",
    "”": "'",
    '"': "'",
    "‘": "'",
    "’": "'",
    "（": "'",
    "）": "'",
    "(": "'",
    ")": "'",
    "《": "'",
    "》": "'",
    "【": "'",
    "】": "'",
    "[": "'",
    "]": "'",
    "—": "-",
    "～": "~",
    "「": "'",
    "」": "'",
    "『": "'", 
    "』": "'",
}
punc_table = str.maketrans(punc_map)

# 数字正则化
number_pattern = re.compile(r'\d+(?:\.?\d+)?')
def replace_number(match):
    return an2cn(match.group())
def normalize_number(text):
    return number_pattern.sub(replace_number, text)

# get symbols of phones
def load_pinyin_symbols(path):
    pinyin_dict={}
    temp = []
    with open(path, "r", encoding='utf-8') as f:
        content = f.readlines()
    for line in content:
        cuts = line.strip().split(',')
        pinyin = cuts[0]
        phones = cuts[1].split(' ')
        pinyin_dict[pinyin] = phones
        temp.extend(phones)
    temp = list(set(temp))
    tone = []
    for phone in temp:
        for i in range(1, 6):
            phone2 = phone + str(i)
            tone.append(phone2)
    print(sorted(tone, key=lambda x: len(x)))
    return pinyin_dict

def load_pinyin_dict(path):
    pinyin_dict = {}
    with open(path, "r", encoding='utf-8') as f:
        for line in f:
            key, value = line.strip().split(',', 1)
            pinyin_dict[key] = value.split()
    return pinyin_dict
pinyin_dict = load_pinyin_dict('text/cnm3/ds_CNM3.txt')

def chinese_to_cnm3(text: str):
    text = text.translate(punc_table)
    text = normalize_number(text)
    words = jieba.lcut(text, cut_all=False)
    phones = []
    for word in words:
        pinyin_list = lazy_pinyin(word, style=Style.TONE3, neutral_tone_with_five=True)
        for pinyin in pinyin_list:
            if pinyin[-1].isdigit():
                tone = pinyin[-1]
                syllable = pinyin[:-1]
                phone = pinyin_dict[syllable]
                phones.extend([ph + tone for ph in phone])
            elif pinyin[-1].isalpha():
                pass
            else:
                phones.extend(pinyin)
    return phones