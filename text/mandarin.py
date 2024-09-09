import re
from typing import Dict, List
from pypinyin import lazy_pinyin, Style
from .custom_pypinyin_dict import phrase_pinyin_data
import jieba
from .cn2an import an2cn

# 加载自定义拼音词典数据
phrase_pinyin_data.load()

# 标点符号正则
PUNC_MAP: Dict[str, str] = {
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
    "/": ",",
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

# from GPT_SoVITS.text.zh_normalization.text_normlization
PUNC_MAP.update ({
    '/': '每',
    '①': '一',
    '②': '二',
    '③': '三',
    '④': '四',
    '⑤': '五',
    '⑥': '六',
    '⑦': '七',
    '⑧': '八',
    '⑨': '九',
    '⑩': '十',
    'α': '阿尔法',
    'β': '贝塔',
    'γ': '伽玛',
    'Γ': '伽玛',
    'δ': '德尔塔',
    'Δ': '德尔塔',
    'ε': '艾普西龙',
    'ζ': '捷塔',
    'η': '依塔',
    'θ': '西塔',
    'Θ': '西塔',
    'ι': '艾欧塔',
    'κ': '喀帕',
    'λ': '拉姆达',
    'Λ': '拉姆达',
    'μ': '缪',
    'ν': '拗',
    'ξ': '克西',
    'Ξ': '克西',
    'ο': '欧米克伦',
    'π': '派',
    'Π': '派',
    'ρ': '肉',
    'ς': '西格玛',
    'σ': '西格玛',
    'Σ': '西格玛',
    'τ': '套',
    'υ': '宇普西龙',
    'φ': '服艾',
    'Φ': '服艾',
    'χ': '器',
    'ψ': '普赛',
    'Ψ': '普赛',
    'ω': '欧米伽',
    'Ω': '欧米伽',
    '+': '加',
    '-': '减',
    '×': '乘',
    '÷': '除',
    '=': '等',
    
    "嗯": "恩",
    "呣": "母"
})

PUNC_TABLE = str.maketrans(PUNC_MAP)

# 数字正则化
NUMBER_PATTERN: re.Pattern = re.compile(r'\d+(?:\.?\d+)?')

# 阿拉伯数字转汉字
def replace_number(match: re.Match) -> str:
    return an2cn(match.group())

def normalize_number(text: str) -> str:
    return NUMBER_PATTERN.sub(replace_number, text)

# get symbols of phones, not used
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

def load_pinyin_dict(path: str) -> Dict[str, List[str]]:
    pinyin_dict = {}
    with open(path, "r", encoding='utf-8') as f:
        for line in f:
            key, value = line.strip().split(',', 1)
            pinyin_dict[key] = value.split()
    return pinyin_dict

import os
pinyin_dict = load_pinyin_dict(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cnm3', 'ds_CNM3.txt'))
# pinyin_dict = load_pinyin_dict('text/cnm3/ds_CNM3.txt')

def chinese_to_cnm3(text: str) -> List[str]:
    # 标点符号和数字正则化
    text = text.translate(PUNC_TABLE)
    text = normalize_number(text)
    # 过滤掉特殊字符
    text = re.sub(r'[#&@“”^_|\\]', '', text)
    
    words = jieba.lcut(text, cut_all=False)
    
    phones = []
    for word in words:
        pinyin_list: List[str] = lazy_pinyin(word, style=Style.TONE3, neutral_tone_with_five=True)
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