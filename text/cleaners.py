import re
import string
import numpy as np
from .langdetect import detect, LangDetectException

from text.english import english_to_ipa2
from text.mandarin import chinese_to_cnm3
from text.japanese import japanese_to_ipa2

language_module_map = {"PAD":0, "ZH": 1, "EN": 2, "JA": 3}

# 预编译正则表达式
ZH_PATTERN = re.compile(r'[\u3400-\u4DBF\u4e00-\u9FFF\uF900-\uFAFF\u3000-\u303F]')
EN_PATTERN = re.compile(r'[a-zA-Z.,!?\'"(){}[\]<>:;@#$%^&*-_+=/\\|~`]+')
JP_PATTERN = re.compile(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF\u31F0-\u31FF\uFF00-\uFFEF\u3000-\u303F]')
CLEANER_PATTERN = re.compile(r'\[(ZH|EN|JA)\]')

def detect_language(text: str, prev_lang=None):
    """
    根据给定的文本检测语言

    :param text: 输入文本
    :param prev_lang: 上一个检测到的语言
    :return: 'ZH' for Chinese, 'EN' for English, 'JA' for Japanese, or prev_lang for spaces
    """
    if ZH_PATTERN.search(text): return 'ZH'
    if EN_PATTERN.search(text): return 'EN'
    if JP_PATTERN.search(text): return 'JA'
    if text.isspace(): return prev_lang  # 若是空格，则返回前一个语言
    return None

def replace_substring(s, start_index, end_index, replacement):
    return s[:start_index] + replacement + s[end_index:]

def replace_sublist(lst, start_index, end_index, replacement_list):
    lst[start_index:end_index] = replacement_list

# convert text to ipa and prepare for language embedding
def append_tags_and_convert(match, conversion_func, tag_value, tags):
    converted_text = conversion_func(match.group(1))
    tags.extend([tag_value] * len(converted_text))
    return converted_text + ' '

# auto detect language using re
def cjke_cleaners4(text: str):
    """
    根据文本内容自动检测语言并转换为IPA音标

    :param text: 输入文本
    :return: 转换为IPA音标的文本
    """
    text = CLEANER_PATTERN.sub('', text)
    pointer = 0
    output = ''
    current_language = detect_language(text[pointer])
    
    while pointer < len(text):
        temp_text = ''
        while pointer < len(text) and detect_language(text[pointer], current_language) == current_language:
            temp_text += text[pointer]
            pointer += 1
        if current_language == 'ZH':
            output += chinese_to_cnm3(temp_text)
        elif current_language == 'JA':
            output += japanese_to_ipa2(temp_text)
        elif current_language == 'EN':
            output += english_to_ipa2(temp_text)
        if pointer < len(text):
            current_language = detect_language(text[pointer])

    output = re.sub(r'\s+$', '', output)
    output = re.sub(r'([^\.,!\?\-…~])$', r'\1.', output)
    return output
