# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from pypinyin import load_phrases_dict

from text.custom_pypinyin_dict import cc_cedict_0
from text.custom_pypinyin_dict import cc_cedict_1
from text.custom_pypinyin_dict import cc_cedict_2
from text.custom_pypinyin_dict import cc_cedict_3
from text.custom_pypinyin_dict import genshin

phrases_dict = {}
phrases_dict.update(cc_cedict_0.phrases_dict)
phrases_dict.update(cc_cedict_1.phrases_dict)
phrases_dict.update(cc_cedict_2.phrases_dict)
phrases_dict.update(cc_cedict_3.phrases_dict)
phrases_dict.update(genshin.phrases_dict)

def load():
    load_phrases_dict(phrases_dict)
    print("加载自定义词典成功")

if __name__ == '__main__':
    print(phrases_dict)