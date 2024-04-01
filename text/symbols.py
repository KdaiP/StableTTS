'''
Defines the set of symbols used in text input to the model.
'''

# japanese_cleaners
# _pad        = '_'
# _punctuation = ',.!?-'
# _letters = 'AEINOQUabdefghijkmnoprstuvwyzʃʧ↓↑ '


'''# japanese_cleaners2
_pad        = '_'
_punctuation = ',.!?-~…'
_letters = 'AEINOQUabdefghijkmnoprstuvwyzʃʧʦ↓↑ '
'''


'''# korean_cleaners
_pad        = '_'
_punctuation = ',.!?…~'
_letters = 'ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎㄲㄸㅃㅆㅉㅏㅓㅗㅜㅡㅣㅐㅔ '
'''

'''# chinese_cleaners
_pad        = '_'
_punctuation = '，。！？—…'
_letters = 'ㄅㄆㄇㄈㄉㄊㄋㄌㄍㄎㄏㄐㄑㄒㄓㄔㄕㄖㄗㄘㄙㄚㄛㄜㄝㄞㄟㄠㄡㄢㄣㄤㄥㄦㄧㄨㄩˉˊˇˋ˙ '
'''

# # zh_ja_mixture_cleaners
# _pad        = '_'
# _punctuation = ',.!?-~…'
# _letters = 'AEINOQUabdefghijklmnoprstuvwyzʃʧʦɯɹəɥ⁼ʰ`→↓↑ '


'''# sanskrit_cleaners
_pad        = '_'
_punctuation = '।'
_letters = 'ँंःअआइईउऊऋएऐओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलळवशषसहऽािीुूृॄेैोौ्ॠॢ '
'''

'''# cjks_cleaners
_pad        = '_'
_punctuation = ',.!?-~…'
_letters = 'NQabdefghijklmnopstuvwxyzʃʧʥʦɯɹəɥçɸɾβŋɦː⁼ʰ`^#*=→↓↑ '
'''

'''# thai_cleaners
_pad        = '_'
_punctuation = '.!? '
_letters = 'กขฃคฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรฤลวศษสหฬอฮฯะัาำิีึืุูเแโใไๅๆ็่้๊๋์'
'''

# # cjke_cleaners2
_pad        = '_'
_punctuation = ',.!?-~…' + "'"
_IPA_letters = 'NQabdefghijklmnopstuvwxyzɑæʃʑçɯɪɔɛɹðəɫɥɸʊɾʒθβŋɦ⁼ʰ`^#*=ˈˌ→↓↑ '
_CNM3_letters = ['y1', 'y2', 'y3', 'y4', 'y5', 'n1', 'n2', 'n3', 'n4', 'n5', 'p1', 'p2', 'p3', 'p4', 'p5', 'x1', 'x2', 'x3', 'x4', 'x5', 'k1', 'k2', 'k3', 'k4', 'k5', 'l1', 'l2', 'l3', 'l4', 'l5', 'q1', 'q2', 'q3', 'q4', 'q5', 'w1', 'w2', 'w3', 'w4', 'w5', 'E1', 'E2', 'E3', 'E4', 'E5', 'b1', 'b2', 'b3', 'b4', 'b5', 'c1', 'c2', 'c3', 'c4', 'c5', 'z1', 'z2', 'z3', 'z4', 'z5', 'e1', 'e2', 'e3', 'e4', 'e5', 'f1', 'f2', 'f3', 'f4', 'f5', 's1', 's2', 's3', 's4', 's5', 'j1', 'j2', 'j3', 'j4', 'j5', 'o1', 'o2', 'o3', 'o4', 'o5', 'i1', 'i2', 'i3', 'i4', 'i5', 'd1', 'd2', 'd3', 'd4', 'd5', 'm1', 'm2', 'm3', 'm4', 'm5', 't1', 't2', 't3', 't4', 't5', 'h1', 'h2', 'h3', 'h4', 'h5', 'g1', 'g2', 'g3', 'g4', 'g5', 'v1', 'v2', 'v3', 'v4', 'v5', 'r1', 'r2', 'r3', 'r4', 'r5', 'a1', 'a2', 'a3', 'a4', 'a5', 'u1', 'u2', 'u3', 'u4', 'u5', 'I01', 'I02', 'I03', 'I04', 'I05', 'i01', 'i02', 'i03', 'i04', 'i05', 'uo1', 'uo2', 'uo3', 'uo4', 'uo5', 'o01', 'o02', 'o03', 'o04', 'o05', 'U01', 'U02', 'U03', 'U04', 'U05', 'v01', 'v02', 'v03', 'v04', 'v05', 'er1', 'er2', 'er3', 'er4', 'er5', 'A01', 'A02', 'A03', 'A04', 'A05', 'ai1', 'ai2', 'ai3', 'ai4', 'ai5', 'e01', 'e02', 'e03', 'e04', 'e05', 'sh1', 'sh2', 'sh3', 'sh4', 'sh5', 'an1', 'an2', 'an3', 'an4', 'an5', 'ou1', 'ou2', 'ou3', 'ou4', 'ou5', 'ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'a01', 'a02', 'a03', 'a04', 'a05', 'N01', 'N02', 'N03', 'N04', 'N05', 'ao1', 'ao2', 'ao3', 'ao4', 'ao5', 've1', 've2', 've3', 've4', 've5', 'ir1', 'ir2', 'ir3', 'ir4', 'ir5', 'ng1', 'ng2', 'ng3', 'ng4', 'ng5', 'ua1', 'ua2', 'ua3', 'ua4', 'ua5', 'zh1', 'zh2', 'zh3', 'zh4', 'zh5', 'O01', 'O02', 'O03', 'O04', 'O05', 'ie1', 'ie2', 'ie3', 'ie4', 'ie5', 'E01', 'E02', 'E03', 'E04', 'E05', 'ia1', 'ia2', 'ia3', 'ia4', 'ia5', 'iE01', 'iE02', 'iE03', 'iE04', 'iE05', 'ang1', 'ang2', 'ang3', 'ang4', 'ang5', 'ng01', 'ng02', 'ng03', 'ng04', 'ng05', 'io01', 'io02', 'io03', 'io04', 'io05', 'iA01', 'iA02', 'iA03', 'iA04', 'iA05', 'uA01', 'uA02', 'uA03', 'uA04', 'uA05', 'ong1', 'ong2', 'ong3', 'ong4', 'ong5', 'oo01', 'oo02', 'oo03', 'oo04', 'oo05', 'uE01', 'uE02', 'uE03', 'uE04', 'uE05', 'vE01', 'vE02', 'vE03', 'vE04', 'vE05', 'ue01', 'ue02', 'ue03', 'ue04', 'ue05', 'ua01', 'ua02', 'ua03', 'ua04', 'ua05', 'iO01', 'iO02', 'iO03', 'iO04', 'iO05']
_additional = ['<sil>', '<asp>']
# _CNM3_letters = []


'''# shanghainese_cleaners
_pad        = '_'
_punctuation = ',.!?…'
_letters = 'abdfghiklmnopstuvyzøŋȵɑɔɕəɤɦɪɿʑʔʰ̩̃ᴀᴇ15678 '
'''

'''# chinese_dialect_cleaners
_pad        = '_'
_punctuation = ',.!?~…─'
_letters = '#Nabdefghijklmnoprstuvwxyzæçøŋœȵɐɑɒɓɔɕɗɘəɚɛɜɣɤɦɪɭɯɵɷɸɻɾɿʂʅʊʋʌʏʑʔʦʮʰʷˀː˥˦˧˨˩̥̩̃̚ᴀᴇ↑↓∅ⱼ '
'''

# Export all symbols:
symbols = [_pad] + list(_punctuation) + list(_IPA_letters) + _CNM3_letters + _additional

# Special symbol ids
SPACE_ID = symbols.index(" ")
