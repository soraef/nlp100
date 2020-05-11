import xml.etree.ElementTree as ET
import re

tree = ET.parse("../data/nlp.txt.xml")
root = tree.getroot()

# (NP ... )という形になっている文字列を見つける
def find_np(s):
    for np_s in re.finditer(r"\(NP", s):
        # カッコの数
        # (NP の次から走査するので1で初期化
        brackets_count = 1
        ptr   = np_s.start() + 3

        # ptrを'(NP'と対応がとれる')'の位置まで進める
        while brackets_count:
            if s[ptr] == "(":
                brackets_count += 1
            elif s[ptr] == ")":
                brackets_count -= 1
            ptr += 1

        yield s[np_s.start():ptr] 

# (HOGR hoge)のような形を探してhogeを返す
def find_word(s):
    return re.findall(r"\([^()]*\s([^()]*)\)", s)

for parse in root.iter("parse"):
    for np_s in find_np(parse.text):
        print(" ".join(find_word(np_s)))

#
# 出力
# 
# Natural language processing
# Wikipedia
# the free encyclopedia Natural language processing -LRB- NLP -RRB-
# the free encyclopedia Natural language processing
# NLP
# a field of computer science , artificial intelligence , and linguistics concerned with the interactions between computers and human -LRB- natural -RRB- languages 
# a field of computer science
# a field
# ...
#
