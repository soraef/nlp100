from nltk.stem import PorterStemmer
import re

file_path = "../data/nlp.txt"

def load_file(path):
    with open(path) as f:
        data = f.read()
    return data

# テキストを分割
def split_text(text):
    return re.findall(r"(.*?[.;:?!])\s\n?(?=[A-Z])", text)

# 記号を除去
def remove_symbol(text):
    return re.sub(r"[.;;?!()'\",]", "", text)

text = load_file(file_path)
sentences = split_text(text)
sentences = list(map(remove_symbol, sentences))
ps = PorterStemmer()

for sentence in sentences[0:2]:
    for word in sentence.split(" "):
        print(f"{word}\t{ps.stem(word)}")
    print("")

# 
# 出力
# 
# Natural natur
# language        languag
# processing      process
# NLP     nlp
# is      is
# a       a
# field   field
# of      of
# computer        comput
# science scienc
# artificial      artifici
# intelligence    intellig
# and     and
# linguistics     linguist
# concerned       concern
# with    with
# the     the
# interactions    interact
# between between
# computers       comput
# and     and
# human   human
# natural natur
# languages       languag

# As      As
# such    such
# NLP     nlp
# is      is
# related relat
# to      to
# the     the
# area    area
# of      of
# humani-computer humani-comput
# interaction     interact



