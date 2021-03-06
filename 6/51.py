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

for sentence in sentences[0:2]:
    for word in sentence.split(" "):
        print(word)
    print("")

# 　
# 出力
# 
# Natural
# language
# processing
# NLP
# is
# a
# field
# of
# computer
# science
# artificial
# intelligence
# and
# linguistics
# concerned
# with
# the
# interactions
# between
# computers
# and
# human
# natural
# languages

# As
# such
# NLP
# is
# related
# to
# the
# area
# of
# humani-computer
# interaction