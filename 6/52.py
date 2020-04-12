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

for sentence in sentences:
    for word in sentence.split(" "):
        print(f"{word}\t{ps.stem(word)}")
    print("")



