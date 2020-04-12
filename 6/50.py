import re

file_path = "../data/nlp.txt"

def load_file(path):
    with open(path) as f:
        data = f.read()
    return data

# テキストを分割
def split_text(text):
    return re.findall(r"(.*?[.;:?!])\s\n?(?=[A-Z])", text)

text = load_file(file_path)
sentences = split_text(text)

for sentence in sentences:
    print(sentence)

