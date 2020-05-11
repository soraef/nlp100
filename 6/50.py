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

#
# 出力(一部)
#
# Natural language processing (NLP) is a field of computer science, artificial intelligence, and linguistics concerned with the interactions between computers and human (natural) languages.
# As such, NLP is related to the area of humani-computer interaction.
# Many challenges in NLP involve natural language understanding, that is, enabling computers to derive meaning from human or natural language input, and others involve natural language generation.
# The history of NLP generally starts in the 1950s, although work can be found from earlier periods.
# In 1950, Alan Turing published an article titled "Computing Machinery and Intelligence" which proposed what is now called the Turing test as a criterion of intelligence.