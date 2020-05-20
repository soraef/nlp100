import csv
import re
import pickle

def read_data(filename):
    with open(filename, encoding="utf-8") as f:
        data = list(csv.reader(f, delimiter="\t"))
    return data

train_data = read_data("../data/train.txt")

word2count = {}

# 単語の出現回数をカウント
for data in train_data:
    words = re.sub(r'[.,:;!?"]', "", data[1]).split()
    words = map(lambda word: word.lower(), words)
    for word in words:
        # wordがない場合に初期化
        word2count.setdefault(word, 0)
        word2count[word] += 1

word2id = {}
new_id = 1
for k, v in sorted(word2count.items(), key=lambda x: -x[1]):
    if v <= 2:
        word2id[k] = 0
    else:
        word2id[k] = new_id
        new_id += 1

with open("../data/word2id-80.pickle", "wb") as f:
    pickle.dump(word2id, f)


def get_id(word):
    return word2id.get(word, 0)

word = input()
print(get_id(word))

