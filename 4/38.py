import pprint
import collections
import matplotlib.pyplot as plt
import japanize_matplotlib

mecab_file = "../data/neko.txt.mecab"


class Node:
    def __init__(self, surface, base, pos, pos1):
        self.surface = surface
        self.base = base
        self.pos = pos
        self.pos1 = pos1

    @classmethod
    def from_txt(cls, line):
        surface = line.split("\t")[0]

        if surface[:3] == "EOS":
            return None
    
        pos     = line.split("\t")[1].split(",")[0]
        pos1    = line.split("\t")[1].split(",")[1]
        base    = line.split("\t")[1].split(",")[6]

        return cls(surface, base, pos, pos1)
    

class Sentence:
    def __init__(self):
        self.nodes = []
    
    def add_node(self, node):
        self.nodes.append(node)


def load_mecab_file(from_path):
    sentences = []
    with open(from_path, encoding="utf-8") as f:
        line = f.readline()
        sentence = Sentence()
        while line:
            node = Node.from_txt(line)
            if node is not None:
                sentence.add_node(node)
            else:
                if sentence.nodes:
                    sentences.append(sentence)
                sentence = Sentence()
            line = f.readline()

    return sentences

sentences = load_mecab_file(mecab_file)
words = [node.surface for sentence in sentences for node in sentence.nodes]

counter = collections.Counter(words)
data = counter.most_common()

count_max = data[0][1]
histogram = [ 0 for _ in range(count_max + 1)]

for d in data:
    histogram[d[1]] += 1

x_lim = 100
x = [i for i in range(count_max + 1)][:x_lim]
y = histogram[:x_lim]

plt.bar(x, y)
plt.xlabel("出現頻度")
plt.ylabel("出現頻度をとる単語の数")
plt.xlim(0.2, x_lim)
plt.savefig("38_result.png")