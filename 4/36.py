
import pprint
import collections

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

for word, count in counter.most_common(10):
    print(f"{word}: {count}")

# の: 9194
# 。: 7486
# て: 6868
# 、: 6772
# は: 6420
# に: 6243
# を: 6071
# と: 5508
# が: 5337
# た: 3988

# 以下遅いやり方
# [(word, count), (word, count), ... ]
# def find_word(counts, word):
#     for i, count in enumerate(counts):
#         if count[0] == word:
#             return i
#     return None

# appearance_count = []
# for sentence in sentences:
#     for node in sentence.nodes:
#         word = node.surface
#         index = find_word(appearance_count, word)
#         if index is None:
#             appearance_count.append([word, 1])
#         else:
#             appearance_count[index][1] += 1

# sorted_appearance_count = sorted(appearance_count, key=lambda x:x[1], reverse=True)
# print(sorted_appearance_count[:100])

