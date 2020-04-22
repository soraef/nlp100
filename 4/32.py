
import pprint

mecab_file = "../data/neko.txt.mecab"

class Node:
    def __init__(self, surface, base, pos, pos1):
        self.surface = surface
        self.base = base
        self.pos = pos
        self.pos1 = pos1

    def is_verb(self):
        return "動詞" == self.pos[:2]

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
    
    def get_base_verbs(self):
        return [node.base for node in self.nodes if node.is_verb()]


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

base_verbs = []
for sentence in sentences:
    base_verbs += sentence.get_base_verbs()

base_verbs = list(set(base_verbs))

for verb in base_verbs:
    print(verb)

# 売り捌く
# つぐ
# 融ける
# 抑える
# 論ずる
# 忍ぶ
# 極まる
# 告げる
# かく
# 仰向く
# ...