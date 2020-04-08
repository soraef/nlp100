
import pprint

mecab_file = "../data/neko.txt.mecab"

class Node:
    def __init__(self, surface, base, pos, pos1):
        self.surface = surface
        self.base = base
        self.pos = pos
        self.pos1 = pos1

    def get_dict(self):
        return {
            "surface": self.surface,
            "base"   : self.base,
            "pos"    : self.pos,
            "pos1"   : self.pos1,
        }

    def is_verb(self):
        return "動詞" == self.pos[:2]
    
    def is_noun(self):
        return "名詞" == self.pos[:2]
    
    def is_sahen(self):
        return self.is_noun() and "サ変接続" == self.pos1[:4]

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
    
    def get_nodes_dict(self):
        return list(map(lambda node: node.get_dict(), self.nodes))

    def get_verbs(self):
        return [node.surface for node in self.nodes if node.is_verb()]
    
    def get_base_verbs(self):
        return [node.base for node in self.nodes if node.is_verb()]
    
    def get_sahen_nouns(self):
        return [node.surface for node in self.nodes if node.is_sahen()]


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

nouns = []

for sentence in sentences:
    nouns += sentence.get_sahen_nouns()

nouns = list(set(nouns))

pprint.pprint(nouns[0:100])