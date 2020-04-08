import pprint

mecab_file = "../data/neko.txt.mecab"

class Node:
    def __init__(self, surface, base, pos, pos1):
        self.dict = {
            "surface": surface,
            "base"   : base,
            "pos"    : pos,
            "pos1"   : pos1,
            }

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
        return list(map(lambda node: node.dict, self.nodes))


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

sentences = [sentence.get_nodes_dict() for sentence in sentences]
pprint.pprint(sentences[0:5])

        
            

    
