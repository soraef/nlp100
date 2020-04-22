
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
    
    def is_noun(self):
        return "名詞" == self.pos[:2]
    
    def is_sahen(self):
        return self.is_noun() and "サ変接続" == self.pos1[:4]
    
    # 「の」かどうか
    def is_no(self):
        return len(self.surface) == 1 and self.surface == "の"

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
    
    def get_connected_nouns(self):
        nouns = []
        noun = ""
        num = 0
        for i in range(len(self.nodes)):
            if self.nodes[i].is_noun():
                noun += self.nodes[i].surface
                num += 1
            else:
                if num > 1:
                    nouns.append(noun)
                noun = ""
                num  = 0

        # 名詞の連接で終わる場合
        if num > 1:
            nouns.append(noun)

        return nouns


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
    nouns += sentence.get_connected_nouns()

nouns = list(set(nouns))

for noun in nouns[0:10]:
    print(noun)

# 三平君
# 大変目
# 美学研究
# 三日三晩
# 表裏二枚合せ
# 月桂寺さん
# 高山彦九郎
# 向う横町
# 二階
# 大分みんな
# ...