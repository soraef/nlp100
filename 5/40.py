import re
import pprint

mecab_file = "../data/neko.txt.cabocha"

class Morph:
    def __init__(self, surface, base, pos, pos1):
        self.surface = surface
        self.base    = base
        self.pos     = pos
        self.pos1    = pos1

    def __str__(self):
        return pprint.pformat({
            "surface": self.surface,
            "base": self.base, 
            "pos": self.pos, 
            "pos1": self.pos1
        })
        
    
    @classmethod
    def from_txt(cls, line):
        surface = line.split("\t")[0]

        if surface[:3] == "EOS" or re.match(r"\* \d+? [-]?\d+?D \d+?/\d+? [-]?\d+(?:\.\d+)?", line):
            return None
    
        pos     = line.split("\t")[1].split(",")[0]
        pos1    = line.split("\t")[1].split(",")[1]
        base    = line.split("\t")[1].split(",")[6]

        return cls(surface, base, pos, pos1)
    

class Sentence:
    def __init__(self):
        self.morphs = []
    
    def add_morph(self, morph):
        self.morphs.append(morph)
    
    def print_morphs(self):
        for morph in self.morphs:
            print(morph)
    
    # linesはEOSまでのtxt.cabocha各行をリストに格納した形(EOSは含まない)
    @classmethod
    def from_cabocha_txt(cls, lines):
        sentence = cls()
        for line in lines:
            if re.match(r"\* \d+? [-]?\d+?D \d+?/\d+? [-]?\d+(?:\.\d+)?", line):
                continue
            morph = Morph.from_txt(line)
            sentence.add_morph(morph)
        return sentence



def load_mecab_file(from_path):
    sentences = []
    with open(from_path, encoding="utf-8") as f:
        line = f.readline()
        lines = []
        while line:
            if line[:3] != "EOS":
                lines.append(line)
            else:
                if lines:
                    sentences.append(Sentence.from_cabocha_txt(lines))
                lines = []
            line = f.readline()

    return sentences

sentences = load_mecab_file(mecab_file)
sentences[2].print_morphs()


# 出力
# 
# {'base': '名前', 'pos': '名詞', 'pos1': '一般', 'surface': '名前'}
# {'base': 'は', 'pos': '助詞', 'pos1': '係助詞', 'surface': 'は'}
# {'base': 'まだ', 'pos': '副詞', 'pos1': '助詞類接続', 'surface': 'まだ'}
# {'base': '無い', 'pos': '形容詞', 'pos1': '自立', 'surface': '無い'}
# {'base': '。', 'pos': '記号', 'pos1': '句点', 'surface': '。'}