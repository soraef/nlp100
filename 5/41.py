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

class Chunk:
    def __init__(self, dst):
        self.morphs = []
        self.dst = dst
        self.srcs = []

    def add_morph(self, morph):
        self.morphs.append(morph)
    
    def add_srcs(self, src):
        self.srcs.append(src)

    def __str__(self):
        chunk_surface = ""
        for morph in self.morphs:
            chunk_surface += morph.surface
        return pprint.pformat({
            "chunk_surface": chunk_surface,
            "dst": self.dst,
            # "srcs": self.srcs
        })
    
    # line
    # * 2 -1D 0/2 0.000000
    @classmethod
    def from_line(cls, line):
        line_items = line.split(" ")
        dst = int(line_items[2].replace("D", ""))
        return cls(dst)

class Sentence:
    def __init__(self):
        self.chunks = []
    
    def add_chunk(self, chunk):
        self.chunks.append(chunk)
    
    def print_chunks(self):
        for chunk in self.chunks:
            print(chunk)
    
    # linesはEOSまでのtxt.cabocha各行をリストに格納した形(EOSは含まない)
    @classmethod
    def from_cabocha_txt(cls, lines):
        sentence = cls()

        for line in lines:
            if re.match(r"\* \d+? [-]?\d+?D \d+?/\d+? [-]?\d+(?:\.\d+)?", line):
                sentence.add_chunk(Chunk.from_line(line))
                continue
            morph = Morph.from_txt(line)
            sentence.chunks[-1].add_morph(morph)

        # chunksのかかり元を設定する
        for i in reversed(range(len(sentence.chunks))):
            chunk_num = i # iは文節番号を表す
            dst = sentence.chunks[chunk_num].dst
            if dst == -1:
                continue
            sentence.chunks[dst].add_srcs(chunk_num)


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
sentences[7].print_chunks()

# 
# 出力
# 
# {'chunk_surface': 'この', 'dst': 1}
# {'chunk_surface': '書生というのは', 'dst': 7}
# {'chunk_surface': '時々', 'dst': 4}
# {'chunk_surface': '我々を', 'dst': 4}
# {'chunk_surface': '捕えて', 'dst': 5}
# {'chunk_surface': '煮て', 'dst': 6}
# {'chunk_surface': '食うという', 'dst': 7}
# {'chunk_surface': '話である。', 'dst': -1}