import pydot
import graphviz

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
    
    def is_symbol(self):
        return self.pos == "記号"

    def is_verb(self):
        return self.pos == "動詞"

    def is_noun(self):
        return self.pos == "名詞"
        
    
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

    def get_chunk_surface(self, ignore_symbol=False):
        chunk_surface = ""

        for morph in self.morphs:
            if not(ignore_symbol and morph.is_symbol()):
                chunk_surface += morph.surface

        return chunk_surface

    def is_contain_verb(self):
        for morph in self.morphs:
            if morph.is_verb():
                return True
        return False

    def is_contain_noun(self):
        for morph in self.morphs:
            if morph.is_noun():
                return True
        return False

    def __str__(self):
        return pprint.pformat({
            "chunk_surface": self.get_chunk_surface(),
            "dst": self.dst
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
    
    def chunks_dot(self):
        return [(chunk.get_chunk_surface(True), self.chunks[chunk.dst].get_chunk_surface(True)) 
                    for chunk in self.chunks 
                        if not chunk.dst == -1 and chunk.get_chunk_surface(True) and self.chunks[chunk.dst].get_chunk_surface(True)]

    
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
                    sentence = Sentence.from_cabocha_txt(lines)
                    sentences.append(sentence)
                lines = []
            line = f.readline()

    return sentences

sentences = load_mecab_file(mecab_file)

chanks = []
chunk_dots = []

for sentence in sentences[7:8]:
    chunk_dots.extend(sentence.chunks_dot())


dot_filename = "44"
datas_dir    = "./"

dot = graphviz.Digraph(
            comment='かかり受け木',
            filename=dot_filename, # DOT言語ファイルのファイル名 (これがグラフ画像のファイル名にも使われる)
            directory=datas_dir, # DOT言語ファイルと画像を保存するフォルダ
            format='png', # グラフの保存形式
            engine='dot',
            )

fontname = 'MS Gothic'

dot.attr('graph', fontname=fontname)
dot.attr('node', fontname=fontname, shape='box', color='blue', style='rounded')
dot.attr('edge', fontname=fontname, penwidth='1.5', color='gray')

for chunk_dot in chunk_dots:
    print(chunk_dot)
    dot.edge(chunk_dot[0], chunk_dot[1])

dot.render()

