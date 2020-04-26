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

    def is_particle(self):
        return self.pos == "助詞"

    def is_sahen_noun(self):
        return self.pos == "名詞" and self.pos1 == "サ変接続"

    def is_surface_wo(self):
        return self.surface == "を"

    def is_match_all(self, pos="", pos1="", surface="", base=""):
        # すべての引数が空ならFalseを返す
        if not pos and not pos1 and not surface and not base:
            return False
        # 引数に値が入っていたら一致しているか確かめる
        # 引数に何も入っていなかったらTrueにする
        is_pos     = self.pos == pos if pos else True
        is_pos1    = self.pos1 == pos1 if pos1 else True
        is_surface = self.surface == surface if surface else True
        is_base    = self.base == base if base else True

        return is_pos and is_pos1 and is_surface and is_base
    
    
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

    def get_chunk_surface(self, ignore_symbol=True):
        chunk_surface = ""

        for morph in self.morphs:
            if not(ignore_symbol and morph.is_symbol()):
                chunk_surface += morph.surface

        return chunk_surface

    # 名詞句を置換したsurfaceを取得する
    def get_surface_sub_noun(self, sub_str):
        surface = ""
        for morph in self.morphs:
            if morph.is_noun():
                surface += sub_str
            elif not morph.is_symbol():
                surface += morph.surface
        # XXのような重複を除く
        return re.sub(r"%s+" % sub_str, r"%s" % sub_str, surface)



    def is_contain(self, pos="", pos1="", surface="", base=""):
        for morph in self.morphs:
            if morph.is_match_all(pos=pos, pos1=pos1, surface=surface, base=base):
                return True
        return False

    def is_contain_verb(self):
        return self.is_contain(pos="動詞")

    def is_contain_noun(self):
        return self.is_contain(pos="名詞")

    def is_contain_particle(self):
        return self.is_contain(pos="助詞")

    # サ変接続名詞+をで構成されるchunkかどうか
    def is_sahen_wo(self):
        if len(self.morphs) < 2:
            return False
        is_wo    = self.morphs[-1].is_match_all(surface="を")
        is_sahen = self.morphs[-2].is_match_all(pos="名詞", pos1="サ変接続")
        return  is_wo and is_sahen

    def get_sahen_wo(self):
        if self.is_sahen_wo():
            return self.morphs[-2].surface + self.morphs[-1].surface
        return ""

    def get_first_verb(self):
        for morph in self.morphs:
            if morph.is_verb():
                return morph.base

    def get_last_particles(self):
        for morph in reversed(self.morphs):
            if morph.is_particle():
                return morph.base

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

    def print_paths_between_nouns(self):
        dst_between_nouns = [] # (i, j)を要素に持つリスト
        len_chunks = len(self.chunks)
        for i in range(len_chunks):
            if not self.chunks[i].is_contain_noun():
                continue
            for j in range(i + 1, len_chunks):
                if not self.chunks[j].is_contain_noun():
                    continue
                dst_between_nouns.append((i, j))
               
        for i, j in dst_between_nouns:
            print(self.get_chunk_path(i, j))
        
        
    def get_chunk_path(self, i, j):
        path = []
        path_num = [] #パスの文節番号を保存しておくリスト
        start_chunk = self.chunks[i]
        path.append(start_chunk.get_surface_sub_noun("X"))
        path_num.append(i)

        next_chunk_num = start_chunk.dst
        next_chunk  = self.get_next_chunk(start_chunk)

        while next_chunk:
            path.append(next_chunk.get_chunk_surface())
            path_num.append(next_chunk_num)
            # i, jのパスが存在した場合はそのまま返す
            if next_chunk_num == j:
                path[-1] = next_chunk.get_surface_sub_noun("Y")
                return " -> ".join(path)

            next_chunk_num = next_chunk.dst
            next_chunk = self.get_next_chunk(next_chunk)
        
        # i, jのパスがなかった場合
        # j から根に向かってchunkを見ていき, kを見つける
        j_path = []
        start_chunk = self.chunks[j]
        next_chunk_num = start_chunk.dst
        next_chunk  = self.get_next_chunk(start_chunk)
        j_path.append(start_chunk.get_surface_sub_noun("Y"))

        while next_chunk:
            j_path.append(next_chunk.get_chunk_surface())
            # 文節iと文節jから構文木の根に至る経路上での共通の文節kを見つけたい
            # next_chunk_numがkかどうか判定する
            if next_chunk_num in path_num:
                idx = path_num.index(next_chunk_num)
                path = path[:idx]
                break
            next_chunk_num = next_chunk.dst
            next_chunk = self.get_next_chunk(next_chunk)
        
        # path: xからkの一つ手前までのリスト
        # j_path yからkまでのリスト

        return " -> ".join(path) + " | " + " -> ".join(j_path[:-1]) + " | " + j_path[-1] 
        
            
    def get_next_chunk(self, chunk):
        if chunk.dst == -1:
            return None
        return self.chunks[chunk.dst]
    
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

verb_case_list = []

for sentence in sentences[0:10]:
    sentence.print_paths_between_nouns()

# Xは -> Yである
# Xで -> 生れたか | Yが | つかぬ
# Xでも -> 薄暗い -> じめじめした -> Yで
# Xでも -> 薄暗い -> じめじめした -> 所で | Y | 泣いて
# Xでも -> 薄暗い -> じめじめした -> 所で -> 泣いて | Yだけは | 記憶している
# Xでも -> 薄暗い -> じめじめした -> 所で -> 泣いて -> Yしている
# Xで | Y | 泣いて
# Xで -> 泣いて | Yだけは | 記憶している
# Xで -> 泣いて -> Yしている
# X -> 泣いて | Yだけは | 記憶している
# X -> 泣いて -> Yしている
# Xだけは -> Yしている
# Xは | Yで -> 始めて -> 人間という -> ものを | 見た
# Xは | Yという -> ものを | 見た
# Xは | Yを | 見た
# Xで -> 始めて -> Yという
# Xで -> 始めて -> 人間という -> Yを
# Xという -> Yを