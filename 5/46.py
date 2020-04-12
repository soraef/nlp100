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

    def is_contain_particle(self):
        for morph in self.morphs:
            if morph.is_particle():
                return True
        return False

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

    def get_connected_two_chunks(self):
        two_chunks_list = []
        for chunk in self.chunks:
            if chunk.dst == -1:
                continue
            
            # 名詞を含む文節が動詞を含む文節にかからないときはループを飛ばす
            if not (chunk.is_contain_noun() and self.chunks[chunk.dst].is_contain_verb()):
                continue

            chunk_surface1 = chunk.get_chunk_surface(ignore_symbol=True)
            chunk_surface2 = self.chunks[chunk.dst].get_chunk_surface(ignore_symbol=True)

            if not chunk_surface1 or not chunk_surface2:
                continue

            surface = f"{chunk_surface1}\t{chunk_surface2}"
            two_chunks_list.append(surface)
        
        return two_chunks_list

    def get_verb_case(self):
        verb_cases = []
        for chunk in self.chunks:

            if not chunk.is_contain_verb():
                continue

            verb = chunk.get_first_verb()

            particle_chunks = []
            for src in chunk.srcs:
                if self.chunks[src].is_contain_particle():
                    particle = self.chunks[src].get_last_particles()
                    particle_chunks.append([particle, self.chunks[src].get_chunk_surface(True)])
            
            if not particle_chunks:
                continue
            
            sorted_particle_chunks = sorted(particle_chunks, key=lambda x: x[0])
            pprint.pprint(sorted_particle_chunks)
            particle_str = " ".join([particle[0] for particle in sorted_particle_chunks])
            chunk_str    = " ".join([particle[1] for particle in sorted_particle_chunks])
            pattern = f"{verb}\t{particle_str}\t{chunk_str}"
            verb_cases.append(pattern)

        return verb_cases
    
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

for sentence in sentences:
    verb_case_list.extend(sentence.get_verb_case())

with open("verb_case_46.txt", encoding="utf-8", mode="w") as f:
    f.writelines("\n".join(verb_case_list))
