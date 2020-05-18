import gensim
import numpy as np

model = gensim.models.KeyedVectors.load_word2vec_format('../data/GoogleNews-vectors-negative300.bin', binary=True)

with open("../data/questions-words.txt", encoding="utf-8", newline="") as f:
    lines = list(map(lambda x: x.split(), f.readlines()))

write_lines = []
for line in lines[1:]:
    if line[0] == ":":
        write_lines.append(" ".join(line) + "\n")
        continue
    sim = model.most_similar(positive=[line[1], line[2]], negative=[line[0]], topn=1)
    word = sim[0][0]
    similarity = str(sim[0][1])
    write_line = line
    write_line.append(word)
    write_line.append(similarity)
    text = " ".join(write_line) + "\n"
    print(text)
    write_lines.append(text)

with open("64_result.txt", mode="w", encoding="utf-8", newline="") as f:
    f.writelines(write_lines)