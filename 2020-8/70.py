import gensim 
import numpy as np
import pickle
import re

model = gensim.models.KeyedVectors.load_word2vec_format('../data/GoogleNews-vectors-negative300.bin', binary=True)
CATEGORY2ID = {"b": 0, "t": 1, "e": 2, "m": 3}

def make_data(phase):
    phase_x = []
    phase_y = []

    with open(f"../data/{phase}.txt") as f:
        data = f.readlines()

        for row in data:
            category = row.split("\t")[0]
            text     = row.split("\t")[1]
            words = re.sub(r'[.,:;!?"]', "", text).split()
            x = np.zeros(300)
            T = 0 # 見出しに含まれる単語数(word2vecでベクトル化できないものは除く)
            for word in words:
                try:
                    wv = model[word]
                    x += wv
                    T += 1
                except KeyError:
                    print(f"{word} not in vocabukary")
                    
            # すべての単語がベクトル化できないものは除外する
            if T == 0:
                continue

            x = x / T
            y = CATEGORY2ID[category]

            phase_x.append(x)
            phase_y.append(y)

    with open(f"../data/{phase}_x.pickle", "wb") as f:
        pickle.dump(phase_x, f)
    
    with open(f"../data/{phase}_y.pickle", "wb") as f:
        pickle.dump(phase_y, f)

make_data("test")
make_data("valid")
make_data("train")



        
        