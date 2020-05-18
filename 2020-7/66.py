import csv

import numpy as np
import gensim

model = gensim.models.KeyedVectors.load_word2vec_format('../data/GoogleNews-vectors-negative300.bin', binary=True)

human_sims  = []
vector_sims = []

with open("../data/combined.csv") as f:
    reader = csv.reader(f)

    # ヘッダを読み飛ばす
    next(reader)

    for row in reader:
        sim = model.similarity(row[0], row[1])
        human_sim = float(row[2])

        human_sims.append(human_sim)
        vector_sims.append(sim)


human_sims  = np.array(human_sims)
vector_sims = np.array(vector_sims)

human_sorted_index  = human_sims.argsort()[::-1]
vector_sorted_index = vector_sims.argsort()[::-1]

N = len(human_sims)

human_sim_rank  = np.zeros(N)
vector_sim_rank = np.zeros(N)

human_rank  = 1
vector_rank = 1

for human_index, vector_index in zip(human_sorted_index, vector_sorted_index):
    human_sim_rank[human_index]   = human_rank
    vector_sim_rank[vector_index] = vector_rank

    human_rank  += 1
    vector_rank += 1


print(1 - (6*sum((human_sim_rank - vector_sim_rank)**2) / (N*(N**2 - 1))))

# 0.6997112576768793