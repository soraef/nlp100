import gensim
import numpy as np

model = gensim.models.KeyedVectors.load_word2vec_format('../data/GoogleNews-vectors-negative300.bin', binary=True)
vector1 = model["United_States"]
vector2 = model["U.S."]

similarity = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

print(similarity)
print(model.similarity("United_States", "U.S."))

# 0.7310775
# 0.73107743