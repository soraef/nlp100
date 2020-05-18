import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import gensim

model = gensim.models.KeyedVectors.load_word2vec_format('../data/GoogleNews-vectors-negative300.bin', binary=True)

countries = []
country_vectors = []

with open("country.txt") as f:
    reader = csv.reader(f, delimiter="\t")
    for row in reader:
        country = row[0].split(",")[0].split("(")[0].replace(" ", "_")
        try:
            # print(model[country])
            country_vectors.append(model[country])
            countries.append(country)
        except KeyError:
            print(f"{country} not in vocabukary")


linkage_result = linkage(country_vectors, method='ward', metric='euclidean')

threshold = 0.7 * np.max(linkage_result[:, 2])

plt.figure(num=None, figsize=(16, 9), dpi=200, facecolor='w', edgecolor='k')
dendrogram(linkage_result, labels=countries, color_threshold=threshold)

plt.savefig("68_result")