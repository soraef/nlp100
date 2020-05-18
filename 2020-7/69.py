import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE
import gensim
import csv
from sklearn.cluster import KMeans


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
            
pred = KMeans(n_clusters=5).fit_predict(country_vectors)

X_reduced = TSNE(n_components=2, random_state=0).fit_transform(country_vectors)

plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=pred)
plt.colorbar()
plt.savefig("69_result")