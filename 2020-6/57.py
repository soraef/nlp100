import csv
import pickle
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report

CATEGORY2ID = {"b": 0, "t": 1, "e": 2, "m": 3}

def load_data(phase):
    phase_X = []
    with open(f"../data/{phase}.feature.txt", encoding="utf-8") as f:
        for row in csv.reader(f):
            row = list(map(int, row))
            phase_X.append(row)

    with open(f"../data/{phase}.txt", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        phase_Y = [CATEGORY2ID[row[0]] for row in reader]

    return np.array(phase_X), np.array(phase_Y)

train_X, train_Y = load_data("train")

# デフォルトのmax_iter=100だと係数が収束しないという警告でた
classifier = LogisticRegression(max_iter=1000)
classifier.fit(train_X, train_Y)

coef = np.array(classifier.coef_)
flatten_coef = coef.flatten()
asc_sorted_index  = flatten_coef.argsort()
dsc_sorted_index = asc_sorted_index[::-1]

with open("data/id2word.pickle", mode="rb") as f:
    id2word = pickle.load(f)

print("重みの高い特徴量")
for i in range(10):
    index = dsc_sorted_index[i]
    weight = flatten_coef[index]
    row = index // coef.shape[1]
    col = index %  coef.shape[1]
    print(f"coef[{row}][{col}] = {weight} ({id2word[col]})")

print("重みの低い特徴量")
for i in range(10):
    index = asc_sorted_index[i]
    weight = flatten_coef[index]
    row = index // coef.shape[1]
    col = index %  coef.shape[1]
    print(f"coef[{row}][{col}] = {weight} ({id2word[col]})")

# 出力


