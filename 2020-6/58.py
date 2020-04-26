import csv
import pickle
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt
import japanize_matplotlib

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

    return np.array(phase_X, dtype='float32'), np.array(phase_Y, dtype='float32')

train_X, train_Y = load_data("train")
test_X,  test_Y  = load_data("test")
valid_X, valid_Y = load_data("valid")

CLIST = [0.01, 0.1, 1, 10, 100, 1000, 10000]
result = {
    "train": [],
    "test":  [],
    "valid": [],
}

for c in CLIST:
    classifier = LogisticRegression(max_iter=2000, C=c)
    classifier.fit(train_X, train_Y)

    pred_train = classifier.predict(train_X)
    pred_test  = classifier.predict(test_X)
    pred_valid  = classifier.predict(valid_X)

    accuracy_train = accuracy_score(train_Y, pred_train)
    accuracy_test  = accuracy_score(test_Y,  pred_test)
    accuracy_valid  = accuracy_score(valid_Y,  pred_valid)

    result["train"].append(accuracy_train)
    result["test"].append(accuracy_test)
    result["valid"].append(accuracy_valid)

with open("data/result_dict.pickle", mode="wb") as f:
    pickle.dump(result, f)



plt.plot(CLIST, result["train"], label="train")
plt.plot(CLIST, result["test"], label="test")
plt.plot(CLIST, result["valid"], label="valid")
plt.xlabel("正則化パラメータ")
plt.ylabel("正解率")
plt.xscale("log")
plt.legend()
plt.savefig("58_result.png")