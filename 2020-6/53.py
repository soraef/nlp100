import csv
import numpy as np
from sklearn.linear_model import LogisticRegression

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

pred_class = classifier.predict(train_X)
pred_proba = classifier.predict_proba(train_X)

print(pred_class[0])
print(pred_proba[0])

# 出力

# 2
# [0.00232934 0.00305629 0.99247411 0.00214026]

