import csv
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

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
test_X,  test_Y  = load_data("test")

# デフォルトのmax_iter=100だと係数が収束しないという警告でた
classifier = LogisticRegression(max_iter=1000)
classifier.fit(train_X, train_Y)

pred_train = classifier.predict(train_X)
pred_test  = classifier.predict(test_X)

accuracy_train = accuracy_score(train_Y, pred_train)
accuracy_test  = accuracy_score(test_Y,  pred_test)

print("学習データの正解率: %.2f" % accuracy_train)
print("検証データの正解率: %.2f" % accuracy_test)


# 出力
# 学習データの正解率: 0.98
# 検証データの正解率: 0.88