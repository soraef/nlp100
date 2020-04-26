import csv
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

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

confusion_matrix_train = confusion_matrix(y_true=train_Y, y_pred=pred_train)
confusion_matrix_test  = confusion_matrix(y_true=test_Y,  y_pred=pred_test)

print(f"学習データの混同行列:\n{confusion_matrix_train}")
print(f"検証データの混同行列:\n{confusion_matrix_test}")


# 出力
# 学習データの混同行列:
# [[4490   19   25    1]
#  [  60 1131   24    0]
#  [  14    5 4192    1]
#  [  27    2    8  673]]
# 検証データの混同行列:
# [[514  14  15   2]
#  [ 38 109  16   3]
#  [ 18   7 492   1]
#  [ 19   5  25  56]]