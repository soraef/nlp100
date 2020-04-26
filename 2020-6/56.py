import csv
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
test_X,  test_Y  = load_data("test")

# デフォルトのmax_iter=100だと係数が収束しないという警告でた
classifier = LogisticRegression(max_iter=1000)
classifier.fit(train_X, train_Y)

pred_train = classifier.predict(train_X)
pred_test  = classifier.predict(test_X)

classification_report_test  = classification_report(y_true=test_Y, y_pred=pred_test)

print(f"検証データ:\n{classification_report_test}")


# 出力
# 検証データ:
#               precision    recall  f1-score   support

#            0       0.87      0.94      0.91       545
#            1       0.81      0.66      0.72       166
#            2       0.90      0.95      0.92       518
#            3       0.90      0.53      0.67       105

#     accuracy                           0.88      1334
#    macro avg       0.87      0.77      0.81      1334
# weighted avg       0.88      0.88      0.87      1334

# micro avg = accuracyなのでmicro avgは表示されないらしい
# https://rf00.hatenablog.com/entry/2020/03/22/141453
