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
# 重みの高い特徴量
# coef[3][2046] = 2.6927833094338336 (ebola)
# coef[1][521] = 2.6866998801562123 (google)
# coef[1][738] = 2.566171094551168 (facebook)
# coef[1][1067] = 2.2260165776902054 (climate)
# coef[3][1169] = 2.1620331444729044 (fda)
# coef[3][388] = 2.104317808163946 (mers)
# coef[3][1171] = 2.0957375640884703 (cancer)
# coef[0][346] = 2.052017734020029 (bank)
# coef[1][546] = 2.0091012859728794 (apple)
# coef[1][2819] = 1.9894211494797056 (heartbleed)
# 重みの低い特徴量
# coef[2][521] = -1.6105267036714461 (google)
# coef[0][1559] = -1.4861415312380326 (aereo)
# coef[2][440] = -1.268583690287424 (rise)
# coef[0][2271] = -1.2335232579193847 (gentiva)
# coef[2][238] = -1.216512164240966 (risk)
# coef[2][603] = -1.2118017248489763 (study)
# coef[2][36] = -1.2064895667522086 (oil)
# coef[0][4752] = -1.200265119161412 (activision)
# coef[2][60] = -1.16600280350957 (billion)
# coef[0][2046] = -1.136648624110002 (ebola)

