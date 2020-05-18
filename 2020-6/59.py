import csv
import pickle
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt

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

CLIST = [1, 10]
PENALTY = ["l1", "l2", "elasticnet"]
SOLVER = ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]
TOL    = [1e-4, 1e-5, 1e-3]

result = {
    "train": [],
    "test":  [],
    "valid": [],
}

for c in CLIST:
    for penalty in PENALTY:
        for solver in SOLVER:
            for tol in TOL:
                try:
                    classifier = LogisticRegression(max_iter=2000, C=c, penalty=penalty, solver=solver, tol=tol)
                    classifier.fit(train_X, train_Y)
                except ValueError as e:
                    continue



                pred_train = classifier.predict(train_X)
                pred_test  = classifier.predict(test_X)
                pred_valid  = classifier.predict(valid_X)

                accuracy_train = accuracy_score(train_Y, pred_train)
                accuracy_test  = accuracy_score(test_Y,  pred_test)
                accuracy_valid  = accuracy_score(valid_Y,  pred_valid)
                
                print("param:")
                print(f"c: {c} penalty: {penalty} solver: {solver} tol: {tol}")
                print(f"train_acc: {accuracy_train} test_acc: {accuracy_test} valid_acc: {accuracy_valid}")

                result["train"].append(accuracy_train)
                result["test"].append(accuracy_test)
                result["valid"].append(accuracy_valid)

                with open("data/result_dict.pickle", mode="wb") as f:
                    pickle.dump(result, f)

# Solver newton-cg supports only 'l2' or 'none' penalties, got l1 penalty.
# Solver newton-cg supports only 'l2' or 'none' penalties, got l1 penalty.
# Solver newton-cg supports only 'l2' or 'none' penalties, got l1 penalty.
# Solver lbfgs supports only 'l2' or 'none' penalties, got l1 penalty.
# Solver lbfgs supports only 'l2' or 'none' penalties, got l1 penalty.
# Solver lbfgs supports only 'l2' or 'none' penalties, got l1 penalty.
# param:
# c: 1 penalty: l1 solver: liblinear tol: 0.0001
# train_acc: 0.950669287653281 test_acc: 0.8959580838323353 valid_acc: 0.8726591760299626
# param:
# c: 1 penalty: l1 solver: liblinear tol: 1e-05
# train_acc: 0.950669287653281 test_acc: 0.8959580838323353 valid_acc: 0.8726591760299626
# param:
# c: 1 penalty: l1 solver: liblinear tol: 0.001
# train_acc: 0.950669287653281 test_acc: 0.8959580838323353 valid_acc: 0.8734082397003745
# Solver sag supports only 'l2' or 'none' penalties, got l1 penalty.
# Solver sag supports only 'l2' or 'none' penalties, got l1 penalty.
# Solver sag supports only 'l2' or 'none' penalties, got l1 penalty.
# /root/.pyenv/versions/3.8.2/lib/python3.8/site-packages/scipy/optimize/linesearch.py:314: LineSearchWarning: The line search algorithm did not converge
#   warn('The line search algorithm did not converge', LineSearchWarning)
# /root/.pyenv/versions/3.8.2/lib/python3.8/site-packages/sklearn/utils/optimize.py:204: UserWarning: Line Search failed
#   warnings.warn('Line Search failed')
# param:
# c: 1 penalty: l2 solver: newton-cg tol: 0.0001
# train_acc: 0.9841804736497238 test_acc: 0.8997005988023952 valid_acc: 0.8816479400749063
# /root/.pyenv/versions/3.8.2/lib/python3.8/site-packages/scipy/optimize/linesearch.py:314: LineSearchWarning: The line search algorithm did not converge
#   warn('The line search algorithm did not converge', LineSearchWarning)
# /root/.pyenv/versions/3.8.2/lib/python3.8/site-packages/sklearn/utils/optimize.py:204: UserWarning: Line Search failed
#   warnings.warn('Line Search failed')
# param:
# c: 1 penalty: l2 solver: newton-cg tol: 1e-05
# train_acc: 0.9841804736497238 test_acc: 0.8997005988023952 valid_acc: 0.8816479400749063
# /root/.pyenv/versions/3.8.2/lib/python3.8/site-packages/scipy/optimize/linesearch.py:314: LineSearchWarning: The line search algorithm did not converge
#   warn('The line search algorithm did not converge', LineSearchWarning)
# /root/.pyenv/versions/3.8.2/lib/python3.8/site-packages/sklearn/utils/optimize.py:204: UserWarning: Line Search failed
#   warnings.warn('Line Search failed')
# param:
# c: 1 penalty: l2 solver: newton-cg tol: 0.001
# train_acc: 0.9841804736497238 test_acc: 0.8997005988023952 valid_acc: 0.8816479400749063
# param:
# c: 1 penalty: l2 solver: lbfgs tol: 0.0001
# train_acc: 0.9841804736497238 test_acc: 0.8997005988023952 valid_acc: 0.8816479400749063
# param:
# c: 1 penalty: l2 solver: lbfgs tol: 1e-05
# train_acc: 0.9841804736497238 test_acc: 0.8997005988023952 valid_acc: 0.8816479400749063
# param:
# c: 1 penalty: l2 solver: lbfgs tol: 0.001
# train_acc: 0.9841804736497238 test_acc: 0.8997005988023952 valid_acc: 0.8816479400749063
# param:
# c: 1 penalty: l2 solver: liblinear tol: 0.0001
# train_acc: 0.9715435739024618 test_acc: 0.9034431137724551 valid_acc: 0.8883895131086142
# param:
# c: 1 penalty: l2 solver: liblinear tol: 1e-05
# train_acc: 0.9715435739024618 test_acc: 0.9034431137724551 valid_acc: 0.8883895131086142
# param:
# c: 1 penalty: l2 solver: liblinear tol: 0.001
# train_acc: 0.9716371805672563 test_acc: 0.9034431137724551 valid_acc: 0.8883895131086142
# param:
# c: 1 penalty: l2 solver: sag tol: 0.0001
# train_acc: 0.9840868669849293 test_acc: 0.8997005988023952 valid_acc: 0.8816479400749063
# param:
# c: 1 penalty: l2 solver: sag tol: 1e-05
# train_acc: 0.9840868669849293 test_acc: 0.8997005988023952 valid_acc: 0.8816479400749063
# param:
# c: 1 penalty: l2 solver: sag tol: 0.001
# train_acc: 0.9840868669849293 test_acc: 0.8997005988023952 valid_acc: 0.8816479400749063
# Solver newton-cg supports only 'l2' or 'none' penalties, got elasticnet penalty.
# Solver newton-cg supports only 'l2' or 'none' penalties, got elasticnet penalty.
# Solver newton-cg supports only 'l2' or 'none' penalties, got elasticnet penalty.
# Solver lbfgs supports only 'l2' or 'none' penalties, got elasticnet penalty.
# Solver lbfgs supports only 'l2' or 'none' penalties, got elasticnet penalty.
# Solver lbfgs supports only 'l2' or 'none' penalties, got elasticnet penalty.
# Only 'saga' solver supports elasticnet penalty, got solver=liblinear.
# Only 'saga' solver supports elasticnet penalty, got solver=liblinear.
# Only 'saga' solver supports elasticnet penalty, got solver=liblinear.
# Solver sag supports only 'l2' or 'none' penalties, got elasticnet penalty.
# Solver sag supports only 'l2' or 'none' penalties, got elasticnet penalty.
# Solver sag supports only 'l2' or 'none' penalties, got elasticnet penalty.
# Solver newton-cg supports only 'l2' or 'none' penalties, got l1 penalty.
# Solver newton-cg supports only 'l2' or 'none' penalties, got l1 penalty.
# Solver newton-cg supports only 'l2' or 'none' penalties, got l1 penalty.
# Solver lbfgs supports only 'l2' or 'none' penalties, got l1 penalty.
# Solver lbfgs supports only 'l2' or 'none' penalties, got l1 penalty.
# Solver lbfgs supports only 'l2' or 'none' penalties, got l1 penalty.
# param:
# c: 10 penalty: l1 solver: liblinear tol: 0.0001
# train_acc: 0.9983150800336984 test_acc: 0.8772455089820359 valid_acc: 0.8674157303370786
# param:
# c: 10 penalty: l1 solver: liblinear tol: 1e-05
# train_acc: 0.9983150800336984 test_acc: 0.8772455089820359 valid_acc: 0.8674157303370786
# param:
# c: 10 penalty: l1 solver: liblinear tol: 0.001
# train_acc: 0.9983150800336984 test_acc: 0.875748502994012 valid_acc: 0.8674157303370786
# Solver sag supports only 'l2' or 'none' penalties, got l1 penalty.
# Solver sag supports only 'l2' or 'none' penalties, got l1 penalty.
# Solver sag supports only 'l2' or 'none' penalties, got l1 penalty.
# /root/.pyenv/versions/3.8.2/lib/python3.8/site-packages/scipy/optimize/linesearch.py:314: LineSearchWarning: The line search algorithm did not converge
#   warn('The line search algorithm did not converge', LineSearchWarning)
# /root/.pyenv/versions/3.8.2/lib/python3.8/site-packages/sklearn/utils/optimize.py:204: UserWarning: Line Search failed
#   warnings.warn('Line Search failed')
# param:
# c: 10 penalty: l2 solver: newton-cg tol: 0.0001
# train_acc: 0.9985022933632874 test_acc: 0.8937125748502994 valid_acc: 0.8741573033707866
# /root/.pyenv/versions/3.8.2/lib/python3.8/site-packages/scipy/optimize/linesearch.py:314: LineSearchWarning: The line search algorithm did not converge
#   warn('The line search algorithm did not converge', LineSearchWarning)
# /root/.pyenv/versions/3.8.2/lib/python3.8/site-packages/sklearn/utils/optimize.py:204: UserWarning: Line Search failed
#   warnings.warn('Line Search failed')
# param:
# c: 10 penalty: l2 solver: newton-cg tol: 1e-05
# train_acc: 0.9985022933632874 test_acc: 0.8937125748502994 valid_acc: 0.8741573033707866
# param:
# c: 10 penalty: l2 solver: newton-cg tol: 0.001
# train_acc: 0.9985022933632874 test_acc: 0.8937125748502994 valid_acc: 0.8741573033707866
# param:
# c: 10 penalty: l2 solver: lbfgs tol: 0.0001
# train_acc: 0.9985022933632874 test_acc: 0.8937125748502994 valid_acc: 0.8741573033707866
# param:
# c: 10 penalty: l2 solver: lbfgs tol: 1e-05
# train_acc: 0.9985022933632874 test_acc: 0.8937125748502994 valid_acc: 0.8741573033707866
# param:
# c: 10 penalty: l2 solver: lbfgs tol: 0.001
# train_acc: 0.9985022933632874 test_acc: 0.8937125748502994 valid_acc: 0.8741573033707866
# param:
# c: 10 penalty: l2 solver: liblinear tol: 0.0001
# train_acc: 0.9961621267434241 test_acc: 0.8944610778443114 valid_acc: 0.8741573033707866
# param:
# c: 10 penalty: l2 solver: liblinear tol: 1e-05
# train_acc: 0.9961621267434241 test_acc: 0.8944610778443114 valid_acc: 0.8741573033707866
# param:
# c: 10 penalty: l2 solver: liblinear tol: 0.001
# train_acc: 0.9961621267434241 test_acc: 0.8937125748502994 valid_acc: 0.8756554307116104
# param:
# c: 10 penalty: l2 solver: sag tol: 0.0001
# train_acc: 0.9985022933632874 test_acc: 0.8937125748502994 valid_acc: 0.8741573033707866
# /root/.pyenv/versions/3.8.2/lib/python3.8/site-packages/sklearn/linear_model/_sag.py:329: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
#   warnings.warn("The max_iter was reached which means "
# param:
# c: 10 penalty: l2 solver: sag tol: 1e-05
# train_acc: 0.9985022933632874 test_acc: 0.8937125748502994 valid_acc: 0.8741573033707866
# param:
# c: 10 penalty: l2 solver: sag tol: 0.001
# train_acc: 0.9985022933632874 test_acc: 0.8929640718562875 valid_acc: 0.8749063670411985
# Solver newton-cg supports only 'l2' or 'none' penalties, got elasticnet penalty.
# Solver newton-cg supports only 'l2' or 'none' penalties, got elasticnet penalty.
# Solver newton-cg supports only 'l2' or 'none' penalties, got elasticnet penalty.
# Solver lbfgs supports only 'l2' or 'none' penalties, got elasticnet penalty.
# Solver lbfgs supports only 'l2' or 'none' penalties, got elasticnet penalty.
# Solver lbfgs supports only 'l2' or 'none' penalties, got elasticnet penalty.
# Only 'saga' solver supports elasticnet penalty, got solver=liblinear.
# Only 'saga' solver supports elasticnet penalty, got solver=liblinear.
# Only 'saga' solver supports elasticnet penalty, got solver=liblinear.
# Solver sag supports only 'l2' or 'none' penalties, got elasticnet penalty.
# Solver sag supports only 'l2' or 'none' penalties, got elasticnet penalty.
# Solver sag supports only 'l2' or 'none' penalties, got elasticnet penalty.


# 一番高くなるのは
# 
# param:
# c: 1 penalty: l2 solver: liblinear tol: 0.001
# train_acc: 0.9716371805672563 test_acc: 0.9034431137724551 valid_acc: 0.8883895131086142