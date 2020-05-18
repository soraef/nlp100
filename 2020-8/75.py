import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load_data(phase):
    with open(f"../data/{phase}_x.pickle", "rb") as f:
        phase_x = np.array(pickle.load(f)).astype(np.float32)

    with open(f"../data/{phase}_y.pickle", "rb") as f:
        phase_y = np.array(pickle.load(f))

    return phase_x, phase_y


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(300, 4, bias=False)


    def forward(self, x):
        x = F.softmax(self.fc1(x), dim=-1)
        return x

# モデルと損失関数の定義
model = Net()
loss_fn  = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

# データの準備
train_x, train_y = load_data("train")
test_x, test_y   = load_data("test")

def calc_accuracy(phase_x, phase_y):
    correct_count = 0
    for x, y in zip(phase_x, phase_y):

        x = torch.tensor(x)
        y = np.array(y)

        with torch.no_grad():
            pred_y = model(x)
            pred_num = np.array(pred_y).argmax()
            if y == pred_num:
                correct_count += 1

    accuracy = correct_count / len(phase_x)
    print(f"accuracy: {accuracy}")

    return accuracy

train_acc_list = []
test_acc_list  = []
epochs = list(range(100))
for epoch in epochs:
    total_loss = 0
    for x, y in zip(train_x, train_y):
        x = torch.tensor(x)
        y = torch.tensor(y).view(1)

        pred_y = model(x).view(1, -1)

        # x_iに対する損失の計算
        loss = loss_fn(pred_y, y)

        # 勾配を初期化
        optimizer.zero_grad()

        # x_iに対する勾配の計算
        loss.backward()

        # パラメーターの更新
        optimizer.step()

        with torch.no_grad():
            total_loss += loss.item()

    print(f"loss: {total_loss / len(train_x)}")

    train_acc = calc_accuracy(train_x, train_y)
    test_acc  = calc_accuracy(test_x, test_y)

    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)

    plt.plot(epochs[:epoch+1], train_acc_list, color="r", label="train")
    plt.plot(epochs[:epoch+1], test_acc_list, color="b", label="test")

    plt.legend()
    plt.savefig("75_result")
    plt.close("all")









