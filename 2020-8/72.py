import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def load_data(phase):
    with open(f"../data/{phase}_x.pickle", "rb") as f:
        phase_x = np.array(pickle.load(f)).astype(np.float32)

    with open(f"../data/{phase}_y.pickle", "rb") as f:
        phase_y = np.array(pickle.load(f))

    return phase_x, phase_y


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(300, 4)

    def forward(self, x):
        x = F.softmax(self.fc1(x), dim=-1)
        return x

# モデルと損失関数の定義
model = Net()
loss_fn  = nn.CrossEntropyLoss()


# データの準備
train_x, train_y = load_data("train")

x = torch.tensor(train_x[0])
y = torch.tensor(train_y[0]).view(1)
pred_y = model(x).view(1, -1)

# x_1に対する損失の計算
loss = loss_fn(pred_y, y)
print(loss.item())

# x_1に対する勾配の計算
loss.backward()


# 勾配の初期化
model.zero_grad()

# 集合事例の準備
X = torch.tensor(train_x[0:3])
Y = torch.tensor(train_y[0:3])
pred_Y = model(X)

# 集合事例に対する損失の計算
loss = loss_fn(pred_Y, Y)
print(loss.item())

# 集合事例に対する勾配の計算
loss.backward()




