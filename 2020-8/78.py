import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(phase, batch_size):
    with open(f"../data/{phase}_x.pickle", "rb") as f:
        phase_x = np.array(pickle.load(f)).astype(np.float32)

    with open(f"../data/{phase}_y.pickle", "rb") as f:
        phase_y = np.array(pickle.load(f))

    batch_count = 0
    batch_x = []
    batch_y = []
    start = 0
    end   = 0

    for i in range(batch_size, len(phase_x), batch_size):
        start = end
        end = i
        batch_x.append(phase_x[start:end])
        batch_y.append(phase_y[start:end])

    return np.array(batch_x).astype(np.float32), np.array(batch_y)


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

train_x, train_y = load_data("train", batch_size=1024)

model   = model.to(device)

for epochs in range(1000):
    total_loss = 0
    for x, y in zip(train_x, train_y):
        x = torch.tensor(x).to(device)
        y = torch.tensor(y).to(device)

        pred_y = model(x)

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











