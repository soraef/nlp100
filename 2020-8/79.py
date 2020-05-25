

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
cpu = torch.device("cpu")

def load_data(phase, batch_size):
    with open(f"{phase}_x.pickle", "rb") as f:
        phase_x = np.array(pickle.load(f)).astype(np.float32)

    with open(f"{phase}_y.pickle", "rb") as f:
        phase_y = np.array(pickle.load(f))

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

        self.fc1 = nn.Linear(300, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 100)
        self.fc4 = nn.Linear(100, 4)

        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)
        self.dropout3 = nn.Dropout(p=0.5)



    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.dropout3(x)
        x = F.relu(self.fc4(x))
        x = F.softmax(x, dim=-1)
        return x

# モデルと損失関数の定義
model = Net()
loss_fn  = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

train_x, train_y = load_data("train", batch_size=2048)
valid_x, valid_y = load_data("valid", batch_size=1)

model   = model.to(device)
def calc_accuracy(phase_x, phase_y, batch_size):
    correct_count = 0
    for x, y in zip(phase_x, phase_y):

        x = torch.tensor(x).to(device)
        y = np.array(y)

        with torch.no_grad():
            pred_y = model(x).to(cpu)
            for pred_y_i, y_i in zip(pred_y, y):
                pred_num = np.array(pred_y_i).argmax()
                if y_i == pred_num:
                    correct_count += 1

    accuracy = correct_count / (len(phase_x) * batch_size) 
    print(f"accuracy: {accuracy}")


for epochs in range(5000):
    total_loss = 0
    model.train()
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
    model.eval()
    if epochs % 100 == 0:
        calc_accuracy(valid_x, valid_y, 1)
        print(f"loss: {total_loss / len(train_x)}")


# 出力

# accuracy: 0.8860569715142429
# loss: 0.7646535992622375