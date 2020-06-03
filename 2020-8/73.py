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

        self.fc1 = nn.Linear(300, 4, bias=False)


    def forward(self, x):
        x = F.softmax(self.fc1(x), dim=-1)
        return x

# モデルと損失関数の定義
model = Net()
loss_fn  = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# データの準備
train_x, train_y = load_data("train")

for epoch in range(10):
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

torch.save(model.state_dict(), "../data/73_model")








