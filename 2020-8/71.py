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

model = Net()

train_x, train_y = load_data("train")

x = torch.tensor(train_x[0])
y = model(x)
print(y)

X = torch.tensor(train_x[0:3])
Y = model(X)
print(Y)