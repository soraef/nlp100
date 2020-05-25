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

X = torch.tensor(train_x[0:4])
Y = model(X)
print(Y)

# tensor([0.2396, 0.2401, 0.2590, 0.2613], grad_fn=<SoftmaxBackward>)
# tensor([[0.2396, 0.2401, 0.2590, 0.2613],
#         [0.2457, 0.2501, 0.2573, 0.2469],
#         [0.2448, 0.2507, 0.2529, 0.2516],
#         [0.2515, 0.2464, 0.2534, 0.2488]], grad_fn=<SoftmaxBackward>)