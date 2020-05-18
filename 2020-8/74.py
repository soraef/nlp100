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

# モデルの読み込み
model = Net()
model.load_state_dict(torch.load("../data/73_model"))

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

    print(f"accuracy: {correct_count / len(phase_x)}")
    

# データの準備
train_x, train_y = load_data("train")
test_x, test_y   = load_data("test")

print("== train ==")
calc_accuracy(train_x, train_y)
print("== test ==")
calc_accuracy(test_x, test_y)

# == train ==
# accuracy: 0.7793484366223553
# == test ==
# accuracy: 0.7799401197604791