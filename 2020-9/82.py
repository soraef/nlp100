import pickle
import re 
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

CATEGORY2ID = {"b": 0, "t": 1, "e": 2, "m": 3}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cpu = torch.device("cpu")

class Net(nn.Module):
    def __init__(self, emb_dim, hidden_dim, vocab_size, output_dim, n_layers=1):
        super().__init__()

        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.hidden_dim = hidden_dim
        self.rnn = nn.LSTM(emb_dim, hidden_dim, n_layers)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.LogSoftmax(dim=1)

    # xs: [x_len, batch_size]
    def forward(self, xs):
        embedded = self.emb(xs)

        # LSTMは各時間の隠れ状態outputs
        # 最終的な隠れ状態hiddenと最終的なセルの状態cellが返される
        outputs, (hidden, cell) = self.rnn(embedded)

        #outputs = [src len, batch size, hid dim * n directions]

        x = self.fc_out(hidden.view(-1, self.hidden_dim))

        return self.softmax(x)



with open("word2id-80.pickle", "rb") as f:
    word2id = pickle.load(f)

def get_id(word):
    return word2id.get(word, 0)

VOCAB_SIZE = len(word2id.items())

def load_data(phase):
    phase_x = []
    phase_y = []

    with open(f"{phase}.txt") as f:
        data = f.readlines()
    
    for row in data:
        category = row.split("\t")[0]

        words = row.split("\t")[1]
        words = re.sub(r'[.,:;!?"]', "", words)
        words = map(lambda word: word.lower(), words)
        word_ids = list(map(get_id, words))
        
        phase_x.append(word_ids)
        phase_y.append(CATEGORY2ID[category])
    
    return np.array(phase_x), np.array(phase_y)

train_x, train_y = load_data("train")
valid_x, valid_y = load_data("valid")

train_x = train_x[:]
train_y = train_y[:]

valid_x = valid_x[:]
valid_y = valid_y[:]
 
model = Net(emb_dim=300, hidden_dim=500, vocab_size=VOCAB_SIZE, output_dim=4).to(device)
loss_fn  = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def calc_acc(phase_x, phase_y):
    correct = 0
    with torch.no_grad():
        for x, y in zip(phase_x, phase_y):
            x = torch.tensor(x).view(-1, 1).to(device)
            pred_y = model(x).to(cpu)
            y_num = np.array(pred_y).argmax()
            if y_num == y:
                correct += 1
        
        print(f"acc: {correct/len(phase_x)}")

for epoch in range(10):
    total_loss = 0
    correct = 0
    for x, y in zip(train_x, train_y):
        x = torch.tensor(x).view(-1, 1).to(device)
        y = torch.tensor(y).view(1).to(device)

        pred_y = model(x)
        # print(y.size())
        # print(pred_y.size())

        loss = loss_fn(pred_y, y)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()
        with torch.no_grad():
            total_loss += loss.item()
            y_num = np.array(pred_y.to(cpu)).argmax()
            if y_num == y.to(cpu):
                correct += 1

    print(f"== epoch: {epoch} ==")
    print(f"loss: {total_loss / len(train_x)}")
    print(f"train_acc: {correct / len(train_x)}")
    calc_acc(valid_x, valid_y)

# == epoch: 0 ==
# loss: 1.1479430521102498
# train_acc: 0.47486661050266776
# acc: 0.4898876404494382
# == epoch: 1 ==
# loss: 1.134658599205592
# train_acc: 0.49068613685294393
# acc: 0.4966292134831461
# == epoch: 2 ==
# loss: 1.1296229435689416
# train_acc: 0.49995319666760274
# acc: 0.5056179775280899
# == epoch: 3 ==
# loss: 1.124059723717526
# train_acc: 0.5106243564541796
# acc: 0.5250936329588015
# == epoch: 4 ==
# loss: 1.1176115069107695
# train_acc: 0.5183001029673313
# acc: 0.5340823970037453
# == epoch: 5 ==
# loss: 1.1108004006399461
# train_acc: 0.5230740428718524
# acc: 0.5378277153558052
# == epoch: 6 ==
# loss: 1.103141737276613
# train_acc: 0.532809136010484
# acc: 0.5348314606741573
# == epoch: 7 ==
# loss: 1.0884535228180594
# train_acc: 0.544229149115417
# acc: 0.5438202247191011
# == epoch: 8 ==
# loss: 1.0497661320935314
# train_acc: 0.581952635027614
# acc: 0.597752808988764
# == epoch: 9 ==
# loss: 1.0073244174600462
# train_acc: 0.6120939810914537
# acc: 0.6074906367041198
    