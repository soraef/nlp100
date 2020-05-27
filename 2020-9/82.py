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
        words = re.sub(r'[.,:;!?"]', "", words).split()
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
# loss: 1.1571088608179625
# train_acc: 0.5148366563699336
# acc: 0.5535580524344569
# == epoch: 1 ==
# loss: 1.0124334959203602
# train_acc: 0.6059159412150145
# acc: 0.6164794007490637
# == epoch: 2 ==
# loss: 0.8692381963026409
# train_acc: 0.6739679865206403
# acc: 0.697378277153558
# == epoch: 3 ==
# loss: 0.7417112771607535
# train_acc: 0.7234859121969485
# acc: 0.7161048689138577
# == epoch: 4 ==
# loss: 0.6603394892011969
# train_acc: 0.7523167649536647
# acc: 0.7348314606741573
# == epoch: 5 ==
# loss: 0.5864066359513174
# train_acc: 0.7847046709725732
# acc: 0.7580524344569288
# == epoch: 6 ==
# loss: 0.5160963787329975
# train_acc: 0.8144715903772348
# acc: 0.7640449438202247
# == epoch: 7 ==
# loss: 0.45350091318216906
# train_acc: 0.8377796499110737
# acc: 0.7782771535580524
# == epoch: 8 ==
# loss: 0.4001274161109828
# train_acc: 0.8600580361321726
# acc: 0.7932584269662921
# == epoch: 9 ==
# loss: 0.3537102997788416
# train_acc: 0.8760647758120378
# acc: 0.8052434456928839