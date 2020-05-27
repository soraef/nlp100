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

        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.hidden_dim = hidden_dim
        self.rnn = nn.LSTM(emb_dim, hidden_dim, n_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.LogSoftmax(dim=1)

    # xs: [x_len, batch_size]
    def forward(self, xs, lengths):
        embedded = self.emb(xs)

        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)

        # LSTMは各時間の隠れ状態outputs
        # 最終的な隠れ状態hiddenと最終的なセルの状態cellが返される
        outputs, (hidden, cell) = self.rnn(packed)

        # outputs = [src len, batch size, hid dim * n directions]
        # hidden  = [n layers * n directions, batch size, hid dim]
        x = self.fc_out(hidden.squeeze())

        return self.softmax(x)



with open("word2id-80.pickle", "rb") as f:
    word2id = pickle.load(f)

def get_id(word):
    return word2id.get(word, 0)

VOCAB_SIZE = len(word2id.items())

def load_data(phase, batch_size=1, fix_len=30):
    phase_x = []
    phase_y = []
    lengths  = []
    with open(f"{phase}.txt") as f:
        data = f.readlines()

    
    for row in data:
        category = row.split("\t")[0]

        words = row.split("\t")[1]
        words = re.sub(r'[.,:;!?"]', "", words).split()
        words = map(lambda word: word.lower(), words)
        word_ids = list(map(get_id, words))
        lengths.append(len(word_ids))

        # word_idsを固定長にする
        add_len = fix_len - len(word_ids)

        # なぜかデータ右側(後方)をゼロ埋めすると精度が上がらない
        phase_x.append(word_ids + [0] * add_len)
        phase_y.append(CATEGORY2ID[category])

    batch_x = []
    batch_y = []
    batch_len = []
    start = 0
    end   = 0

    for i in range(batch_size, len(phase_x), batch_size):
        start = end
        end = i
        batch_x.append(phase_x[start:end])
        batch_y.append(phase_y[start:end])
        batch_len.append(lengths[start:end])

    return np.array(batch_x), np.array(batch_y), np.array(batch_len)

BATCH_SIZE = 1024
    
train_x, train_y, train_len = load_data("train", batch_size=BATCH_SIZE)
valid_x, valid_y, valid_len = load_data("valid", batch_size=256)

 
model = Net(emb_dim=300, hidden_dim=500, vocab_size=VOCAB_SIZE, output_dim=4).to(device)
loss_fn  =  nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def calc_acc(phase_x, phase_y, lengths, batch_size=1):
    correct = 0
    with torch.no_grad():
        for x, y, length in zip(phase_x, phase_y, lengths):
            x = torch.tensor(x).to(device)
            lengths= torch.tensor(length).to(device)
            pred_y = model(x, lengths).to(cpu)
            y_num = np.array(pred_y).argmax(axis=1)
            for y_num_i, y_i in zip(y_num, y):
                if y_num_i == y_i:
                    correct += 1
        
        print(f"acc: {correct/(len(phase_x) * batch_size)}")

for epoch in range(1, 101):
    total_loss = 0
    correct = 0
    for x, y, lengths in zip(train_x, train_y, train_len):
        x = torch.tensor(x).to(device)
        y = torch.tensor(y).to(device)
        lengths = torch.tensor(lengths).to(device)

        pred_y = model(x, lengths)

        loss = loss_fn(pred_y, y)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()
        with torch.no_grad():
            total_loss += loss.item()
            y_num = np.array(pred_y.to(cpu)).argmax(axis=1)
            for y_num_i, y_i in zip(y_num, y.to(cpu)):
                if y_num_i == y_i:
                    correct += 1
    if epoch % 10 == 0:
        print(f"== epoch: {epoch} ==")
        print(f"loss: {total_loss / len(train_x)}")
        print(f"train_acc: {correct / (len(train_x) * BATCH_SIZE)}")
        calc_acc(valid_x, valid_y, valid_len, 256)

# SGDでの結果
# 
# == epoch: 1000 ==
# loss: 0.17879072427749634
# train_acc: 0.947265625
# acc: 0.8015625

# Adamでの結果
#  
# == epoch: 10 ==
# loss: 0.1316780373454094
# train_acc: 0.9599609375
# acc: 0.840625
# == epoch: 20 ==
# loss: 0.004030785011127591
# train_acc: 0.99873046875
# acc: 0.84140625
# == epoch: 30 ==
# loss: 0.0018415329279378057
# train_acc: 0.9990234375
# acc: 0.83359375
# == epoch: 40 ==
# loss: 0.0013483781833201647
# train_acc: 0.99892578125
# acc: 0.83046875
# == epoch: 50 ==
# loss: 0.0011797231622040273
# train_acc: 0.99892578125
# acc: 0.8265625
# == epoch: 60 ==
# loss: 0.0010868325945921243
# train_acc: 0.998828125
# acc: 0.82578125
# == epoch: 70 ==
# loss: 0.0010362311499193312
# train_acc: 0.998828125
# acc: 0.82578125
# == epoch: 80 ==
# loss: 0.001003012398723513
# train_acc: 0.998828125
# acc: 0.82734375
# == epoch: 90 ==
# loss: 0.0009795306948944926
# train_acc: 0.998828125
# acc: 0.82890625
# == epoch: 100 ==
# loss: 0.000962060084566474
# train_acc: 0.998828125
# acc: 0.82890625