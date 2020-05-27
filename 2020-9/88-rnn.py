import gensim
import pickle
import re 
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

wv_model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
torch.cuda.empty_cache()

CATEGORY2ID = {"b": 0, "t": 1, "e": 2, "m": 3}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cpu = torch.device("cpu")

class Net(nn.Module):
    def __init__(self, hidden_dim, output_dim, wv_model, n_layers=2, dropout=0):
        super().__init__()

        # 事前学習済み単語ベクトルを用意
        weights = wv_model.vectors
        vocab_size = weights.shape[0]
        emb_dim = weights.shape[1]

        # embに事前学習済みのパラメータを適用する
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.emb.weight = nn.Parameter(torch.from_numpy(weights))
        self.emb.weight.requires_grad = False

        self.hidden_dim = hidden_dim
        self.n_layers   = n_layers
        self.rnn = nn.LSTM(emb_dim, hidden_dim, n_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.fc_out = nn.Linear(hidden_dim * 2, output_dim)
        self.softmax = nn.LogSoftmax(dim=1)

    # xs: [x_len, batch_size]
    def forward(self, xs, lengths):
        embedded = self.emb(xs)

        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)

        # LSTMは各時間の隠れ状態outputs
        # 最終的な隠れ状態hiddenと最終的なセルの状態cellが返される
        outputs, (hidden, cell) = self.rnn(packed)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True) #lstmからの最後の隠れ状態を入れる
        # outputs = [src len, batch size, hid dim * n directions]
        # hidden  = [n layers * n directions, batch size, hid dim]

        hidden = hidden.view(self.n_layers, 2, -1, self.hidden_dim)
        hidden_1 = hidden[-1, 0]
        hidden_2 = hidden[-1, 1]
    
        cat_hidden = torch.cat([hidden_1, hidden_2], dim=-1)
        
        x = self.fc_out(cat_hidden)

        return self.softmax(x)


with open("word2id-80.pickle", "rb") as f:
    word2id = pickle.load(f)

def get_id(word):
    word_data = wv_model.vocab.get(word, None)
    if word_data:
        return word_data.index
    else:
        return 0

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

BATCH_SIZE = 256
VALID_BATCH_SIZE = 256
    
train_x, train_y, train_len = load_data("train", batch_size=BATCH_SIZE)
valid_x, valid_y, valid_len = load_data("valid", batch_size=VALID_BATCH_SIZE)

def calc_acc(phase_x, phase_y, lengths, model, batch_size=1):
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
        
        # print(f"acc: {correct/(len(phase_x) * batch_size)}")

    return correct/(len(phase_x) * batch_size)


def train(params):

    model = Net(hidden_dim=params["hidden_dim"], output_dim=4, wv_model=wv_model, n_layers=params["n_layers"], dropout=params["dropout"]).to(device)
    loss_fn  =  nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
    max_acc = 0

    for epoch in range(1, 21):
        total_loss = 0
        correct = 0
        model.train()
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

        # print(f"== epoch: {epoch} ==")
        # print(f"loss: {total_loss / len(train_x)}")
        # print(f"train_acc: {correct / (len(train_x) * BATCH_SIZE)}")

        model.eval()
        acc = calc_acc(valid_x, valid_y, valid_len, model, VALID_BATCH_SIZE)
        max_acc = max(max_acc, acc)

    print(params)
    print(f"max_acc: {max_acc}")

HID_DIM = [50, 500, 1000]
DROPOUT = [0, 0.2, 0.5]
N_LAYERS = [1, 2, 3]
LR      = [1e-2, 1e-3]

params_list = []

for hidden_dim in HID_DIM:
    for dropout in DROPOUT:
        for n_layers in N_LAYERS:
            for lr in LR:
                params = {}
                params["hidden_dim"] = hidden_dim
                params["dropout"]    = dropout
                params["n_layers"]   = n_layers
                params["lr"]         = lr
                params_list.append(params)

for params in params_list:
    train(params)

# {'hidden_dim': 50, 'dropout': 0, 'n_layers': 1, 'lr': 0.01}
# max_acc: 0.89140625
# {'hidden_dim': 50, 'dropout': 0, 'n_layers': 1, 'lr': 0.001}
# max_acc: 0.878125
# {'hidden_dim': 50, 'dropout': 0, 'n_layers': 2, 'lr': 0.01}
# max_acc: 0.9015625
# {'hidden_dim': 50, 'dropout': 0, 'n_layers': 2, 'lr': 0.001}
# max_acc: 0.88359375
# {'hidden_dim': 50, 'dropout': 0, 'n_layers': 3, 'lr': 0.01}
# max_acc: 0.9
# {'hidden_dim': 50, 'dropout': 0, 'n_layers': 3, 'lr': 0.001}
# max_acc: 0.8828125
# /home/fukui/.pyenv/versions/3.7.4/lib/python3.7/site-packages/torch/nn/modules/rnn.py:54: UserWarning: dropout option adds dropout after all but last recurrent layer, so non-zero dropout expects num_layers greater than 1, but got dropout=0.2 and num_layers=1
#   "num_layers={}".format(dropout, num_layers))
# {'hidden_dim': 50, 'dropout': 0.2, 'n_layers': 1, 'lr': 0.01}
# max_acc: 0.890625
# {'hidden_dim': 50, 'dropout': 0.2, 'n_layers': 1, 'lr': 0.001}
# max_acc: 0.875
# {'hidden_dim': 50, 'dropout': 0.2, 'n_layers': 2, 'lr': 0.01}
# max_acc: 0.89375
# {'hidden_dim': 50, 'dropout': 0.2, 'n_layers': 2, 'lr': 0.001}
# max_acc: 0.8796875
# {'hidden_dim': 50, 'dropout': 0.2, 'n_layers': 3, 'lr': 0.01}
# max_acc: 0.8890625
# {'hidden_dim': 50, 'dropout': 0.2, 'n_layers': 3, 'lr': 0.001}
# max_acc: 0.88515625
# /home/fukui/.pyenv/versions/3.7.4/lib/python3.7/site-packages/torch/nn/modules/rnn.py:54: UserWarning: dropout option adds dropout after all but last recurrent layer, so non-zero dropout expects num_layers greater than 1, but got dropout=0.5 and num_layers=1
#   "num_layers={}".format(dropout, num_layers))
# {'hidden_dim': 50, 'dropout': 0.5, 'n_layers': 1, 'lr': 0.01}
# max_acc: 0.88828125
# {'hidden_dim': 50, 'dropout': 0.5, 'n_layers': 1, 'lr': 0.001}
# max_acc: 0.8796875
# {'hidden_dim': 50, 'dropout': 0.5, 'n_layers': 2, 'lr': 0.01}
# max_acc: 0.89375
# {'hidden_dim': 50, 'dropout': 0.5, 'n_layers': 2, 'lr': 0.001}
# max_acc: 0.88125
# {'hidden_dim': 50, 'dropout': 0.5, 'n_layers': 3, 'lr': 0.01}
# max_acc: 0.89765625
# {'hidden_dim': 50, 'dropout': 0.5, 'n_layers': 3, 'lr': 0.001}
# max_acc: 0.88203125
# {'hidden_dim': 500, 'dropout': 0, 'n_layers': 1, 'lr': 0.01}
# max_acc: 0.896875
# {'hidden_dim': 500, 'dropout': 0, 'n_layers': 1, 'lr': 0.001}
# max_acc: 0.878125
# {'hidden_dim': 500, 'dropout': 0, 'n_layers': 2, 'lr': 0.01}
# max_acc: 0.89453125
# {'hidden_dim': 500, 'dropout': 0, 'n_layers': 2, 'lr': 0.001}
# max_acc: 0.88515625
# {'hidden_dim': 500, 'dropout': 0, 'n_layers': 3, 'lr': 0.01}
# max_acc: 0.8953125
# {'hidden_dim': 500, 'dropout': 0, 'n_layers': 3, 'lr': 0.001}
# max_acc: 0.8828125
# {'hidden_dim': 500, 'dropout': 0.2, 'n_layers': 1, 'lr': 0.01}
# max_acc: 0.88984375
# {'hidden_dim': 500, 'dropout': 0.2, 'n_layers': 1, 'lr': 0.001}
# max_acc: 0.87890625
# {'hidden_dim': 500, 'dropout': 0.2, 'n_layers': 2, 'lr': 0.01}
# max_acc: 0.884375
# {'hidden_dim': 500, 'dropout': 0.2, 'n_layers': 2, 'lr': 0.001}
# max_acc: 0.88125
# {'hidden_dim': 500, 'dropout': 0.2, 'n_layers': 3, 'lr': 0.01}
# max_acc: 0.88125
# {'hidden_dim': 500, 'dropout': 0.2, 'n_layers': 3, 'lr': 0.001}
# max_acc: 0.87890625
# {'hidden_dim': 500, 'dropout': 0.5, 'n_layers': 1, 'lr': 0.01}
# max_acc: 0.89296875
# {'hidden_dim': 500, 'dropout': 0.5, 'n_layers': 1, 'lr': 0.001}
# max_acc: 0.88125
# {'hidden_dim': 500, 'dropout': 0.5, 'n_layers': 2, 'lr': 0.01}
# max_acc: 0.89609375
# {'hidden_dim': 500, 'dropout': 0.5, 'n_layers': 2, 'lr': 0.001}
# max_acc: 0.878125
# {'hidden_dim': 500, 'dropout': 0.5, 'n_layers': 3, 'lr': 0.01}
# max_acc: 0.878125
# {'hidden_dim': 500, 'dropout': 0.5, 'n_layers': 3, 'lr': 0.001}
# max_acc: 0.88515625
# {'hidden_dim': 1000, 'dropout': 0, 'n_layers': 1, 'lr': 0.01}
# max_acc: 0.88046875
# {'hidden_dim': 1000, 'dropout': 0, 'n_layers': 1, 'lr': 0.001}
# max_acc: 0.878125
# {'hidden_dim': 1000, 'dropout': 0, 'n_layers': 2, 'lr': 0.01}
# max_acc: 0.8078125
# {'hidden_dim': 1000, 'dropout': 0, 'n_layers': 2, 'lr': 0.001}
# max_acc: 0.88515625
# {'hidden_dim': 1000, 'dropout': 0, 'n_layers': 3, 'lr': 0.01}
# max_acc: 0.8515625
# {'hidden_dim': 1000, 'dropout': 0, 'n_layers': 3, 'lr': 0.001}
# max_acc: 0.8796875
# {'hidden_dim': 1000, 'dropout': 0.2, 'n_layers': 1, 'lr': 0.01}
# max_acc: 0.88125
# {'hidden_dim': 1000, 'dropout': 0.2, 'n_layers': 1, 'lr': 0.001}
# max_acc: 0.884375
# {'hidden_dim': 1000, 'dropout': 0.2, 'n_layers': 2, 'lr': 0.01}
# max_acc: 0.85625
# {'hidden_dim': 1000, 'dropout': 0.2, 'n_layers': 2, 'lr': 0.001}
# max_acc: 0.8875
# {'hidden_dim': 1000, 'dropout': 0.2, 'n_layers': 3, 'lr': 0.01}
# max_acc: 0.83125
# {'hidden_dim': 1000, 'dropout': 0.2, 'n_layers': 3, 'lr': 0.001}
# max_acc: 0.8828125
# {'hidden_dim': 1000, 'dropout': 0.5, 'n_layers': 1, 'lr': 0.01}
# max_acc: 0.89296875
# {'hidden_dim': 1000, 'dropout': 0.5, 'n_layers': 1, 'lr': 0.001}
# max_acc: 0.88125
# {'hidden_dim': 1000, 'dropout': 0.5, 'n_layers': 2, 'lr': 0.01}
# max_acc: 0.78828125
# {'hidden_dim': 1000, 'dropout': 0.5, 'n_layers': 2, 'lr': 0.001}
# max_acc: 0.8875
# {'hidden_dim': 1000, 'dropout': 0.5, 'n_layers': 3, 'lr': 0.01}
# max_acc: 0.81640625
# {'hidden_dim': 1000, 'dropout': 0.5, 'n_layers': 3, 'lr': 0.001}
# max_acc: 0.88984375

# {'hidden_dim': 50, 'dropout': 0, 'n_layers': 2, 'lr': 0.01}
# max_acc: 0.9015625