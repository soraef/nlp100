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
    def __init__(self, hidden_dim, output_dim, wv_model, n_layers=1):
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
    word_data = wv_model.vocab.get(word, None)
    if word_data:
        return word_data.index
    else:
        return 0

VOCAB_SIZE = len(word2id.items())

def load_data(phase, batch_size=1, fix_len=92):
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

 
model = Net(hidden_dim=500, output_dim=4, wv_model=wv_model).to(device)
loss_fn  =  nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

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

for epoch in range(1, 1001):
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
        calc_acc(valid_x, valid_y, valid_len, VALID_BATCH_SIZE)

# SGD
# 
# == epoch: 1000 ==
# loss: 0.2327060415977385
# train_acc: 0.9213033536585366
# acc: 0.8703125

# Adamでやると
#
# == epoch: 10 ==
# loss: 0.2188988793187025
# train_acc: 0.9269245426829268
# acc: 0.8671875
# == epoch: 20 ==
# loss: 0.087769548885706
# train_acc: 0.971703506097561
# acc: 0.87734375
# == epoch: 30 ==
# loss: 0.02530766396624286
# train_acc: 0.9926638719512195
# acc: 0.86875
# == epoch: 40 ==
# loss: 0.011583722264664928
# train_acc: 0.9969512195121951
# acc: 0.8765625
# == epoch: 50 ==
# loss: 0.0033605176069569295
# train_acc: 0.9981897865853658
# acc: 0.88515625
# == epoch: 60 ==
# loss: 0.0024525736935618447
# train_acc: 0.9983803353658537
# acc: 0.88671875
# == epoch: 70 ==
# loss: 0.0024033279794200163
# train_acc: 0.9981897865853658
# acc: 0.8828125
# == epoch: 80 ==
# loss: 0.0019649325461104147
# train_acc: 0.9984756097560976
# acc: 0.88125
# == epoch: 90 ==
# loss: 0.0020075756715746914
# train_acc: 0.9981897865853658
# acc: 0.88046875
# == epoch: 100 ==
# loss: 0.00181182268325512
# train_acc: 0.9983803353658537
# acc: 0.88046875