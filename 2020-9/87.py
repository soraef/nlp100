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
    def __init__(self, hidden_dim, output_dim, wv_model, n_layers=1, src_len=92):
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
        self.src_len = src_len
        self.cnn = nn.Conv2d(in_channels=1, 
                out_channels=hidden_dim, 
                kernel_size=(3, emb_dim), 
                stride=1,
                padding=(1,0))
        self.max_pool = nn.MaxPool1d(kernel_size=src_len)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, xs):
        # embedded: (batch_size, src_len, emb_dim)
        embedded = self.emb(xs)

        # cnn: (batch_size, out_channel_size, src_len, 1)
        cnn = self.cnn(embedded.unsqueeze(1))

        x = F.relu(cnn)
        x = self.max_pool(x.squeeze()).squeeze()
        x = self.fc_out(x)

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
            pred_y = model(x).to(cpu)
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

        pred_y = model(x)

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


# == epoch: 10 ==
# loss: 0.9807376425440718
# train_acc: 0.7562881097560976
# acc: 0.74453125
# == epoch: 20 ==
# loss: 0.7576966939902887
# train_acc: 0.7652439024390244
# acc: 0.75390625
# == epoch: 30 ==
# loss: 0.6504146703859655
# train_acc: 0.772484756097561
# acc: 0.75859375
# == epoch: 40 ==
# loss: 0.5893845936147178
# train_acc: 0.778391768292683
# acc: 0.75859375
# == epoch: 50 ==
# loss: 0.5434168976981465
# train_acc: 0.7946836890243902
# acc: 0.77734375
# == epoch: 60 ==
# loss: 0.5029816365823513
# train_acc: 0.8209794207317073
# acc: 0.80078125
# == epoch: 70 ==
# loss: 0.46532496301139276
# train_acc: 0.8445121951219512
# acc: 0.82265625
# == epoch: 80 ==
# loss: 0.4306818275916867
# train_acc: 0.8612804878048781
# acc: 0.8375
# == epoch: 90 ==
# loss: 0.4000071032745082
# train_acc: 0.8751905487804879
# acc: 0.846875
# == epoch: 100 ==
# loss: 0.3737961090192562
# train_acc: 0.8834794207317073
# acc: 0.859375