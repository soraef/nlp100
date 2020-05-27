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
    def __init__(self, hidden_dim, output_dim, wv_model, dropout=True, multi_cnn=True, src_len=35):
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
        self.is_dropout = dropout
        self.is_multi_cnn = multi_cnn

        self.cnn = nn.Conv2d(in_channels=1, 
                out_channels=hidden_dim, 
                kernel_size=(3, emb_dim), 
                stride=1,
                padding=(1,0))
        
        self.cnn2 = nn.Conv2d(in_channels=hidden_dim, 
                out_channels=hidden_dim, 
                kernel_size=(3, 1), 
                stride=1,
                padding=(1,0))

        self.max_pool = nn.MaxPool1d(kernel_size=src_len)
        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.2)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, xs):
        # embedded: (batch_size, src_len, emb_dim)
        embedded = self.emb(xs)

        # cnn: (batch_size, out_channel_size, src_len, 1)
        cnn = self.cnn(embedded.unsqueeze(1))

        x = F.relu(cnn)

        if self.is_dropout:
            x = self.dropout1(x)

        if self.is_multi_cnn:
            x = self.cnn2(x)
            x = F.relu(cnn)

            if self.is_dropout:
                x = self.dropout2(x)

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

def load_data(phase, batch_size=1, fix_len=35):
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

BATCH_SIZE = 256
VALID_BATCH_SIZE = 256
    
train_x, train_y, train_len = load_data("train", batch_size=BATCH_SIZE)
valid_x, valid_y, valid_len = load_data("valid", batch_size=VALID_BATCH_SIZE)

def calc_acc(phase_x, phase_y, model, batch_size=1):
    correct = 0
    with torch.no_grad():
        for x, y in zip(phase_x, phase_y):
            x = torch.tensor(x).to(device)
            pred_y = model(x).to(cpu)
            y_num = np.array(pred_y).argmax(axis=1)
            for y_num_i, y_i in zip(y_num, y):
                if y_num_i == y_i:
                    correct += 1
        
        # print(f"acc: {correct/(len(phase_x) * batch_size)}")


    return correct/(len(phase_x) * batch_size)

def train(params):

    model = Net(hidden_dim=params["hidden_dim"], output_dim=4, wv_model=wv_model, dropout=params["dropout"], multi_cnn=params["multi_cnn"]).to(device)   
    loss_fn  =  nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])

    max_acc = 0

    for epoch in range(1, 10):
        total_loss = 0
        correct = 0
        model.train()
        for x, y in zip(train_x, train_y):
            x = torch.tensor(x).to(device)
            y = torch.tensor(y).to(device)

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

        if epoch % 1 == 0:
            model.eval()
            # print(f"== epoch: {epoch} ==")
            # print(f"loss: {total_loss / len(train_x)}")
            # print(f"train_acc: {correct / (len(train_x) * BATCH_SIZE)}")
            acc = calc_acc(valid_x, valid_y, model, VALID_BATCH_SIZE)
            max_acc = max(acc, max_acc)
    
    print(params)
    print(f"max acc: {max_acc}")

HID_DIM = [50, 100, 500, 1000]
DROPOUT = [True, False]
MULTI_CNN = [True, False]
LR      = [1e-2, 1e-3, 1e-4, 1e-5]

params_list = []

for hidden_dim in HID_DIM:
    for dropout in DROPOUT:
        for multi_cnn in MULTI_CNN:
            for lr in LR:
                params = {}
                params["hidden_dim"] = hidden_dim
                params["dropout"] = dropout
                params["multi_cnn"] = multi_cnn
                params["lr"] = lr
                params_list.append(params)

# print(params_list)

for params in params_list:
    train(params)
                

# {'hidden_dim': 50, 'dropout': True, 'multi_cnn': True, 'lr': 0.01}
# max acc: 0.8953125
# {'hidden_dim': 50, 'dropout': True, 'multi_cnn': True, 'lr': 0.001}
# max acc: 0.8859375
# {'hidden_dim': 50, 'dropout': True, 'multi_cnn': True, 'lr': 0.0001}
# max acc: 0.7765625
# {'hidden_dim': 50, 'dropout': True, 'multi_cnn': True, 'lr': 1e-05}
# max acc: 0.49453125
# {'hidden_dim': 50, 'dropout': True, 'multi_cnn': False, 'lr': 0.01}
# max acc: 0.9015625
# {'hidden_dim': 50, 'dropout': True, 'multi_cnn': False, 'lr': 0.001}
# max acc: 0.88125
# {'hidden_dim': 50, 'dropout': True, 'multi_cnn': False, 'lr': 0.0001}
# max acc: 0.7609375
# {'hidden_dim': 50, 'dropout': True, 'multi_cnn': False, 'lr': 1e-05}
# max acc: 0.70078125
# {'hidden_dim': 50, 'dropout': False, 'multi_cnn': True, 'lr': 0.01}
# max acc: 0.8953125
# {'hidden_dim': 50, 'dropout': False, 'multi_cnn': True, 'lr': 0.001}
# max acc: 0.88203125
# {'hidden_dim': 50, 'dropout': False, 'multi_cnn': True, 'lr': 0.0001}
# max acc: 0.7609375
# {'hidden_dim': 50, 'dropout': False, 'multi_cnn': True, 'lr': 1e-05}
# max acc: 0.5859375
# {'hidden_dim': 50, 'dropout': False, 'multi_cnn': False, 'lr': 0.01}
# max acc: 0.89296875
# {'hidden_dim': 50, 'dropout': False, 'multi_cnn': False, 'lr': 0.001}
# max acc: 0.8828125
# {'hidden_dim': 50, 'dropout': False, 'multi_cnn': False, 'lr': 0.0001}
# max acc: 0.74921875
# {'hidden_dim': 50, 'dropout': False, 'multi_cnn': False, 'lr': 1e-05}
# max acc: 0.72265625
# {'hidden_dim': 100, 'dropout': True, 'multi_cnn': True, 'lr': 0.01}
# max acc: 0.9
# {'hidden_dim': 100, 'dropout': True, 'multi_cnn': True, 'lr': 0.001}
# max acc: 0.89140625
# {'hidden_dim': 100, 'dropout': True, 'multi_cnn': True, 'lr': 0.0001}
# max acc: 0.7953125
# {'hidden_dim': 100, 'dropout': True, 'multi_cnn': True, 'lr': 1e-05}
# max acc: 0.70546875
# {'hidden_dim': 100, 'dropout': True, 'multi_cnn': False, 'lr': 0.01}
# max acc: 0.903125
# {'hidden_dim': 100, 'dropout': True, 'multi_cnn': False, 'lr': 0.001}
# max acc: 0.8875
# {'hidden_dim': 100, 'dropout': True, 'multi_cnn': False, 'lr': 0.0001}
# max acc: 0.8
# {'hidden_dim': 100, 'dropout': True, 'multi_cnn': False, 'lr': 1e-05}
# max acc: 0.709375
# {'hidden_dim': 100, 'dropout': False, 'multi_cnn': True, 'lr': 0.01}
# max acc: 0.9
# {'hidden_dim': 100, 'dropout': False, 'multi_cnn': True, 'lr': 0.001}
# max acc: 0.89140625
# {'hidden_dim': 100, 'dropout': False, 'multi_cnn': True, 'lr': 0.0001}
# max acc: 0.7875
# {'hidden_dim': 100, 'dropout': False, 'multi_cnn': True, 'lr': 1e-05}
# max acc: 0.5109375
# {'hidden_dim': 100, 'dropout': False, 'multi_cnn': False, 'lr': 0.01}
# max acc: 0.89921875
# {'hidden_dim': 100, 'dropout': False, 'multi_cnn': False, 'lr': 0.001}
# max acc: 0.89296875
# {'hidden_dim': 100, 'dropout': False, 'multi_cnn': False, 'lr': 0.0001}
# max acc: 0.79375
# {'hidden_dim': 100, 'dropout': False, 'multi_cnn': False, 'lr': 1e-05}
# max acc: 0.7234375
# {'hidden_dim': 500, 'dropout': True, 'multi_cnn': True, 'lr': 0.01}
# max acc: 0.896875
# {'hidden_dim': 500, 'dropout': True, 'multi_cnn': True, 'lr': 0.001}
# max acc: 0.903125
# {'hidden_dim': 500, 'dropout': True, 'multi_cnn': True, 'lr': 0.0001}
# max acc: 0.8609375
# {'hidden_dim': 500, 'dropout': True, 'multi_cnn': True, 'lr': 1e-05}
# max acc: 0.73828125
# {'hidden_dim': 500, 'dropout': True, 'multi_cnn': False, 'lr': 0.01}
# max acc: 0.90546875
# {'hidden_dim': 500, 'dropout': True, 'multi_cnn': False, 'lr': 0.001}
# max acc: 0.90234375
# {'hidden_dim': 500, 'dropout': True, 'multi_cnn': False, 'lr': 0.0001}
# max acc: 0.8578125
# {'hidden_dim': 500, 'dropout': True, 'multi_cnn': False, 'lr': 1e-05}
# max acc: 0.7375
# {'hidden_dim': 500, 'dropout': False, 'multi_cnn': True, 'lr': 0.01}
# max acc: 0.8984375
# {'hidden_dim': 500, 'dropout': False, 'multi_cnn': True, 'lr': 0.001}
# max acc: 0.90078125
# {'hidden_dim': 500, 'dropout': False, 'multi_cnn': True, 'lr': 0.0001}
# max acc: 0.8515625
# {'hidden_dim': 500, 'dropout': False, 'multi_cnn': True, 'lr': 1e-05}
# max acc: 0.74140625
# {'hidden_dim': 500, 'dropout': False, 'multi_cnn': False, 'lr': 0.01}
# max acc: 0.896875
# {'hidden_dim': 500, 'dropout': False, 'multi_cnn': False, 'lr': 0.001}
# max acc: 0.89609375
# {'hidden_dim': 500, 'dropout': False, 'multi_cnn': False, 'lr': 0.0001}
# max acc: 0.85390625
# {'hidden_dim': 500, 'dropout': False, 'multi_cnn': False, 'lr': 1e-05}
# max acc: 0.73203125
# {'hidden_dim': 1000, 'dropout': True, 'multi_cnn': True, 'lr': 0.01}
# max acc: 0.90390625
# {'hidden_dim': 1000, 'dropout': True, 'multi_cnn': True, 'lr': 0.001}
# max acc: 0.903125
# {'hidden_dim': 1000, 'dropout': True, 'multi_cnn': True, 'lr': 0.0001}
# max acc: 0.86171875
# {'hidden_dim': 1000, 'dropout': True, 'multi_cnn': True, 'lr': 1e-05}
# max acc: 0.75
# {'hidden_dim': 1000, 'dropout': True, 'multi_cnn': False, 'lr': 0.01}
# max acc: 0.89921875
# {'hidden_dim': 1000, 'dropout': True, 'multi_cnn': False, 'lr': 0.001}
# max acc: 0.9
# {'hidden_dim': 1000, 'dropout': True, 'multi_cnn': False, 'lr': 0.0001}
# max acc: 0.865625
# {'hidden_dim': 1000, 'dropout': True, 'multi_cnn': False, 'lr': 1e-05}
# max acc: 0.753125
# {'hidden_dim': 1000, 'dropout': False, 'multi_cnn': True, 'lr': 0.01}
# max acc: 0.9046875
# {'hidden_dim': 1000, 'dropout': False, 'multi_cnn': True, 'lr': 0.001}
# max acc: 0.90390625
# {'hidden_dim': 1000, 'dropout': False, 'multi_cnn': True, 'lr': 0.0001}
# max acc: 0.8671875
# {'hidden_dim': 1000, 'dropout': False, 'multi_cnn': True, 'lr': 1e-05}
# max acc: 0.74765625
# {'hidden_dim': 1000, 'dropout': False, 'multi_cnn': False, 'lr': 0.01}
# max acc: 0.9
# {'hidden_dim': 1000, 'dropout': False, 'multi_cnn': False, 'lr': 0.001}
# max acc: 0.90234375
# {'hidden_dim': 1000, 'dropout': False, 'multi_cnn': False, 'lr': 0.0001}
# max acc: 0.8671875
# {'hidden_dim': 1000, 'dropout': False, 'multi_cnn': False, 'lr': 1e-05}
# max acc: 0.74453125

#　一番accのよかったパラメータ
# {'hidden_dim': 500, 'dropout': True, 'multi_cnn': False, 'lr': 0.01}
# max acc: 0.90546875