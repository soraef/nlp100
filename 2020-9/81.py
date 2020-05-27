import torch
import torch.nn as nn
import torch.nn.functional as F

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

        x = self.fc_out(outputs[-1])

        return self.softmax(x)

model = Net(emb_dim=300, hidden_dim=50, vocab_size=5000, output_dim=4)

x = torch.tensor([1, 2, 3, 4]).view(-1, 1)
print(model(x))
        