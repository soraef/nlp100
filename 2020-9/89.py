import torch
import string
from transformers import BertTokenizer, BertModel
import torch.nn as nn
from transformers import BertForSequenceClassification


import gensim
import pickle
import re 
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchtext

CATEGORY2ID = {"b": 0, "t": 1, "e": 2, "m": 3}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cpu = torch.device("cpu")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def preprocessing(text, tokenizer=tokenizer.tokenize):
    text = re.sub('\r', '', text)
    text = re.sub('\n', '', text)

    for p in string.punctuation:
        if (p == '.') or (p == ","):
            continue
        else:
            text = text.replace(p, " ")
   
    text = text.replace("."," . ")
    text = text.replace(","," , ")
   
    return tokenizer(text.lower())


def load_data(batch_size):
    TEXT = torchtext.data.Field(sequential=True, 
                                tokenize=preprocessing, 
                                use_vocab=True, 
                                include_lengths=True,
                                batch_first=True,
                                fix_length=30,
                                init_token='[CLS]',
                                eos_token='[SEP]',
                                pad_token='[PAD]',
                                unk_token='[UNK]',
                                )

    LABEL = torchtext.data.Field(sequential=False, use_vocab=False, preprocessing=lambda l: CATEGORY2ID[l])

    train_ds, val_ds, test_ds = torchtext.data.TabularDataset.splits(
        path='./', 
        train='train.txt',
        validation='valid.txt',
        test='test.txt',
        format='tsv',
        skip_header=True,
        fields=[('label', LABEL),('title', TEXT)]
    )

    TEXT.build_vocab(train_ds, min_freq=1)
    TEXT.vocab.stoi = tokenizer.vocab

    train_dl = torchtext.data.Iterator(train_ds, batch_size=batch_size, train=True)
    val_dl = torchtext.data.Iterator(val_ds, batch_size=batch_size, train=False, sort=False)
    test_dl = torchtext.data.Iterator(test_ds, batch_size=batch_size, train=False, sort=False)

    return train_dl, val_dl, test_dl


BATCH_SIZE = 256
VALID_BATCH_SIZE = 256
    
# データセットの準備
train, valid, test = load_data(batch_size=BATCH_SIZE)

for batch in train:
    print(batch.title[0][0])
    for i in range(50):
        text = batch.title[0][i].numpy()
        print(tokenizer.convert_ids_to_tokens(text))
    print(batch.label)

    print(train.dataset)
    print(len(train.dataset))
    break

# モデルのロード
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4).to(device)

for name, param in model.named_parameters():
    param.requires_grad = False

for name, param in model.bert.encoder.layer[-1].named_parameters():
    param.requires_grad = True

for name, param in model.classifier.named_parameters():
    param.requires_grad = True

optimizer = torch.optim.Adam([
    {'params': model.bert.encoder.layer[-1].parameters(), 'lr': 5e-5},
    {'params': model.classifier.parameters(), 'lr': 5e-5}])

loss_fn = nn.CrossEntropyLoss()

def calc_acc(phase, batch_size=1):
    correct = 0
    with torch.no_grad():
        for batch in phase:
            x = batch.title[0].to(device)
            y = batch.label.to(device)

            outputs = model(x, labels=y)

            loss, logits = outputs[:2]
            _, preds = torch.max(logits, 1)

            correct += torch.sum(preds == y.data)
            print(correct)
        
        print(f"acc: {correct.double() / len(phase.dataset)}")

for epoch in range(1, 100):
    total_loss = 0
    correct = 0
    model.train()
    for batch in train:
        x = batch.title[0].to(device)
        y = batch.label.to(device)

        outputs = model(x, labels=y)

        loss, logits = outputs[:2]
        _, preds = torch.max(logits, 1)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        with torch.no_grad():
            total_loss += loss.item()
            correct += torch.sum(preds == y.data)


    if epoch % 1 == 0:
        model.eval()
        print(f"== epoch: {epoch} ==")
        print(f"loss: {total_loss / len(train)}")
        print(f"train_acc: {correct.double() / len(train.dataset)}")
        calc_acc(valid, BATCH_SIZE)

# == epoch: 99 ==
# loss: 0.013669364037923515
# train_acc: 0.9947575360419397
# acc: 0.931784107946027