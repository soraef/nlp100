import csv
import re
import pickle

def read_data(filename):
    with open(filename, encoding="utf-8") as f:
        data = list(csv.reader(f, delimiter="\t"))
    return data

train_data = read_data("../data/train.txt")
valid_data = read_data("../data/valid.txt")
test_data  = read_data("../data/test.txt")

word2count = {}

all_data = train_data + valid_data + test_data

# 単語の出現回数をカウント
for data in all_data:
    words = re.sub(r'[.,:;!?"]', "", data[1]).split()
    words = map(lambda word: word.lower(), words)
    for word in words:
        # wordがない場合に初期化
        word2count.setdefault(word, 0)
        word2count[word] += 1

# 出現回数が3回以下の単語と200回以上の単語をblack_listに追加
black_list = []
for word, count in word2count.items():
    if count <= 3 or count >= 200:
        black_list.append(word)


# 出現回数の調査
# count_list = [v for k, v in word2count.items()]

# for i in range(50):
#     count = 0
#     range_str = f"{i * 10}~{i * 10 + 9}までの出現回数"
#     for j in range(10):
#         count += count_list.count(i * 10 + j)
#     print(f"{range_str}: {count}")

word2id = {}
id2word = {}
new_id  = 0

# word2idを作成
for data in all_data:
    words = re.sub(r'[.,:;!?"]', "", data[1]).split()
    words = map(lambda word: word.lower(), words)
    for word in words:
        # ブラックリストの確認
        if word in black_list:
            continue
        
        if word2id.get(word, None) is None:
            word2id[word] = new_id
            id2word[new_id] = word
            new_id += 1

with open("data/id2word.pickle", mode="wb") as f:
    pickle.dump(id2word, f)

def make_feature(phase_data):
    features = []
    for data in phase_data:
        feature = [0 for i in range(len(word2id))]
        words = re.sub(r'[.,:;!?"]', "", data[1]).split()
        words = map(lambda word: word.lower(), words)

        for word in words:
            id = word2id.get(word, None)
            if id is None:
                continue

            feature[id] += 1

        features.append(feature)
    return features

def save_feature(filename, features):
    with open(filename, mode="w", encoding="utf-8", newline='') as f:
        writer = csv.writer(f)
        writer.writerows(features)

train_features = make_feature(train_data)
save_feature("../data/train.feature.txt", train_features)

valid_features = make_feature(valid_data)
save_feature("../data/valid.feature.txt", valid_features)

test_features = make_feature(test_data)
save_feature("../data/test.feature.txt", test_features)





# 次元数が多すぎるので出現回数が多い単語と少ない単語を除去して次元数を減らす

# 0~9までの出現回数: 16425
# 10~19までの出現回数: 1177
# 20~29までの出現回数: 446
# 30~39までの出現回数: 234
# 40~49までの出現回数: 120
# 50~59までの出現回数: 95
# 60~69までの出現回数: 48
# 70~79までの出現回数: 41
# 80~89までの出現回数: 31
# 90~99までの出現回数: 23
# 100~109までの出現回数: 16
# 110~119までの出現回数: 15
# 120~129までの出現回数: 12
# 130~139までの出現回数: 18
# 140~149までの出現回数: 9
# 150~159までの出現回数: 3
# 160~169までの出現回数: 10
# 170~179までの出現回数: 4
# 180~189までの出現回数: 5
# 190~199までの出現回数: 4
# 200~209までの出現回数: 1
# 210~219までの出現回数: 5
# 220~229までの出現回数: 3
# 230~239までの出現回数: 0
# 240~249までの出現回数: 7
# 250~259までの出現回数: 1
# 260~269までの出現回数: 1
# 270~279までの出現回数: 1
# 280~289までの出現回数: 1
# 290~299までの出現回数: 0
# 300~309までの出現回数: 0
# 310~319までの出現回数: 3
# 320~329までの出現回数: 0
# 330~339までの出現回数: 0
# 340~349までの出現回数: 1
# 350~359までの出現回数: 0
# 360~369までの出現回数: 0
# 370~379までの出現回数: 0
# 380~389までの出現回数: 0
# 390~399までの出現回数: 0
# 400~409までの出現回数: 0
# 410~419までの出現回数: 0
# 420~429までの出現回数: 1
# 430~439までの出現回数: 0
# 440~449までの出現回数: 1
# 450~459までの出現回数: 0
# 460~469までの出現回数: 0
# 470~479までの出現回数: 1
# 480~489までの出現回数: 0
# 490~499までの出現回数: 0