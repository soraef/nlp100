import csv
import random

PUBLISHERS = ["Reuters", "Huffington Post", "Businessweek", "Contactmusic.com", "Daily Mail"] 

with open("../data/NewsAggregatorDataset/newsCorpora.csv", encoding="utf-8", newline="") as f:
    reader = csv.reader(f, delimiter = "\t", quotechar='|')
    data = f.readlines()
    data = map(lambda r: r.split("\t"), data)


    # csvファイルのカラム名は次の通り
    # ID TITLE URL PUBLISHER CATEGORY STORY HOSTNAME TIMESTAMP
    data = [row for row in data if row[3] in PUBLISHERS]

random.shuffle(data)

end_train_data = int(len(data) * 0.8)
end_valid_data = int(len(data) * 0.9)

train_data = data[:end_train_data]
valid_data = data[end_train_data:end_valid_data]
test_data  = data[end_valid_data:]

def write_data(filename, data):
    writing_data = [(row[4], row[1]) for row in data]
    with open(filename, mode="w", encoding="utf-8", newline='') as f:
        writer = csv.writer(f, delimiter = "\t")
        writer.writerows(writing_data)

write_data("../data/train2.txt", train_data)
write_data("../data/valid2.txt", valid_data)
write_data("../data/test2.txt", test_data)