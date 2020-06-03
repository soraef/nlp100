import MeCab
import os

mecab = MeCab.Tagger("-Owakati")

sources = []
targets = []

with open("../data/twitter_pair_example.txt") as f:
    data = f.readlines()

    for row in data:
        if not "\t" in row:
            continue
        row = row.replace("\n", "")
        source = row.split("\t")[0]
        target = row.split("\t")[1]

        parsed_source = mecab.parse(source)
        parsed_target = mecab.parse(target)

        sources.append(parsed_source)
        targets.append(parsed_target)

# train:valid:test = 8:1:1に分割
end_train_index = int(len(sources) * 0.8)
end_valid_index = int(len(sources) * 0.9)

train_sources = sources[:end_train_index]
train_targets = targets[:end_train_index]
valid_sources = sources[end_train_index:end_valid_index]
valid_targets = targets[end_train_index:end_valid_index]
test_sources  = sources[end_valid_index:]
test_targets = targets[end_valid_index:]

def write_data(phase, is_target, data):
    prefix = "tgt" if is_target else "src"
    with open(f"data/{prefix}-{phase}.txt", "w") as f:
        for row in data:
            f.write(row)

os.makedirs("data", exist_ok=True)

write_data("train", False, train_sources)
write_data("train", True , train_targets)
write_data("valid", False, valid_sources)
write_data("valid", True , valid_targets)
write_data("test" , False, test_sources)
write_data("test" , True , test_targets)


# bash
# 
# onmt_preprocess -train_src data/src-train.txt -train_tgt data/tgt-train.txt -valid_src data/src-valid.txt -valid_tgt data/tgt-valid.txt -save_data data/demo
