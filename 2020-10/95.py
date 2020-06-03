import sentencepiece as sp
import os

tokenizer = sp.SentencePieceProcessor()
tokenizer.load("../data/wiki-ja.model")

def parse(text):
    return " ".join(tokenizer.EncodeAsPieces(text))

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

        parsed_source = parse(source) + "\n"
        parsed_target = parse(target) + "\n"

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
    with open(f"data/{prefix}-{phase}-sub.txt", "w") as f:
        for row in data:
            f.write(row)

os.makedirs("data", exist_ok=True)

write_data("train", False, train_sources)
write_data("train", True , train_targets)
write_data("valid", False, valid_sources)
write_data("valid", True , valid_targets)
write_data("test" , False, test_sources)
write_data("test" , True , test_targets)


# User> 今日はいい天気だと思いませんか
# Sys> はい!

# BLEU = 0.60, 23.2/4.6/3.0/1.9 (BP=0.120, ratio=0.321, hyp_len=259229, ref_len=808695)


# beam_max_len: 1
# BLEU = 0.00, 92.7/0.0/0.0/0.0 (BP=0.000, ratio=0.037, hyp_len=300, ref_len=8023)
# beam_max_len: 21
# BLEU = 0.29, 23.6/3.6/2.3/1.4 (BP=0.071, ratio=0.274, hyp_len=2197, ref_len=8023)
# beam_max_len: 41
# BLEU = 0.30, 23.1/3.5/2.2/1.4 (BP=0.077, ratio=0.280, hyp_len=2250, ref_len=8023)
# beam_max_len: 61
# BLEU = 0.31, 22.8/3.5/2.2/1.3 (BP=0.080, ratio=0.284, hyp_len=2276, ref_len=8023)
# beam_max_len: 81
# BLEU = 0.32, 22.6/3.4/2.2/1.3 (BP=0.083, ratio=0.286, hyp_len=2296, ref_len=8023)
# beam_max_len: 101
# BLEU = 0.32, 22.7/3.5/2.2/1.3 (BP=0.082, ratio=0.286, hyp_len=2291, ref_len=8023)