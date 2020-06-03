import MeCab
import os
import subprocess

mecab = MeCab.Tagger("-Owakati")

text = input("User> ")
parsed_text = mecab.parse(text)

with open("92-input.txt", "w") as f:
    f.write(parsed_text)

status = subprocess.call("onmt_translate -model demo-model_step_100000.pt -src 92-input.txt -output pred.txt -replace_unk -verbose > /dev/null 2>&1", shell=True)

if status == 0:
    with open("pred.txt") as f:
        text = f.read()
        print("Sys> ", end="")
        print(text.replace(" ", "").rstrip("\n"))

# User> 今日はいい天気だと思いませんか
# Sys> そうなんですね




