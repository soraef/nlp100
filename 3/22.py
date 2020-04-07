# 21
import json
import gzip
import re

file_path  = "jawiki-country.json.gz"
title      = "イギリス"

def extract(title, from_path):
    with gzip.open(from_path, "rt", encoding="utf-8") as f:
        lines = f.read().rstrip("\n").split("\n")
        for line in lines:
            data = json.loads(line)
            if data["title"] == title:
                return data["text"]

text = extract(title, file_path)

# カテゴリーの部分のみほしいので丸かっこで囲んでグループ化
# カテゴリー以降の|*はいらないので(?:...)構文を使って取り出さないようにする
# 正規表現.*はそのままでは貪欲マッチしてしまうので, ?をつけて非貪欲にする
# .は改行以外のすべての文字にマッチする
categories = re.findall(r'\[\[Category:(.*?)(?:\|.*)?\]\]', text)

for category in categories:
    print(category)