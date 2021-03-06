# 24
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

file_names = re.findall(r"(?:File|ファイル):(.+?)\|", text)

for name in file_names:
    print(name)