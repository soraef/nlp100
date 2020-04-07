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

def arrange_section(text):
    text = text.replace(" ", "")
    section = int(text.count("=") / 2) - 1
    section_name = text.replace("=", "")
    return f"{'  ' * section}{section}. {section_name}"

text = extract(title, file_path)
lines = text.split("\n")

lines = [line for line in lines if re.match(r"^==+.+==+$", line)]
lines = map(arrange_section, lines)

for line in lines:
    print(line)