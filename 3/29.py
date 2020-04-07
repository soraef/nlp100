#
# 29
# 

import json
import gzip
import re
import requests

file_path  = "jawiki-country.json.gz"
title      = "イギリス"

def extract(title, from_path):
    with gzip.open(from_path, "rt", encoding="utf-8") as f:
        lines = f.read().rstrip("\n").split("\n")
        for line in lines:
            data = json.loads(line)
            if data["title"] == title:
                return data["text"]

def find_basic_info(text):
    # 正規表現.が改行にもマッチするようにre.Sを指定
    # 行の先頭と末尾が正規表現^, $にマッチするようにre.Mを指定
    return re.search(r"^{{基礎情報(?:.*?\n)(.*?)[\n]?}}$", text, re.S | re.M).groups()[0]

text = extract(title, file_path)
basic_info_text = find_basic_info(text)
basic_info_lines = basic_info_text.split("\n")

basic_info_dict = {}
key_before = ""

# 複数行にわたる基礎情報を結合して辞書に格納する
for basic_info_line in basic_info_lines:
    if basic_info_line[0] == "|":
        basic_info = re.match(r"\|(.+?)\s*=\s*(.+)", basic_info_line).groups()
        basic_info_dict[basic_info[0]] = basic_info[1]
        key_before = basic_info[0]
    elif key_before:
        basic_info_dict[key_before] += basic_info_line
    

URL = "https://en.wikipedia.org/w/api.php"
img_name = basic_info_dict["国旗画像"]

S = requests.Session()

PARAMS = {
    "action": "query",
    "format": "json",
    "prop": "imageinfo",
    "titles": "File:" + img_name,
    "iiprop": "url"
}

R = S.get(url=URL, params=PARAMS)
DATA = R.json()

url = DATA["query"]["pages"].popitem()[1]["imageinfo"][0]["url"]
print(url)
