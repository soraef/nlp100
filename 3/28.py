#
# 28
# 

import json
import gzip
import re
import pprint

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
        basic_info_dict[key_before] += "\n" + basic_info_line

def remove_strong(text):
    # '''hoge'''をhogeに置き換える
    return re.sub(r"''+(.+?)''+", r"\1", text)

def remove_internal_link(text):
    # [[group1|group2]] に対して
    # group2があればgroup2に置き換え
    # そうでなければgroup1に置き換える
    # ファイルの場合[[group1|group2|group3]]のようになるのでgroup3に置き換える
    return re.sub(r"\[\[(.+?)(?:\|(.+?))?(?:\|(.+?))?\]\]", lambda m: m.group(3) or m.group(2) or m.group(1), text)

def remove_external_link(text):
    return re.sub(r"\[http.*? (.*?)\]", r" \1", text)

def remove_tag(text):
    return re.sub(r"<.*?>", "", text)

def remove_list(text):
    return re.sub(r"\n\*{1,2}", "\n", text, re.M | re.S)

def remove_lang(text):
    return re.sub(r"\{\{lang\|.*?\|(.*?)\}\}", r"\1", text)

def remove_special_character(text):
    return re.sub(r"\s*\(&.*?\)", "", text)

def remove_markup(text):
    text = remove_list(text)
    text = remove_special_character(text)
    text = remove_tag(text)
    text = remove_external_link(text)
    text = remove_internal_link(text)
    text = remove_strong(text)
    text = remove_lang(text)
    return text
  
basic_info_dict_removed = {
    k: remove_markup(v) for k, v in basic_info_dict.items() 
}

pprint.pprint(basic_info_dict_removed)