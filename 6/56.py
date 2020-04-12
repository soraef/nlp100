import xml.etree.ElementTree as ET

tree = ET.parse("../data/nlp.txt.xml")
root = tree.getroot()

# 文字変換表を作成する
# reconf_table[sentence_id][token_id] = 変換したい文字
reconf_table = {}
coreferences = root.findall(".//coreference/coreference")

for coreference in coreferences:
    representative_text = ""
    for mention in coreference.iter("mention"):
        if mention.attrib.get("representative"):
            representative_text = mention.find("text").text
            continue

        text        = mention.find("text").text
        sentence_id = int(mention.find("sentence").text)
        start       = int(mention.find("start").text)
        end         = int(mention.find("end").text)

        # tableを初期化する
        for token_id in range(start, end):
            reconf_table[sentence_id] = {token_id: ""}
        
        # 変換したい文字を設定する
        reconf_table[sentence_id][start] = f"「{representative_text}({text})」"

# xmlをtextに置き換えていく
# 変換表に文字列があれば文字列を置き換える
def xml2text(root, reconf_table):
    sentences = root.findall(".//sentences/sentence")
    for sentence in sentences:
        sentence_id = int(sentence.attrib["id"])
        for token in sentence.iter("token"):
            token_id = int(token.attrib["id"])
            word = token.find("word").text
            text_or_none = reconf_table.get(sentence_id, {}).get(token_id, None)
            
            if text_or_none is None:
                print(word, end=" ")
            elif text_or_none:
                print(text_or_none, end=" ")
                

xml2text(root, reconf_table)