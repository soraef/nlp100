import xml.etree.ElementTree as ET

tree = ET.parse("../data/nlp.txt.xml")
root = tree.getroot()

for token in root.iter("token"):
    word = token.find("word").text
    lemma = token.find("lemma").text
    pos = token.find("POS").text

    print(f"{word}\t{lemma}\t{pos}")
