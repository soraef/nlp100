import xml.etree.ElementTree as ET

tree = ET.parse("../data/nlp.txt.xml")
root = tree.getroot()

for token in root.iter("token"):
    word = token.find("word").text
    ner = token.find("NER").text
    if ner == "PERSON":
        print(word)

# 
# 出力
# 
# Alan
# Turing
# Joseph
# Weizenbaum
# MARGIE
# Schank
# Wilensky
# Meehan
# Lehnert
# Carbonell
# Lehnert
# Racter
# Jabberwacky
# Moore