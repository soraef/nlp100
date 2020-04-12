import xml.etree.ElementTree as ET

tree = ET.parse("../data/nlp.txt.xml")
root = tree.getroot()

for token in root.iter("token"):
    word = token.find("word").text
    ner = token.find("NER").text
    if ner == "PERSON":
        print(word)