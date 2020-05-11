import xml.etree.ElementTree as ET

tree = ET.parse("../data/nlp.txt.xml")
root = tree.getroot()

for word in root.iter("word"):
    print(word.text)


# 
# 出力（一部）
# 
# atural
# language
# processing
# From
# Wikipedia
# ,
# the
# free
# encyclopedia
# Natural
# language