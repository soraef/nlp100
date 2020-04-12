import xml.etree.ElementTree as ET
import re

tree = ET.parse("../data/nlp.txt.xml")
root = tree.getroot()

def find_np(s):
    for np in re.finditer(r"\(NP", s):
        stack = ["("]
        cnt   = np.start() + 3

        while stack:
            if s[cnt] == "(":
                stack.append("(")
            elif s[cnt] == ")":
                stack.pop()
            cnt += 1

        np_s = s[np.start():cnt]
        print(np_s)
        remove_s(np_s)
        break

def remove_s(s):
    print(re.sub(r"^\(.*?\s(.*)\)$", r"\1", s))

for parse in root.iter("parse"):
    find_np(parse.text)
    break
