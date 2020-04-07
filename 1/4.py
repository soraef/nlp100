text = "Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can."
element2pos = {}
words = text.replace(",", " ").replace(".",  " ").split()

for i, c in enumerate(words):
    number = i + 1
    if(number in [1, 5, 6, 7, 8, 9, 15, 16, 19]):
        element2pos[c[0]] = number
    else:
        element2pos[c[0:2]] = number
print(element2pos)