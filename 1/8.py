import re

def convert(match):
    return chr(219 - ord(match.group()))

def cipher(text):
    return re.sub(r"[a-z]", convert, text)

plain_text = "ABCdefg1234あいうえおに"

encrypted_text = cipher(plain_text)
print(encrypted_text)

plain_text = cipher(encrypted_text)
print(plain_text)