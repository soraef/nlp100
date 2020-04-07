# 09
import random

def shafful(word):
    if len(word) <= 4:
        return word
    middle = word[1:-1]
    suffle_word = "".join(random.sample(middle, len(middle)))
    return word[0] + suffle_word + word[-1]

text = "I couldn't believe that I could actually understand what I was reading : the phenomenal power of the human mind ."

words = text.split(" ")
shaffuled = map(shafful, words)

print(" ".join(shaffuled))