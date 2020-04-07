text = "Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics."

words = text.replace(",", " ").replace(".",  " ").split()
words_count = map(len, words)

print(list(words_count))