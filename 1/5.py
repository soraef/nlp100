def ngram(n, words):
    ngrams = []
    
    for i in range(len(words)):
        if(i+n > len(words)):
            break
        ngrams.append(words[i:i+n])

    return ngrams

text = "I am an NLPer"
words = text.split(" ")

print(ngram(2, text))
print(ngram(2, words))