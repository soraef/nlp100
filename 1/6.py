def ngram(n, words):
    ngrams = []
    
    for i in range(len(words)):
        if(i+n > len(words)):
            break
        ngrams.append(words[i:i+n])

    return ngrams

text1 = "paraparaparadise"
text2 = "paragraph"

X = set(ngram(2, text1))
Y = set(ngram(2, text2))

print(X | Y)       #和集合
print(X & Y)    #積集合
print(X - Y)      #差集合
print("se" in X)#Xにseが含まれる
print("se" in Y)#Yにseが含まれる