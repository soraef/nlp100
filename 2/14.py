# 14
n = int(input())

with open("hightemp.txt") as f:
    texts =  f.read().rstrip("\n").split("\n")[:n]
    for text in texts:
        print(text)
        
# head -n 3 hightemp.txt 
# 高知県	江川崎	41	2013-08-12
# 埼玉県	熊谷	40.9	2007-08-16
# 岐阜県	多治見	40.9	2007-08-16