# 15
n = int(input())

with open("hightemp.txt") as f:
    texts = f.read().rstrip("\n").split("\n")[-n:]
    for text in texts:
        print(text)
        
# tail -n 3 hightemp.txt 
# 山梨県	大月	39.9	1990-07-19
# 山形県	鶴岡	39.9	1978-08-03
# 愛知県	名古屋	39.9	1942-08-02