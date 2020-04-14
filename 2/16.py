#16
n = int(input())

with open("hightemp.txt") as f:
    data = f.read().rstrip("\n").split("\n")

data_len = len(data)

num = data_len // n    #dataをn等分した時の個数
add_count = data_len % n #dataのあまりの個数

start = 0
for i in range(n):
    # add_countが0より大きいときはnum + 1個のデータを追加する
    end = start + num + (add_count > 0) 
    with open(f"hightemp_split_{i}.txt", mode="w") as f:
        f.writelines("\n".join(data[start:end]))
    start = end
    add_count -= 1

# split --number=l/5 hightemp.txt hightemp_
# hightemp_aa, hightemp_ab, hightemp_ac, hightemp_ad, hightemp_ae
# に分割される