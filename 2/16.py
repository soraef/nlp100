#16
n = int(input())

with open("hightemp.txt") as f:
    data = f.read().rstrip("\n").split("\n")

wc = len(data)

num = wc // n
add_num = wc % n

start = 0
for i in range(n):
    end = start + num + (add_num > 0)
    with open(f"hightemp_split_{i}.txt", mode="w") as f:
        f.writelines("\n".join(data[start:end]))
    start = end
    add_num -= 1