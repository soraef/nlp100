# 12
import csv

with open("hightemp.txt", encoding="utf-8") as f:
    reader = list(csv.reader(f, delimiter = "\t"))

col1 = [row[0] for row in reader]
col2 = [row[1] for row in reader]

print(col1)

with open("col1.txt", mode="w", encoding="utf-8") as f:
    f.writelines("\n".join(col1))

with open("col2.txt", mode="w",  encoding="utf-8") as f:
    f.writelines("\n".join(col2))
    
# cut -f 1 hightemp.txt
# 高知県
# 埼玉県
# 岐阜県
# 山形県
# 山梨県
# 和歌山県
# 静岡県
# 山梨県
# 埼玉県
# 群馬県
# 群馬県
# 愛知県
# 千葉県
# 静岡県
# 愛媛県
# 山形県
# 岐阜県
# 群馬県
# 千葉県
# 埼玉県
# 大阪府
# 山梨県
# 山形県
# 愛知県

# cut -f 2 hightemp.txt
# 江川崎
# 熊谷
# 多治見
# 山形
# 甲府
# かつらぎ
# 天竜
# 勝沼
# 越谷
# 館林
# 上里見
# 愛西
# 牛久
# 佐久間
# 宇和島
# 酒田
# 美濃
# 前橋
# 茂原
# 鳩山
# 豊中
# 大月
# 鶴岡
# 名古屋