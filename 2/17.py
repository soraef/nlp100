# 17
import csv

with open("hightemp.txt", encoding="utf-8") as f:
    reader = list(csv.reader(f, delimiter = "\t"))
    
reader = [row[0] for row in reader]
prefs = list(set(reader))
sorted_prefs = sorted(sorted(prefs), key=len)

for pref in sorted_prefs:
    print(pref)

# cut -f 1 hightemp.txt | sort | uniq
# 千葉県
# 埼玉県
# 大阪府
# 山形県
# 山梨県
# 岐阜県
# 愛媛県
# 愛知県
# 群馬県
# 静岡県
# 高知県
# 和歌山県