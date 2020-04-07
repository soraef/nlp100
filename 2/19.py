# 19
import csv

with open("hightemp.txt") as f:
    reader = list(csv.reader(f, delimiter = "\t"))
    
prefs = [data[0] for data in reader]
pref_list = list(set(prefs))
sorted_pref_list = sorted(pref_list, key=lambda pref: prefs.count(pref), reverse=True)

for row in sorted_pref_list:
    print(row)
    
# cut -f 1 hightemp.txt | sort | uniq -c | sort -r
#    3 群馬県
#    3 山梨県
#    3 山形県
#    3 埼玉県
#    2 静岡県
#    2 愛知県
#    2 岐阜県
#    2 千葉県
#    1 和歌山県
#    1 高知県
#    1 愛媛県
#    1 大阪府