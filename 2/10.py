# 10
with open("hightemp.txt") as f:
    print(f.read().count("\n"))
    
# 確認
#  wc -l hightemp.txt
#        24 hightemp.txt