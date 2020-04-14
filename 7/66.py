from pymongo import MongoClient
from pprint import pprint

clinet = MongoClient()
db = clinet["test"]

artists_japan_count = db.artists.count_documents({"area": "Japan"})
print(artists_japan_count)


# MongoDBのshell
#
# > use test
# switched to db test
# > db.artists.find({"area":"Japan"}).count()
# 22821
# 

#
# 出力
# 22821
#