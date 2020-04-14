from pymongo import MongoClient, DESCENDING
from pprint import pprint

clinet = MongoClient()
db = clinet["test"]

artists = db.artists.find({"tags.value": "dance"}).sort("rating.count", DESCENDING).limit(10)

for artist in artists:
    print(f'name: {artist["name"]} rating_count: {artist["rating"]["count"]}')

#
# 出力
#
# name: Madonna rating_count: 26
# name: Björk rating_count: 23
# name: The Prodigy rating_count: 23
# name: Rihanna rating_count: 15
# name: Britney Spears rating_count: 13
# name: Maroon 5 rating_count: 11
# name: Adam Lambert rating_count: 7
# name: Fatboy Slim rating_count: 7
# name: Basement Jaxx rating_count: 6
# name: Cornershop rating_count: 5
# 