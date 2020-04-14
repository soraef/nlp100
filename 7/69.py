from pymongo import MongoClient, DESCENDING
from pprint import pprint

clinet = MongoClient()
db = clinet["test"]

artists = db.artists.find({"tags.value": "dance"}).sort("rating.count", DESCENDING).limit(10)

for artist in artists:
    print(f'name: {artist["name"]} rating_count: {artist["rating"]["count"]}')