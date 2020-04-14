from pymongo import MongoClient
from pymongo import IndexModel, ASCENDING, DESCENDING
import json
import gzip

clinet = MongoClient()
db = clinet["test"]
articles = []

with gzip.open("../data/artist.json.gz", "rt", encoding="utf-8") as f:
    for line in f.readlines():
        data = json.loads(line)
        # jsonのキーidを_idに変更
        # mongoDBでは_idを識別子としているため
        data["_id"] = data.pop("id", None)
        articles.append(data)

db.artists.insert_many(articles)

db.artists.create_index([("name", ASCENDING)])
db.artists.create_index([("aliases.name", ASCENDING)])
db.artists.create_index([("tags.value", ASCENDING)])
db.artists.create_index([("rating.value", ASCENDING)])

print(sorted(list(db.artists.index_information())))


#
# 出力
#
# ['_id_', 'aliases.name_1', 'name_1', 'rating.value_1', 'tags.value_1']
#


