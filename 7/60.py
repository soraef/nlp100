import redis
import json
import gzip

redis = redis.Redis(host="localhost", port=6379, db=0)
redis.flushall()


with gzip.open("../data/artist.json.gz", "rt", encoding="utf-8") as f:
    for line in f.readlines():
        data = json.loads(line)
        redis.set(f"{data.get('name')}:{data.get('id')}", data.get("area", ""))



