import redis
import json
import gzip

redis = redis.Redis(host="localhost", port=6379, db=1)
redis.flushall()


with gzip.open("../data/artist.json.gz", "rt", encoding="utf-8") as f:
    for line in f.readlines():
        data = json.loads(line)
        tags = {}
        for tag in data.get("tags", [{"value": "", "count": 0}]):
            count = tag.get("count", 0)
            val = tag.get("value", "")
            tags[val] = count

        redis.hmset(f"{data.get('name')}:{data.get('id')}", tags)

print("アーティスト名を入力してください")
name = input()
keys = redis.keys(f"{name}:*")
for key in keys:
    tags = redis.hgetall(key)
    decode_key = key.decode()
    id = decode_key.split(":")[1]
    print(f"id:{id}, name:{name}")
    for k, v in tags.items():
        if k.decode():
            print(f"{k.decode()}: {v.decode()}")
        else:
            print("-")
