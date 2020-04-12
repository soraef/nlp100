import redis

redis = redis.Redis(host="localhost", port=6379, db=0)

name = input()
keys = redis.keys(f"{name}:*")

for key in keys:
    area = redis.get(key).decode()
    key  = key.decode()
    id   = key.split(":")[1]
    name = key.split(":")[0]
    print(f"id:{id}, name:{name}, area:{area}")
