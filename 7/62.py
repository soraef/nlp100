import redis

redis = redis.Redis(host="localhost", port=6379, db=0)

keys = redis.keys("*")
count = 0
areas = redis.mget(keys)

for area in areas:
    if area.decode() == "Japan":
        count += 1

print(count)
