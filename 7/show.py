import redis
import json
import gzip



with gzip.open("../data/artist.json.gz", "rt", encoding="utf-8") as f:
   print(f.readline())