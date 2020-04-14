from pymongo import MongoClient
from pprint import pprint

clinet = MongoClient()
db = clinet["test"]

name = input("名前を入力してください: ")

artists = db.artists.find({"aliases.name": name})
pprint(list(artists))

#
# 出力
#

# 名前を入力してください: LiSA
# [{'_id': 711029,
#   'aliases': [{'name': 'lxixsxa', 'sort_name': 'lxixsxa'},
#               {'name': 'LiSA', 'sort_name': 'リサ'}],
#   'area': 'Japan',
#   'begin': {'date': 24, 'month': 6, 'year': 1978},
#   'ended': True,
#   'gender': 'Female',
#   'gid': '85d76093-9865-4605-97fa-8c910929d366',
#   'name': 'LiSA',
#   'sort_name': 'LiSA',
#   'type': 'Person'}]