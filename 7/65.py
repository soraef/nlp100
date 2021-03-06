from pymongo import MongoClient
from pprint import pprint

clinet = MongoClient()
db = clinet["test"]

queens = db.artists.find({"name": "Queen"})

for queen in queens:
    pprint(queen)

# 
# MongoDBのshell
#
# > use test
# switched to db test
# > db.artists.find({"name":"Queen"})
# { "_id" : 701492, "name" : "Queen", "area" : "Japan", "gender" : "Female", "tags" : [ { "count" : 1, "value" : "kamen rider w" }, { "count" : 1, "value" : "related-akb48" } ], "sort_name" : "Queen", "ended" : true, "gid" : "420ca290-76c5-41af-999e-564d7c71f1a7", "type" : "Character", "aliases" : [ { "name" : "Queen", "sort_name" : "Queen" } ] }
# { "_id" : 192, "rating" : { "count" : 24, "value" : 92 }, "begin" : { "date" : 27, "month" : 6, "year" : 1970 }, "name" : "Queen", "area" : "United Kingdom", "tags" : [ { "count" : 2, "value" : "hard rock" }, { "count" : 1, "value" : "70s" }, { "count" : 1, "value" : "queen family" }, { "count" : 1, "value" : "90s" }, { "count" : 1, "value" : "80s" }, { "count" : 1, "value" : "glam rock" }, { "count" : 4, "value" : "british" }, { "count" : 1, "value" : "english" }, { "count" : 2, "value" : "uk" }, { "count" : 1, "value" : "pop/rock" }, { "count" : 1, "value" : "pop-rock" }, { "count" : 1, "value" : "britannique" }, { "count" : 1, "value" : "classic pop and rock" }, { "count" : 1, "value" : "queen" }, { "count" : 1, "value" : "united kingdom" }, { "count" : 1, "value" : "langham 1 studio bbc" }, { "count" : 1, "value" : "kind of magic" }, { "count" : 1, "value" : "band" }, { "count" : 6, "value" : "rock" }, { "count" : 1, "value" : "platinum" } ], "sort_name" : "Queen", "ended" : true, "gid" : "0383dadf-2a4e-4d10-a46a-e9e041da8eb3", "type" : "Group", "aliases" : [ { "name" : "女王", "sort_name" : "女王" } ] }
# { "_id" : 992994, "ended" : true, "gid" : "5eecaf18-02ec-47af-a4f2-7831db373419", "sort_name" : "Queen", "name" : "Queen" }
# 

#
# 出力
# {'_id': 701492,
#  'aliases': [{'name': 'Queen', 'sort_name': 'Queen'}],
#  'area': 'Japan',
#  'ended': True,
#  'gender': 'Female',
#  'gid': '420ca290-76c5-41af-999e-564d7c71f1a7',
#  'name': 'Queen',
#  'sort_name': 'Queen',
#  'tags': [{'count': 1, 'value': 'kamen rider w'},
#           {'count': 1, 'value': 'related-akb48'}],
#  'type': 'Character'}
# {'_id': 192,
#  'aliases': [{'name': '女王', 'sort_name': '女王'}],
#  'area': 'United Kingdom',
#  'begin': {'date': 27, 'month': 6, 'year': 1970},
#  'ended': True,
#  'gid': '0383dadf-2a4e-4d10-a46a-e9e041da8eb3',
#  'name': 'Queen',
#  'rating': {'count': 24, 'value': 92},
#  'sort_name': 'Queen',
#  'tags': [{'count': 2, 'value': 'hard rock'},
#           {'count': 1, 'value': '70s'},
#           {'count': 1, 'value': 'queen family'},
#           {'count': 1, 'value': '90s'},
#           {'count': 1, 'value': '80s'},
#           {'count': 1, 'value': 'glam rock'},
#           {'count': 4, 'value': 'british'},
#           {'count': 1, 'value': 'english'},
#           {'count': 2, 'value': 'uk'},
#           {'count': 1, 'value': 'pop/rock'},
#           {'count': 1, 'value': 'pop-rock'},
#           {'count': 1, 'value': 'britannique'},
#           {'count': 1, 'value': 'classic pop and rock'},
#           {'count': 1, 'value': 'queen'},
#           {'count': 1, 'value': 'united kingdom'},
#           {'count': 1, 'value': 'langham 1 studio bbc'},
#           {'count': 1, 'value': 'kind of magic'},
#           {'count': 1, 'value': 'band'},
#           {'count': 6, 'value': 'rock'},
#           {'count': 1, 'value': 'platinum'}],
#  'type': 'Group'}
# {'_id': 992994,
#  'ended': True,
#  'gid': '5eecaf18-02ec-47af-a4f2-7831db373419',
#  'name': 'Queen',
#  'sort_name': 'Queen'}
# 