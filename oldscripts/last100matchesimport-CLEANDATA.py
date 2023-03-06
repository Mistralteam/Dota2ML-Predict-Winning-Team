import json
import pymongo
import requests

# Connect to the MongoDB server
client = pymongo.MongoClient("mongodb://root:your_password@localhost:27017/")
db = client["dota2"]

# Set up the matches collection
matches_collection = db["matches"]

# Retrieve the latest 1000 matches from the API
api_url = "https://api.opendota.com/api/publicMatches"
params = {"mmr_descending": "true", "limit": "1000"}
response = requests.get(api_url, params=params)
match_data = response.json()

# Filter out any matches with null values for the features we're interested in
filtered_matches = []
for match in match_data:
    if all(match.get(feature) is not None for feature in ["radiant_win", "radiant_team", "dire_team", "duration", "avg_mmr", "num_mmr", "game_mode", "lobby_type", "cluster"]):
        filtered_matches.append(match)

# Insert the filtered matches into the database
matches_collection.insert_many(filtered_matches)
