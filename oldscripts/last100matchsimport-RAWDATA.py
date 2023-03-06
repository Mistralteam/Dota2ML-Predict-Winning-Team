import json
import pymongo
import requests

# Retrieve latest 1000 match IDs from API endpoint
api_url = "https://api.opendota.com/api/publicMatches"
response = requests.get(api_url, params={"mmr_descending": "true", "limit": "1000"})
matches_data = response.json()

# Connect to the MongoDB server
client = pymongo.MongoClient("mongodb://root:your_password@localhost:27017/")
db = client["dota2"]

# Insert match data into the matches collection
matches_collection = db["matches"]
for match in matches_data:
    match_id = match["match_id"]
    api_url = f"https://api.opendota.com/api/matches/{match_id}"
    response = requests.get(api_url)
    match_data = response.json()
    matches_collection.insert_one(match_data)
