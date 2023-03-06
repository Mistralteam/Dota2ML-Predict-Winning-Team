import json
import pymongo
import requests

# Retrieve match data from API endpoint
match_id = 7045558573
api_url = f"https://api.opendota.com/api/matches/{match_id}"
response = requests.get(api_url)
match_data = response.json()

# Connect to the MongoDB server
client = pymongo.MongoClient("mongodb://root:your_password@localhost:27017/")
db = client["dota2"]

# Insert match data into the matches collection
matches_collection = db["matches"]
matches_collection.insert_one(match_data)
