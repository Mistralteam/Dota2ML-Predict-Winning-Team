import pymongo
import requests

# Connect to the MongoDB server
client = pymongo.MongoClient("mongodb://root:your_password@localhost:27017/")
db = client["dota2"]

# Set up the matches collection
matches_collection = db["matches"]

# Set up the heroes collection
heroes_collection = db["heroes"]

# Retrieve all heroes from the API and store them in the heroes collection
# api_url = "https://api.opendota.com/api/heroes"
# response = requests.get(api_url)
# heroes_data = response.json()
# heroes_collection.insert_many(heroes_data)


## Retrieve all heroes from the API and store them in the heroes collection
api_url = "https://api.opendota.com/api/heroes"
response = requests.get(api_url)
heroes_data = response.json()

# Add the hero ID to each hero
for hero in heroes_data:
    hero["id"] = hero["id"]

heroes_collection.insert_many(heroes_data)

# Define a function to get the localized name of a hero based on its ID
def get_hero_name(hero_id):
    hero = heroes_collection.find_one({"id": hero_id})
    if hero:
        return hero["localized_name"]
    else:
        return None


# Retrieve the latest 1000 matches from the API
api_url = "https://api.opendota.com/api/publicMatches"
params = {"mmr_descending": "true", "limit": "1000"}
response = requests.get(api_url, params=params)
match_data = response.json()

# Filter out any matches with null values for the features we're interested in
filtered_matches = []
for match in match_data:
    if all(match.get(feature) is not None for feature in ["radiant_win", "radiant_team", "dire_team", "duration", "avg_mmr", "num_mmr", "game_mode", "lobby_type", "cluster"]):
        filtered_match = {}
        filtered_match["match_id"] = match["match_id"]
        filtered_match["match_seq_num"] = match["match_seq_num"]
        filtered_match["radiant_win"] = match["radiant_win"]
        filtered_match["start_time"] = match["start_time"]
        filtered_match["duration"] = match["duration"]
        filtered_match["num_mmr"] = match["num_mmr"]
        filtered_match["lobby_type"] = match["lobby_type"]
        filtered_match["game_mode"] = match["game_mode"]
        filtered_match["avg_rank_tier"] = match["avg_rank_tier"]
        filtered_match["num_rank_tier"] = match["num_rank_tier"]
        filtered_match["cluster"] = match["cluster"]
        filtered_match["radiant_team"] = match["radiant_team"]
        filtered_match["dire_team"] = match["dire_team"]

        # Get the hero names for each team
        radiant_heroes = [get_hero_name(
            int(hero_id)) for hero_id in filtered_match["radiant_team"].split(",")]
        dire_heroes = [get_hero_name(int(hero_id))
                       for hero_id in filtered_match["dire_team"].split(",")]

        filtered_match["radiant_heroes"] = radiant_heroes
        filtered_match["dire_heroes"] = dire_heroes

        filtered_matches.append(filtered_match)

# Insert the filtered matches into the database
matches_collection.insert_many(filtered_matches)

# Set up the picks collection
picks_collection = db["picks"]

# Set up the players collection
players_collection = db["players"]

# Get picks data for each match
for match in filtered_matches:
    match_id = match["match_id"]
    api_url = f"https://api.opendota.com/api/matches/{match_id}"
    response = requests.get(api_url)
    match_data = response.json()

    if "picks_bans" not in match_data:
        continue

    radiant_picks = [pick["hero_id"] for pick in match_data["picks_bans"] if pick["is_pick"] and pick["team"] == 0]
    dire_picks = [pick["hero_id"] for pick in match_data["picks_bans"] if pick["is_pick"] and pick["team"] == 1]

    picks = {"match_id": match_id, "radiant_picks": radiant_picks, "dire_picks": dire_picks}
    picks_collection.insert_one(picks)

    # Get player data for each match
    players_data = match_data["players"]
    for player in players_data:
        player_id = player["account_id"]
        api_url = f"https://api.opendota.com/api/players/{player_id}"
        response = requests.get(api_url)
        player_data = response.json()
        if "competitive_rank" in player_data and player_data["competitive_rank"] is not None:
            mmr = player_data["competitive_rank"]
            players_collection.update_one(
                {"player_id": player_id},
                {"$set": {"last_mmr": mmr}},
                upsert=True
            )
        else:
            continue

# Extract the MMR of all players in the match and store it in the matches collection
radiant_mmr = []
dire_mmr = []
for player in players_data:
    player_id = player["account_id"]
    mmr = players_collection.find_one(
        {"player_id": player_id}, {"last_mmr": 1})
    if mmr:
        mmr = mmr["last_mmr"]
    else:
        mmr = None
    if player["player_slot"] < 5:
        radiant_mmr.append(mmr)
    else:
        dire_mmr.append(mmr)

filtered_match = matches_collection.find_one({"match_id": match_id})
filtered_match["radiant_mmr"] = radiant_mmr
filtered_match["dire_mmr"] = dire_mmr
matches_collection.replace_one({"match_id": match_id}, filtered_match)

print("Done!")
