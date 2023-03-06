# Dota2ML-Predict-Winning-Team
Dota 2 basic ML to select a winning team based on heros. 
Very Basic model, was used for ML learning. 

script GETCLEANDATA-heropicksmodel.py will capture 100 items from the opendota API 
script RFC or SVMprediction.py will predict a winner based on the heros you enter and from the data in your mongoDB. 

First you will need to run the docker-compose file to allow it to spin up a MongoDB. 
you will then need to create a db called dota2, one your have done this the script to collect data will automatically create the tables/collections required.


Future plans with the code, 
Possibly create a basic web front end allowing people to pick the heros and work out best heros match ups or predictions on who would win, 

To do this we would need to add a range of features not just heros, Such as but not limited to MMR and Gold Difference for the model to learn and get a much better understanding of how the game works. 

Anyway, look forward to hearing back feedback and feel free to post your ideas or changes. 
