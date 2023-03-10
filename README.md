# Dota2ML-Predict-Winning-Team

## Introduction
Dota 2 basic ML to select a winning team based on heroes. A very basic model was used for ML learning.

## Scripts
The following scripts are included:
- `GETCLEANDATA-heropicksmodel.py`: captures 100 items from the OpenDota API.
- `RFC or SVMprediction.py`: predicts a winner based on the heroes you enter and from the data in your MongoDB. this has changed. sorry.

## Getting Started
Before running the scripts, you will need to:
1. Run the Docker Compose file to allow it to spin up a MongoDB.
2. Create a database called `dota2`. Once you have done this, the script to collect data will automatically create the tables/collections required.
3, I have put all the old scripts in oldscript folder, I was orginally using MySQL and abandoned that, so ignore anything you see with mysql. 


## Example response on the web app
SVM predicted probability of radiant win: 0.16641761794833765
Logistic Regression predicted probability of radiant win: 0.0011618670432271027
Random Forest predicted probability of radiant win: 0.24358752972403383
Neural Network predicted probability of radiant win: 0.0019183646725138119

Predicted winner: Dire

## Future Plans
Future plans for this code include:
- Creating a basic web front-end allowing people to pick the heroes and work out best hero matchups or predictions on who would win.
- Adding a range of features, such as but not limited to MMR and Gold Difference, for the model to learn and get a much better understanding of how the game works.

Any feedback or ideas for changes are welcome.
