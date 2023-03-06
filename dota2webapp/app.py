
from flask import Flask, request, render_template, jsonify
from pymongo import MongoClient
import pymongo
from bson.binary import Binary
from sklearn.feature_selection import SelectKBest, f_classif
import pickle
import numpy as np
from sklearn.model_selection import train_test_split


app = Flask(__name__)

# Connect to the MongoDB server
client = pymongo.MongoClient("mongodb://root:your_password@localhost:27017/")

# Access the Dota 2 matches database
db = client["dota2"]

# Retrieve the hero names from the models collection
models_collection = db["models"]
model_document = models_collection.find_one()
hero_names = model_document['hero_names']

# Load the model from MongoDB
model_document = db['models'].find_one()
clf = pickle.loads(model_document['model'])
selector_params = model_document['selector']
selector = SelectKBest(score_func=globals()[selector_params['score_func']], k=selector_params['k'])

# Retrieve the match data from the matches collection
matches_collection = db["matches"]
matches_cursor = matches_collection.find()
X = []
y = []

# Extract the feature vectors and labels from the matches
for match in matches_cursor:
    radiant_picks = match.get("radiant_heroes", [])[:5]
    dire_picks = match.get("dire_heroes", [])[:5]

    # Convert the hero names to one-hot encoded vectors
    radiant_picks_vector = [1 if hero in radiant_picks else 0 for hero in hero_names]
    dire_picks_vector = [1 if hero in dire_picks else 0 for hero in hero_names]

    # Concatenate the vectors to create the feature vector for this match
    feature_vector = radiant_picks_vector + dire_picks_vector

    # Add the feature vector and label to the X and y lists
    label = match.get("radiant_win", False)
    X.append(feature_vector)
    y.append(label)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

@app.route('/')
def index():
    return render_template('index.html', hero_names=hero_names)

@app.route('/predict', methods=['POST'])
def predict():
    radiant_picks = request.form.getlist('radiant_picks')
    dire_picks = request.form.getlist('dire_picks')

    # Convert the hero names to one-hot encoded vectors
    radiant_picks_vector = [1 if hero in radiant_picks else 0 for hero in hero_names]
    dire_picks_vector = [1 if hero in dire_picks else 0 for hero in hero_names]

    # Concatenate the vectors to create the feature vector for this match
    feature_vector = radiant_picks_vector + dire_picks_vector

    # Fit the feature selector on the training data
    selector.fit(X_train, y_train)

    # Perform feature selection on the feature vector
    feature_vector_new = selector.transform([feature_vector])

    # Predict the winner
    prediction = clf.predict(feature_vector_new)
    if prediction[0] == 1:
        return "Radiant wins!"
    else:
        return "Dire wins!"


if __name__ == '__main__':
    app.run(debug=True)
