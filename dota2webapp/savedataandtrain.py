from flask import Flask, jsonify, request
from pymongo import MongoClient
import pymongo
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV
import numpy as np
from bson.binary import Binary
import pickle

# Connect to the MongoDB server
client = pymongo.MongoClient("mongodb://root:your_password@localhost:27017/")

# Access the Dota 2 matches database
db = client["dota2"]

# Retrieve the hero names from the heroes collection
heroes_collection = db["heroes"]
hero_data = list(heroes_collection.find({}, {"localized_name": 1}))
hero_names = [hero["localized_name"] for hero in hero_data]

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

# Perform feature selection to reduce dimensionality
selector = SelectKBest(score_func=f_classif, k=20)
X_train_new = selector.fit_transform(X_train, y_train)
X_test_new = selector.transform(X_test)

# Perform grid search to find the best hyperparameters for the model
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto'],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
}

grid_search = GridSearchCV(estimator=SVC(random_state=42), param_grid=param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train_new, y_train)

# Train the model on the full training set using the best hyperparameters
clf = SVC(**grid_search.best_params_, probability=True, random_state=42)
clf.fit(X_train_new, y_train)

selector = SelectKBest(score_func=f_classif, k=20)
X_train_new = selector.fit_transform(X_train, y_train)
X_test_new = selector.transform(X_test)
selector_params = {'score_func': selector.score_func.__name__, 'k': selector.k}

# Save the model in MongoDB
model_bytes = pickle.dumps(clf)
model_document = {
    'model': Binary(model_bytes),
    'selector': selector_params,
    'hero_names': hero_names
}
db['models'].insert_one(model_document)

print("Model saved in MongoDB")

# Initialize
