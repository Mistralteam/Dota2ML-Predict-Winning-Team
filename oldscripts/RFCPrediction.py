from pymongo import MongoClient
import pymongo
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV
from bson.binary import Binary

import pickle
import numpy as np

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
    X.append(feature_vector)
    y.append(match["radiant_win"])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Perform feature selection to reduce dimensionality
selector = SelectKBest(score_func=f_classif, k=20)
X_train_new = selector.fit_transform(X_train, y_train)
X_test_new = selector.transform(X_test)

# Perform grid search to find the best hyperparameters for the model
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
}

grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train_new, y_train)

# Print the best hyperparameters found by the grid search
print("Best hyperparameters:", grid_search.best_params_)

# Train the model on the full training set using the best hyperparameters
clf = RandomForestClassifier(**grid_search.best_params_, random_state=42)
clf.fit(X_train_new, y_train)

# Test the model on the testing data
y_pred = clf.predict(X_test_new)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Use k-fold cross-validation to evaluate the model's performance
# Use k-fold cross-validation to evaluate the model's performance
kf = KFold(n_splits=5, shuffle=True, random_state=42)
accuracies = []
for train_idx, test_idx in kf.split(X):
    X_train, X_test = np.array(X)[train_idx], np.array(X)[test_idx]
    y_train, y_test = np.array(y)[train_idx], np.array(y)[test_idx]
    X_train_new = selector.fit_transform(X_train, y_train)
    X_test_new = selector.transform(X_test)

    clf = RandomForestClassifier(**grid_search.best_params_, random_state=42)
    clf.fit(X_train_new, y_train)

    y_pred = clf.predict(X_test_new)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

mean_accuracy = np.mean(accuracies)
print("Mean accuracy:", mean_accuracy)

# Convert the trained model to bytes
model_bytes = pickle.dumps(clf)

# Store the model bytes in the database
model_collection = db["models"]
model_collection.insert_one({"name": "Random Forest Classifier", "model": Binary(model_bytes)})

# Retrieve the trained model from the database
model_document = model_collection.find_one({"name": "Random Forest Classifier"})
model_bytes = model_document["model"]
clf = pickle.loads(model_bytes)

# Predict the winner for a new match
radiant_picks = input("Enter the Radiant picks separated by commas: ").split(",")
dire_picks = input("Enter the Dire picks separated by commas: ").split(",")

# Select only the first 5 picks for each team
radiant_picks = radiant_picks[:5]
dire_picks = dire_picks[:5]

# Convert the hero names to one-hot encoded vectors
radiant_picks_vector = [1 if hero in radiant_picks else 0 for hero in hero_names]
dire_picks_vector = [1 if hero in dire_picks else 0 for hero in hero_names]

# Concatenate the vectors to create the feature vector for this match
feature_vector = radiant_picks_vector + dire_picks_vector

# Perform feature selection on the feature vector
feature_vector_new = selector.transform([feature_vector])

# Predict the winner
prediction = clf.predict(feature_vector_new)
if prediction[0] == 1:
    print("Radiant wins!")
else:
    print("Dire wins!")
