import nntplib
from flask import Flask, jsonify, request
from pymongo import MongoClient
import pymongo
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_selection import f_classif
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import make_scorer, f1_score
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_selection import chi2
from sklearn.pipeline import Pipeline
import numpy as np
from bson.binary import Binary
import pickle
import requests

# Connect to the MongoDB server
# client = pymongo.MongoClient("mongodb://root:your_password@localhost:27017/")
client = pymongo.MongoClient("mongodb://root:your_password@db:27017/")

# Access the Dota 2 matches database
db = client["dota2"]
models_collection = db["models"]

response = requests.get('https://api.opendota.com/api/heroes')
hero_data = response.json()
hero_names = [hero["localized_name"] for hero in hero_data]

# Store the hero names in the models collection
models_collection = db["models"]
models_collection.update_one({}, {'$set': {'hero_names': hero_names}}, upsert=True)


# # Retrieve the hero names from the heroes collection
# heroes_collection = db["heroes"]
# hero_data = list(heroes_collection.find({}, {"localized_name": 1}))
# hero_names = [hero["localized_name"] for hero in hero_data]
# Retrieve the hero names from OpenDota API


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

# def f_classif_wrapper(X, y):
#     return f_classif(X, y)

# Define the pipeline for feature selection and model training
selector = SelectKBest(score_func=chi2, k=10)
svm = SVC(probability=True, random_state=42)
logistic = LogisticRegression(random_state=42)
rf = RandomForestClassifier(random_state=42)
nn = MLPClassifier(random_state=42)


pipeline_svm = Pipeline(steps=[('selector', selector), ('clf', svm)])
pipeline_logistic = Pipeline(steps=[('selector', selector), ('clf', logistic)])
pipeline_rf = Pipeline(steps=[('selector', selector), ('clf', rf)])
pipeline_nn = Pipeline(steps=[('selector', selector), ('clf', nn)])


# Define the hyperparameter grids for each model


svm_param_grid = {
    'selector__k': [10, 20, 30],
    'clf__C': [0.1, 1, 10],
    'clf__gamma': ['scale', 'auto'],
    'clf__kernel': ['linear', 'poly', 'rbf', 'sigmoid']
}

logreg_param_grid = {
    'selector__k': [10, 20, 30],
    'clf__penalty': ['l1', 'l2'],
    'clf__C': [0.1, 1, 10],
    'clf__solver': ['liblinear']
}

rf_param_grid = {
    'selector__k': [10, 20, 30],
    'clf__n_estimators': [100, 200, 300],
    'clf__max_depth': [5, 10, 15]
}

nn_param_grid = {
    'selector__k': [10, 20, 30],
    'clf__hidden_layer_sizes': [(100,), (100, 50), (200, 100, 50)],
    'clf__activation': ['relu', 'logistic'],
    'clf__solver': ['adam'],
    'clf__alpha': [0.0001, 0.001, 0.01]
}

logistic_param_grid = {
    'selector__k': [10, 20, 30, 40, 50],
    'clf__penalty': ['l1', 'l2'],
    'clf__C': [0.1, 1, 10, 100],
    'clf__solver': ['liblinear']
}

# Perform grid search to find the best hyperparameters for each model
grid_search_svm = GridSearchCV(estimator=pipeline_svm, param_grid=svm_param_grid, cv=5, n_jobs=-1, scoring='f1_macro')
grid_search_svm.fit(X_train, y_train)
svm_model = grid_search_svm.best_estimator_

grid_search_logreg = GridSearchCV(estimator=pipeline_logistic, param_grid=logistic_param_grid, cv=5, n_jobs=-1, scoring='f1_macro')
grid_search_logreg.fit(X_train, y_train)
logistic_model = grid_search_logreg.best_estimator_

grid_search_rf = GridSearchCV(estimator=pipeline_rf, param_grid=rf_param_grid, cv=5, n_jobs=-1, scoring='f1_macro')
grid_search_rf.fit(X_train, y_train)
rf_model = grid_search_rf.best_estimator_

grid_search_nn = GridSearchCV(estimator=pipeline_nn, param_grid=nn_param_grid, cv=5, n_jobs=-1, scoring='f1_macro')
grid_search_nn.fit(X_train, y_train)
nn_model = grid_search_nn.best_estimator_

# Print the best hyperparameters for each model
print("SVM best hyperparameters: {}".format(grid_search_svm.best_params_))
print("Logistic Regression best hyperparameters: {}".format(grid_search_logreg.best_params_))
print("Random Forest best hyperparameters: {}".format(grid_search_rf.best_params_))
print("Neural Network best hyperparameters: {}".format(grid_search_nn.best_params_))

# Print the best F1 score for each model
print("SVM best F1 score: {}".format(grid_search_svm.best_score_))
print("Logistic Regression best F1 score: {}".format(grid_search_logreg.best_score_))
print("Random Forest best F1 score: {}".format(grid_search_rf.best_score_))
print("Neural Network best F1 score: {}".format(grid_search_nn.best_score_))

# Print the F1 score on the test set for each model
print("SVM test F1 score: {}".format(f1_score(y_test, svm_model.predict(X_test), average='macro')))
print("Logistic Regression test F1 score: {}".format(f1_score(y_test, logistic_model.predict(X_test), average='macro')))
print("Random Forest test F1 score: {}".format(f1_score(y_test, rf_model.predict(X_test), average='macro')))
print("Neural Network test F1 score: {}".format(f1_score(y_test, nn_model.predict(X_test), average='macro')))


# Save the models in MongoDB
# selector_params = selector.get_params()
selector_params = {
    'k': selector.k,
    'score_func': str(selector.score_func.__name__)
}

svm_document = {
    'model_type': 'svm',
    'model': Binary(pickle.dumps(svm_model)),
    'selector': selector_params,
    'clf': svm_model.named_steps['clf'].get_params(),
    'hero_names': hero_names
}

logistic_document = {
    'model_type': 'logistic_regression',
    'model': Binary(pickle.dumps(logistic_model)),
    'selector': selector_params,
    'clf': logistic_model.named_steps['clf'].get_params(),
    'hero_names': hero_names
}

rf_document = {
    'model_type': 'random_forest',
    'model': Binary(pickle.dumps(rf_model)),
    'selector': selector_params,
    'clf': rf_model.named_steps['clf'].get_params(),
    'hero_names': hero_names
}

nn_document = {
    'model_type': 'neural_network',
    'model': Binary(pickle.dumps(nn_model)),
    'selector': selector_params,
    'clf': nn_model.named_steps['clf'].get_params(),
    'hero_names': hero_names
}


models_collection.insert_one(svm_document)
models_collection.insert_one(logistic_document)
models_collection.insert_one(rf_document)
models_collection.insert_one(nn_document)



print('Models saved to MongoDB')


# Define the sample match
radiant_picks = ['Shadow Fiend', 'Lina', 'Crystal Maiden', 'Pudge', 'Clockwerk']
dire_picks = ['Phantom Assassin', 'Drow Ranger', 'Axe', 'Omniknight', 'Lion']

# Convert the hero names to one-hot encoded vectors
radiant_picks_vector = [1 if hero in radiant_picks else 0 for hero in hero_names]
dire_picks_vector = [1 if hero in dire_picks else 0 for hero in hero_names]

# Concatenate the vectors to create the feature vector for this match
feature_vector = radiant_picks_vector + dire_picks_vector

# Load the models from MongoDB
svm_model = pickle.loads(db['models'].find_one({'model_type': 'svm'})['model'])
logistic_model = pickle.loads(db['models'].find_one({'model_type': 'logistic_regression'})['model'])
rf_model = pickle.loads(db['models'].find_one({'model_type': 'random_forest'})['model'])
nn_model = pickle.loads(db['models'].find_one({'model_type': 'neural_network'})['model'])

# Get the predicted probabilities for each model
svm_prob = svm_model.predict_proba([feature_vector])[0][1]
logistic_prob = logistic_model.predict_proba([feature_vector])[0][1]
rf_prob = rf_model.predict_proba([feature_vector])[0][1]
nn_prob = nn_model.predict_proba([feature_vector])[0][1]

# Print the predicted probabilities for each model
print("SVM predicted probability of radiant win: {}".format(svm_prob))
print("Logistic Regression predicted probability of radiant win: {}".format(logistic_prob))
print("Random Forest predicted probability of radiant win: {}".format(rf_prob))
print("Neural Network predicted probability of radiant win: {}".format(nn_prob))

