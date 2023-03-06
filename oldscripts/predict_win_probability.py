import pandas as pd
import pymongo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np


# Connect to the MongoDB server and retrieve the matches collection
client = pymongo.MongoClient("mongodb://root:your_password@localhost:27017/")
db = client["dota2"]
matches_collection = db["matches"]

# Create a dataframe from the matches collection
matches = list(matches_collection.find())
match_df = pd.DataFrame(matches)

# Create a new dataframe with the following columns:
# match_id, radiant_win, radiant_hero_1, radiant_hero_2, ..., dire_hero_1, dire_hero_2, ...
hero_df = pd.DataFrame()
hero_df["match_id"] = match_df["match_id"]
hero_df["radiant_win"] = match_df["radiant_win"]
for i in range(1, 6):
    hero_df[f"radiant_hero_{i}"] = match_df["radiant_heroes"].apply(lambda x: x[i-1] if len(x) >= i else None)
for i in range(1, 6):
    hero_df[f"dire_hero_{i}"] = match_df["dire_heroes"].apply(lambda x: x[i-1] if len(x) >= i else None)


train_df, test_df = train_test_split(hero_df, test_size=0.2, random_state=42)


hero_encoder = OneHotEncoder(handle_unknown="ignore")
hero_encoder.fit(hero_df[["radiant_hero_1", "radiant_hero_2", "radiant_hero_3", "radiant_hero_4", "radiant_hero_5", "dire_hero_1", "dire_hero_2", "dire_hero_3", "dire_hero_4", "dire_hero_5"]])
train_hero_features = hero_encoder.transform(train_df[["radiant_hero_1", "radiant_hero_2", "radiant_hero_3", "radiant_hero_4", "radiant_hero_5", "dire_hero_1", "dire_hero_2", "dire_hero_3", "dire_hero_4", "dire_hero_5"]])
test_hero_features = hero_encoder.transform(test_df[["radiant_hero_1", "radiant_hero_2", "radiant_hero_3", "radiant_hero_4", "radiant_hero_5", "dire_hero_1", "dire_hero_2", "dire_hero_3", "dire_hero_4", "dire_hero_5"]])


logreg = LogisticRegression(max_iter=10000)
logreg.fit(train_hero_features, train_df["radiant_win"])


rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(train_hero_features, train_df["radiant_win"])


logreg_predictions = logreg.predict(test_hero_features)
rf_predictions = rf.predict(test_hero_features)


print("Logistic Regression Accuracy:", accuracy_score(test_df["radiant_win"], logreg_predictions))
print("Logistic Regression Confusion Matrix:\n", confusion_matrix(test_df["radiant_win"], logreg_predictions))
print("Random Forest Accuracy:", accuracy_score(test_df["radiant_win"], rf_predictions))
print("Random Forest Confusion Matrix:\n", confusion_matrix(test_df["radiant_win"], rf_predictions))





