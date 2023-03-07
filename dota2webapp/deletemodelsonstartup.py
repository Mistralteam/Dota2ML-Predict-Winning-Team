import pymongo

# Set up a MongoClient and connect to the "dota2" database
# client = pymongo.MongoClient("mongodb://root:your_password@localhost:27017/")
client = pymongo.MongoClient("mongodb://root:your_password@db:27017/")
db = client["dota2"]

# Get the "models" collection
models = db["models"]

# Delete all documents in the collection
models.delete_many({})

# Print a message indicating how many documents were deleted
print(f"{models.count_documents({})} documents were deleted from the 'models' collection.")
