import json

with open('oldscripts/match_data.json') as f:
    match_data = json.load(f)

# Define a function to recursively traverse the data and generate the MongoDB structure
def generate_structure(data, structure):
    for key, value in data.items():
        if isinstance(value, dict):
            if key not in structure:
                structure[key] = {}
            generate_structure(value, structure[key])
        else:
            if key not in structure:
                structure[key] = None

# Generate the MongoDB structure
mongodb_structure = {}
generate_structure(match_data, mongodb_structure)

# Print the MongoDB structure
print(mongodb_structure)
