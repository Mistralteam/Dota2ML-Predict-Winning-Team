import json
import mysql.connector

# Load match data from JSON file
with open('match_data.json', 'r') as f:
    match_data = json.load(f)

# Define MySQL connection and cursor
mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="your_password",
  database="dota2"
)
cursor = mydb.cursor()

# Create SQL table based on JSON schema
table_name = 'matchesnew'
create_table_query = f"CREATE TABLE {table_name} ("
for key, value in match_data.items():
    if isinstance(value, bool):
        column_type = 'BOOL'
    elif isinstance(value, int):
        column_type = 'INT'
    elif isinstance(value, float):
        column_type = 'FLOAT'
    else:
        column_type = 'VARCHAR(255)'
    create_table_query += f"{key} {column_type}, "
create_table_query = create_table_query[:-2] + ")"
cursor.execute(create_table_query)
mydb.commit()

# Close MySQL connection
mydb.close()
