# import requests
# import mysql.connector

# # Set up MySQL connection
# mydb = mysql.connector.connect(
#   host="localhost",
#   user="root",
#   password="your_password",
#   database="dota2"
# )

# # Set up OpenDota API endpoint
# api_endpoint = "https://api.opendota.com/api/matches/7045558573"

# # Get data from OpenDota API
# response = requests.get(api_endpoint)
# data = response.json()

# # Parse data and insert into MySQL database
# for match in match_data:
#     cursor = mydb.cursor()
#     sql = f"INSERT INTO {table_name} ({','.join(match.keys())}) VALUES ({','.join(['%s']*len(match.keys()))})"
#     val = tuple(match.values())
#     cursor.execute(sql, val)

# mydb.commit()
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
for key, value in match_data[0].items():
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

# Insert data into SQL table
for match in match_data:
    sql = f"INSERT INTO {table_name} ("
    values = "VALUES ("
    for key, value in match.items():
        sql += f"{key}, "
        if isinstance(value, str):
            values += f"'{value}', "
        else:
            values += f"{value}, "
    sql = sql[:-2] + ") "
    values = values[:-2] + ")"
    cursor.execute(sql + values)
mydb.commit()

# Close MySQL connection
mydb.close()
