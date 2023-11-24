# from utils.tokenization_tapex import TapexTokenizer

# tokenizer = TapexTokenizer.from_pretrained('neulab/omnitab-large')
# ext = tokenizer(
#                 answer='abd sss',
#                 padding="max_length",
#                 truncation=True,
#                 max_length=1024,
#             )
# print(ext)
# print(tokenizer.decode(ext))

import sqlite3

db_path = './data/downloads/extracted/7c4615a9b6581e8663546188e8a9a3359389ffbceff14219f538827f9041f75c/spider/database/farm/farm.sqlite'
query = 'SELECT Official_Name FROM city WHERE City_ID NOT IN (SELECT Host_city_ID FROM farm_competition)'

# Connect to the SQLite database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

try:
    # Execute the SQL query
    cursor.execute(query)

    # Fetch all the results
    results = cursor.fetchall()

    # Print the results
    for row in results:
        print(row[0])

finally:
    # Close the database connection
    conn.close()