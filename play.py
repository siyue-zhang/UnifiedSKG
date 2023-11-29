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

# import sqlite3

# db_path = './data/downloads/extracted/7c4615a9b6581e8663546188e8a9a3359389ffbceff14219f538827f9041f75c/spider/database/farm/farm.sqlite'
# query = 'SELECT Official_Name FROM city WHERE City_ID NOT IN (SELECT Host_city_ID FROM farm_competition)'

# # Connect to the SQLite database
# conn = sqlite3.connect(db_path)
# cursor = conn.cursor()

# try:
#     # Execute the SQL query
#     cursor.execute(query)

#     # Fetch all the results
#     results = cursor.fetchall()

#     # Print the results
#     for row in results:
#         print(row[0])

# finally:
#     # Close the database connection

import re
pred = "select count ( c3 ) from w where c4_list = '7\" vinyl'"
ans = re.finditer(r'where (c[0-9]{1,}.{,20}?)\s*?[!=><]{1,}\s*?\'(.{1,}?)\'', pred)
for x in ans:
    s = x.start(0)
    e = x.end(0)
    print(pred[s:e])
    s = x.start(1)
    e = x.end(1)
    print(pred[s:e])
    s = x.start(2)
    e = x.end(2)
    print(pred[s:e])
