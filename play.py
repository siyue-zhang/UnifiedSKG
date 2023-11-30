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
from copy import deepcopy
pred = " select ( select c6_number from w where c2 = 'india' ) - ( select c6_number from w where c2 = 'pakistan' )"
pairs = re.findall(r'where (c[0-9]{1,}.{,20}?)\s*?[!=><]{1,}\s*?\'(.*?".*?\'.*".*?)\'', pred)
pairs = re.finditer(r'where (c[0-9]{1,}.{,20}?)\s*?[!=><]{1,}\s*?\'(.{1,}?)\'', pred)
print(pred)
tokens = []
replacement = []
for idx, match in enumerate(pairs):
    start = match.start(0)
    end = match.end(0)
    col = pred[match.start(1):match.end(1)]
    ori = pred[match.start(2):match.end(2)]
    to_replace = pred[start:end]

    token = str(idx) + '_'*(end-start-len(str(idx)))
    tokens.append(token)
    pred = pred[:start] + token + pred[end:]
    to_replace = to_replace.replace(ori, 'pakistan')
    replacement.append(to_replace)
print(pred)

for i in range(len(tokens)):
    pred = pred.replace(tokens[i], replacement[i])
print(pred)

# for idx, match in enumerate(pairs_copy):
#     start = match.start(0)
#     end = match.end(0)
#     col = pred[match.start(1):match.end(1)]
#     ori = pred[match.start(2):match.end(2)]
#     to_replace = pred[start:end]
#     print(f'B: part to be replaced: {to_replace}, col: {col}, string: {ori}')
#     to_replace = to_replace.replace(ori, 'pakistan')
#     pred_copy = pred_copy.replace(str(idx) + '_'*(end-start-len(str(idx))), to_replace)
# print(pred_copy)

