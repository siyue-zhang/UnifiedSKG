import re

# pred = "select ( select c5 from w where c2 = '\"die 'hard\"' ) - ( select c5 from w where c2 = '\"die hard\"' )"
# # pred = "select c5 from w where c2 = '\"hor'semen\"'"
# print(pred)

# pairs = re.findall(r'where (c[0-9]{1,}.{,20}?)\s*?[!=><]{1,}\s*?\'([^"]*?"[^"]*?\'[^"]*".*?)\'', pred)
# # select c5 from w where c2 = '"i'll be your fool tonight"'
# buf = []
# n = 0
# if len(pairs)>0:
#     for col, ori in pairs:
#         pred = pred.replace(f'\'{ori}\'', f'[X{n}]')
#         n += 1

# print(pred)

pred = "select c3 from w where c1 = 'american mcgee's crooked house'"
indices = []
for i, char in enumerate(pred):
    if char == "'":
        indices.append(i)
print(indices)
if len(indices) == 3:
    pred = pred[:indices[0]] + "\"" + pred[indices[0]+1:]
    pred = pred[:indices[2]] + "\"" + pred[indices[2]+1:]
print(pred)