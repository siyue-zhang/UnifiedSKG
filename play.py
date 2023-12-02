import re

pred = "select count ( * ) from w where c5 in ( 'barrington', 'farmington', 'rochester' )"
print(pred)
x = re.findall(r'where (c[0-9]{1,}.{,20}?) in \(\s*?\'(.{1,}?)\'\s*?,\s*?\'(.{1,}?)\'\s*?, \'(.{1,}?)\'\s*?\)', pred)
print(x)
pairs = re.finditer(r'where (c[0-9]{1,}.{,20}?) in \(\s*?\'(.{1,}?)\'\s*?,\s*?\'(.{1,}?)\'\s*?, \'(.{1,}?)\'\s*?\)', pred)
tokens = []
replacement = []
for idx, match in enumerate(pairs):
    start = match.start(0)
    end = match.end(0)
    col = pred[match.start(1):match.end(1)]
    ori1 = pred[match.start(2):match.end(2)]
    ori2 = pred[match.start(3):match.end(3)]
    ori3 = pred[match.start(4):match.end(4)]
    print(col, ori1, ori2, ori3)
    to_replace = pred[start:end]
    print(to_replace)

    token = str(idx) + '_'*(end-start-len(str(idx)))
    tokens.append(token)
    pred = pred[:start] + token + pred[end:]


    for ori in [ori1, ori2, ori3]:
        best_match = 'xx'
        to_replace = to_replace.replace(ori, best_match)
    replacement.append(to_replace)

for i in range(len(tokens)):
    pred = pred.replace(tokens[i], replacement[i])

print(pred)