# encoding=utf8

from .evaluator_squall import Evaluator
import re
from fuzzywuzzy import process
import json

def find_best_match(contents, col, ori):
    final_strings = []
    done = False
    for c in contents:
        for cc in c:
            if col == cc['col']:
                strings = cc['data']
                for item in strings:
                    if isinstance(item, list):
                        for ii in item:
                            final_strings.append(str(ii))
                    else:
                        final_strings.append(str(item))
                done = True
            if done:
                break
        if done:
            break
    assert len(final_strings)>0, f'strings empty {final_strings}'
    final_strings = list(set(final_strings))
    best_match, _ = process.extractOne(ori, final_strings)
    return best_match

def find_fuzzy_col(col, mapping):
    assert col not in mapping
    # col->ori
    mapping_b = {value: key for key, value in mapping.items()}
    match = re.match(r'^(c\d+)', col)
    if match:
        c_num = match.group(1)
        # assert c_num in mapping, f'{c_num} not in {mapping}'
        if c_num not in mapping:
            print(f'predicted {c_num} is not valid ({mapping})')
            return mapping.keys()[0]
        else:
            best_match, _ = process.extractOne(col.replace(c_num, mapping[c_num]), [value for _, value in mapping.items()])
    else:
        best_match, _ = process.extractOne(col, [value for _, value in mapping.items()])
    return mapping_b[best_match]

def fuzzy_replace(pred, table_id, mapping):
    exception_keywords = ['from', 'w', 'select', 'where', 'limit', 'order']
    exception_keywords += [str(i) for i in range(10)]
    
    table_path = f'./third_party/squall/tables/json/{table_id}.json'
    with open(table_path, 'r') as file:
        contents = json.load(file)
    contents = contents["contents"]
    ori_pred = str(pred)

    cols = []
    for c in contents:
        for cc in c:
            cols.append(cc['col'])

    pairs = re.findall(r'where (c[0-9]{1,}.{,20}?)\s*?[!=><]{1,}\s*?\'(.*?".*?\'.*".*?)\'', pred)
    # select c5 from w where c2 = '"i'll be your fool tonight"'
    # print(pairs)
    buf = []
    n = 0
    if len(pairs)>0:
        for col, ori in pairs:
            if col not in cols:
                if 'and' in col:
                    col = col.split('and')[-1].strip()
                if 'or ' in col:
                    col = col.split('or')[-1].strip()
            if col not in cols:
                print(f'A: {col} not in {cols}, query ({pred})')
                col_replace = find_fuzzy_col(col, mapping)
                pred = pred.replace(col, col_replace)
                print(f' {col}-->{col_replace}')
                col = col_replace
            best_match = find_best_match(contents, col, ori)
            best_match = best_match.replace('\'','\'\'')
            pred = pred.replace(f'\'{ori}\'', f'[X{n}]')
            n += 1
            buf.append(best_match)

    pairs = re.findall(r'where (c[0-9]{1,}.{,20}?)\s*?[!=><]{1,}\s*?\'(.{1,}?)\'', pred)
    # print(pairs,'ppppp')
    if len(pairs)>0:
        for col, ori in pairs:
            if col not in cols:
                if 'and' in col:
                    col = col.split('and')[-1].strip()
                if 'or ' in col:
                    col = col.split('or')[-1].strip()
            if col not in cols:
                print(f'B: {col} not in {cols}, query ({pred})')
                col_replace = find_fuzzy_col(col, mapping)
                pred = pred.replace(col, col_replace)
                print(f' {col}-->{col_replace}')
                col = col_replace
            best_match = find_best_match(contents, col, ori)
            if best_match not in exception_keywords:
                pred = pred.replace(ori, best_match)
    
    pairs = re.findall(r'where (c[0-9]{1,}.{,20}?) in \(\s*?\'(.{1,}?)\'\s*?,\s*?\'(.{1,}?)\'\s*?\)', pred)
    # print(pairs, 'ttttt')
    if len(pairs)>0:
        for col, ori1, ori2 in pairs:
            if col not in cols:
                if 'and' in col:
                    print()
                    col = col.split('and')[-1].strip()
                if 'or ' in col:
                    col = col.split('or')[-1].strip()
            if col not in cols:
                print(f'C: {col} not in {cols}, query ({pred})')
                col_replace = find_fuzzy_col(col, mapping)
                pred = pred.replace(col, col_replace)
                print(f' {col}-->{col_replace}')
                col = col_replace
            for ori in [ori1, ori2]:
                best_match = find_best_match(contents, col, ori)
                if best_match not in exception_keywords:
                    pred = pred.replace(ori, best_match)

    pairs = re.findall(r'where (c[0-9]{1,}.{,20}?) in \(\s*?\'(.{1,}?)\'\s*?,\s*?\'(.{1,}?)\'\s*?, \'(.{1,}?)\'\s*?\)', pred)
    # print(pairs)
    if len(pairs)>0:
        for col, ori1, ori2, ori3 in pairs:
            if col not in cols:
                print(f'D: {col} not in {cols}, query ({pred})')
                col_replace = find_fuzzy_col(col, mapping)
                pred = pred.replace(col, col_replace)
                print(f' {col}-->{col_replace}')
                col = col_replace
            for ori in [ori1, ori2, ori3]:
                best_match = find_best_match(contents, col, ori)
                if best_match not in exception_keywords:
                    pred = pred.replace(ori, best_match)

    for j in range(len(buf)):
        pred = pred.replace(f'[X{j}]', f'\'{buf[j]}\'')
    
    if pred != ori_pred:
        print('\nString is replaced by fuzzy match!')
        print(table_path)
        print(f'From: {ori_pred}')
        print(f'To  : {pred}')

    return pred

def postprocess_text(preds, golds, section, fuzzy):

    # preds and labels for all eval samples
    # prepare the prediction format for the wtq evaluator
    predictions = []
    for pred, gold in zip(preds, golds):
        table_id = gold['db_id']
        nt_id = gold['id']
        column_name = gold['db_column_names']['column_name'][1:]
        ori_column_name = gold['db_column_names']['ori_column_name'][1:]
        nl = gold['question']
        label = gold['query']
        # repalce the natural language header with c1, c2, ... headers
        for j, h in enumerate(column_name):
            pred=pred.replace(h, ori_column_name[j])
            label=label.replace(h, ori_column_name[j])
            
        if fuzzy:
            mapping = {ori: col for ori, col in zip(ori_column_name, column_name)}
            pred = fuzzy_replace(pred, table_id, mapping)

        result_dict = {"sql": pred, "id": nt_id, "tgt": label}
        res = {"table_id": table_id, "result": [result_dict], 'nl': nl}
        predictions.append(res)
    return predictions


class EvaluateTool(object):
    def __init__(self, args):
        self.args = args
        self.evaluator = Evaluator(
        f"./third_party/squall/tables/tagged/",
        f"./third_party/squall/tables/db/",
)

    def evaluate(self, preds, golds, section):
        total = len(golds)
        predictions = postprocess_text(preds, golds, section, self.args.seq2seq.postproc_fuzzy_string)
        execution_accuracy = self.evaluator.evaluate(predictions)

        if section=='test':
            return {"execution_accuracy":execution_accuracy/total}
        else:
            logical_form = 0
            for d in predictions:
                if d['result'][0]['sql'] == d['result'][0]['tgt']:
                    logical_form += 1
            return {"logical_form": logical_form/total, 
                    "execution_accuracy":execution_accuracy/total}
