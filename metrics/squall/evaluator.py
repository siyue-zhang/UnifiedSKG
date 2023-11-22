# encoding=utf8

from .evaluator_squall import Evaluator
import re
from fuzzywuzzy import process
import json

def find_best_match(contents, col, ori):
    strings = []
    for c in contents:
        for cc in c:
            if col == cc['col']:
                strings = cc['data']
                strings = [str(s) for s in strings]
                strings = list(set(strings))
    assert len(strings)>0
    best_match, _ = process.extractOne(ori, strings)
    return best_match


def fuzzy_replace(pred, table_id):
    table_path = f'.third_party/squall/tables/json/{table_id}.json'
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
            assert col in cols
            best_match = find_best_match(contents, col, ori)
            best_match = best_match.replace('\'','\'\'')
            pred = pred.replace(f'\'{ori}\'', f'[X{n}]')
            n += 1
            buf.append(best_match)

    pairs = re.findall(r'where (c[0-9]{1,}.{,20}?)\s*?[!=><]{1,}\s*?\'(.{1,}?)\'', pred)
    # print(pairs)
    if len(pairs)>0:
        for col, ori in pairs:
            if col not in cols:
                if 'and' in col:
                    col = col.split('and')[-1].strip()
                if 'or ' in col:
                    col = col.split('or')[-1].strip()
            assert col in cols
            best_match = find_best_match(contents, col, ori)
            pred = pred.replace(ori, best_match)
    
    pairs = re.findall(r'where (c[0-9]{1,}.{,20}?) in \(\s*?\'(.{1,}?)\'\s*?,\s*?\'(.{1,}?)\'\s*?\)', pred)
    # print(pairs)
    if len(pairs)>0:
        for col, ori1, ori2 in pairs:
            if col not in cols:
                if 'and' in col:
                    print()
                    col = col.split('and')[-1].strip()
                if 'or ' in col:
                    col = col.split('or')[-1].strip()
            assert col in cols
            for ori in [ori1, ori2]:
                best_match = find_best_match(contents, col, ori)
                pred = pred.replace(ori, best_match)

    pairs = re.findall(r'where (c[0-9]{1,}.{,20}?) in \(\s*?\'(.{1,}?)\'\s*?,\s*?\'(.{1,}?)\'\s*?, \'(.{1,}?)\'\s*?\)', pred)
    # print(pairs)
    if len(pairs)>0:
        for col, ori1, ori2, ori3 in pairs:
            assert col in cols
            for ori in [ori1, ori2, ori3]:
                best_match = find_best_match(contents, col, ori)
                pred = pred.replace(ori, best_match)

    for j in range(len(buf)):
        pred = pred.replace(f'[X{j}]', f'\'{buf[j]}\'')
    
    if pred != ori_pred:
        print('\nString is replaced by fuzzy match!')
        print(table_path)
        print(f'From: {ori_pred}')
        print(f'To  : {pred}')

    return pred

def postprocess_text(preds, labels, section, fuzzy):
    # preds and labels for all eval samples
    # prepare the prediction format for the wtq evaluator
    predictions = []
    for idex, (pred, label) in enumerate(zip(preds, labels)):
        table_id = section["tbl"][idex]
        nt_id = section["nt"][idex]
        header = section["header"][idex]
        nl = section["nl"][idex]
        # repalce the natural language header with c1, c2, ... headers
        for j, h in enumerate(header):
            pred=pred.replace(h, 'c'+str(j+1))
            label=label.replace(h, 'c'+str(j+1))
            
        if fuzzy:
            pred = fuzzy_replace(pred, table_id)

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
        print('---------', preds, '\n', golds, '\n', section)
        assert 1==2
        predictions = postprocess_text(preds, golds, section, self.args.seq2seq.postproc_fuzzy_string)
        total = len(golds)
        execution_accuracy = self.evaluator.evaluate(predictions)
        logical_form = 0
        for d in predictions:
            if d['result'][0]['sql'] == d['result'][0]['tgt']:
                logical_form += 1

        return {"logical_form": logical_form/total, 
                "execution_accuracy":execution_accuracy/total}
