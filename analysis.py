import json
import pandas as pd
import numpy as np
from matplotlib import pyplot

def load_json(path):
    with open(path, 'r') as file:
        data = json.load(file)
    return data

def json_to_df(keys, json_data):
    df = []
    for item in json_data:
        ex = {k:item[k] for k in item if k in keys}
        df.append(ex)
    df = pd.DataFrame(df)
    df = df[[k for k in keys if k in df]]
    return df

def add_flag(df, path_flag):
    with open(path_flag, 'r') as file:
        data = json.load(file)
    keys = list(data[0].keys())
    for k in keys:
        col = [d[k] for d in data]
        if isinstance(col[0],bool):
            col = [int(cell) for cell in col]
        df[k] = col
    if 'correct' in keys:
        print(f'Mean accuracy: {np.mean(df["correct"]):.4}')
    return df

if __name__=='__main__':
    # file = '/scratch/sz4651/Projects/UnifiedSKG/output/Omnitab_large_finetune_squall_tableqa/predictions_predict.json'
    # file = '/scratch/sz4651/Projects/UnifiedSKG/output/Omnitab_large_finetune_squall_tableqa2/predictions_eval.json'
    # file = '/scratch/sz4651/Projects/UnifiedSKG/output/T5_large_finetune_squall/predictions_eval_12.850310008857395.json'
    # file = '/scratch/sz4651/Projects/UnifiedSKG/output/Omnitab_large_finetune_squall_tableqa2/predictions_eval_36.16976127320955.json'
    file = '/scratch/sz4651/Projects/UnifiedSKG/output/T5_large_finetune_squall2/predictions_eval.json'
    flag_file = file[:-5]+'_flag.json'

    keys = ['id', 'question', 'prediction', 'label' ,'query', 'converted_query', 'seq_in']
    df = json_to_df(keys, load_json(file))
    df = add_flag(df, flag_file)
    print(df.head())
    df.to_csv('./t5_squall_text2sql_dev.csv')
    # df.to_csv('./omnitab_squall_tableqa_dev.csv')


