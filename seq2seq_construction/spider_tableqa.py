import os
import torch
import random
import re
from copy import deepcopy
from typing import List, Dict

from datasets.dataset_dict import DatasetDict
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from transformers import AutoTokenizer

from third_party.miscs.bridge_content_encoder import get_database_matches

from tqdm import tqdm
import sqlite3
import pandas as pd

from utils.processor import get_default_processor

"""
This part of seq2seq construction of spider dataset was partly borrowed from PICARD model.
https://github.com/ElementAI/picard

And we followed their configuration of normalization and serialization.
their configuration is as followed:
{
    "source_prefix": "",
    "schema_serialization_type": "peteshaw",
    "schema_serialization_randomized": false,
    "schema_serialization_with_db_id": true,
    "schema_serialization_with_db_content": true,
    "normalize_query": true,
    "target_with_db_id": true,
}
"""


def execute_query(db_path, query):
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()
    
    print('executing query ...')
    print(query)
    cursor.execute(query)
    result = cursor.fetchall()
    ret = []
    if result:
        for row in result:
            for cell in row:
                ret.append(str(cell))
    connection.close()
    return ret


def fetch_table_data(connection, table_name):
    cursor = connection.execute(f"SELECT * FROM {table_name}")
    header = [col[0] for col in cursor.description]
    rows = [list(map(str, row)) for row in cursor.fetchall()]
    return {'header': header, 'rows': rows}

def read_sqlite_database(db_path):
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row  # Use Row factory to get rows as dictionaries

    # Get a list of all tables in the database
    print('reading database ...')
    cursor = connection.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [table[0] for table in cursor.fetchall()]

    # Fetch data for each table
    database_dict = {}
    for table in tables:
        database_dict[table] = fetch_table_data(connection, table)

    connection.close()
    return database_dict

"""
    Wrap the raw dataset into the seq2seq one.
    And the raw dataset item is formatted as
    {
        "query": sample["query"],
        "question": sample["question"],
        "db_id": db_id,
        "db_path": db_path,
        "db_table_names": schema["table_names_original"],
        "db_column_names": [
            {"table_id": table_id, "column_name": column_name}
            for table_id, column_name in schema["column_names_original"]
        ],
        "db_column_types": schema["column_types"],
        "db_primary_keys": [{"column_id": column_id} for column_id in schema["primary_keys"]],
        "db_foreign_keys": [
            {"column_id": column_id, "other_column_id": other_column_id}
            for column_id, other_column_id in schema["foreign_keys"]
        ],
    }
    """


class Constructor(object):
    def __init__(self, args):
        self.args = args

    def to_seq2seq(self, raw_datasets: DatasetDict, cache_root: str):
        if not len(raw_datasets) == 2:
            raise AssertionError("Train, Dev sections of dataset expected.")
        if getattr(self.args.seq2seq, "few_shot_rate"):
            raw_train = random.sample(list(raw_datasets["train"]), int(self.args.seq2seq.few_shot_rate * len(raw_datasets["train"])))
            train_dataset = TrainDataset(self.args, raw_train, cache_root)
        else:
            train_dataset = TrainDataset(self.args, raw_datasets["train"], cache_root)
        dev_dataset = DevDataset(self.args, raw_datasets["validation"], cache_root)

        return train_dataset, dev_dataset


class TrainDataset(Dataset):
    def __init__(self, args, raw_datasets, cache_root):
        self.args = args
        self.raw_datasets = raw_datasets
        self.db_contents = {}
        cache_path = os.path.join(cache_root, 'spider_train.cache')
        self.tokenizer = AutoTokenizer.from_pretrained(args.bert.location, use_fast=False)
        if os.path.exists(cache_path) and args.dataset.use_cache:
            self.extended_data = torch.load(cache_path)
        else:
            self.tab_processor = get_default_processor(max_cell_length=15,
                                                       tokenizer=AutoTokenizer.from_pretrained(args.bert.location, use_fast=False),
                                                       max_input_length=args.seq2seq.table_truncation_max_length)
            self.extended_data = []
            for raw_data in tqdm(self.raw_datasets):
                extend_data = deepcopy(raw_data)

                db_id = raw_data['db_id']
                db_path = raw_data['db_path']
                db_path = db_path + "/" + db_id + "/" + db_id + ".sqlite"
                question = raw_data["question"]
                query = raw_data["query"]

                if db_id not in self.db_contents:
                    database_dict = read_sqlite_database(db_path)
                    self.db_contents['db_id'] = database_dict
                else:
                    database_dict = self.db_contents['db_id']

                try:
                    gold_result = execute_query(db_path, query)
                except Exception as e:
                    print(f'error target query: {query} ; database: {db_id}')
                    continue

                if isinstance(gold_result, list) and len(gold_result)>10:
                    continue 

                n_max_rows = max([len(database_dict[x]['rows']) for x in database_dict])
                if n_max_rows>60 or len(database_dict.keys())>10:
                    continue

                table_contexts = []
                struct_in = ''
                for tbl in database_dict:
                    struct_in += f'tab : {tbl} '
                    table_context = deepcopy(database_dict[tbl])
                    print('processing table ...')
                    print('  gold result: ', gold_result)
                    print('  table: ', tbl, 'in', {x: len(database_dict[x]['rows']) for x in database_dict})
                    for truncate_func in self.tab_processor.table_truncate_funcs:
                        truncate_func.truncate_table(table_context, question, gold_result)
                    table_contexts.append(table_context)
                    # linearize a table into a string
                    linear_table = self.tab_processor.table_linearize_func.process_table(table_context)
                    struct_in += linear_table + ' '

                if len(gold_result)==0:
                    seq_out='None'
                else:
                    seq_out = self.tab_processor.process_output(gold_result)
                extend_data.update({"struct_in": struct_in.lower(),
                                    "text_in": question.lower(),
                                    "seq_out": seq_out.lower(),
                                    "table_context": table_contexts})
                self.extended_data.append(extend_data)

            if args.dataset.use_cache:
                torch.save(self.extended_data, cache_path)

    def __getitem__(self, index) -> T_co:
        return self.extended_data[index]

    def __len__(self):
        return len(self.extended_data)


class DevDataset(Dataset):
    def __init__(self, args, raw_datasets, cache_root):
        self.args = args
        self.raw_datasets = raw_datasets
        self.db_contents = {}
        cache_path = os.path.join(cache_root, 'spider_dev.cache')
        self.tokenizer = AutoTokenizer.from_pretrained(args.bert.location, use_fast=False)
        if os.path.exists(cache_path) and args.dataset.use_cache:
            self.extended_data = torch.load(cache_path)
        else:
            self.tab_processor = get_default_processor(max_cell_length=15,
                                                       tokenizer=AutoTokenizer.from_pretrained(args.bert.location, use_fast=False),
                                                       max_input_length=args.seq2seq.table_truncation_max_length)
            self.extended_data = []
            for raw_data in tqdm(self.raw_datasets):
                extend_data = deepcopy(raw_data)

                db_id = raw_data['db_id']
                db_path = raw_data['db_path']
                db_path = db_path + "/" + db_id + "/" + db_id + ".sqlite"
                question = raw_data["question"]
                query = raw_data["query"]
                
                if db_id not in self.db_contents:
                    database_dict = read_sqlite_database(db_path)
                    self.db_contents['db_id'] = database_dict
                else:
                    database_dict = self.db_contents['db_id']

                try:
                    gold_result = execute_query(db_path, query)
                except Exception as e:
                    print(f'error target query: {query} ; database: {db_id}')
                    continue

                if isinstance(gold_result, list) and len(gold_result)>10:
                    continue 

                n_max_rows = max([len(database_dict[x]['rows']) for x in database_dict])
                if n_max_rows>60 or len(database_dict.keys())>10:
                    continue

                table_contexts = []
                struct_in = ''
                for tbl in database_dict:
                    struct_in += f'tab : {tbl} '
                    table_context = deepcopy(database_dict[tbl])
                    for truncate_func in self.tab_processor.table_truncate_funcs:
                        truncate_func.truncate_table(table_context, question, gold_result)
                    table_contexts.append(table_context)
                    # linearize a table into a string
                    linear_table = self.tab_processor.table_linearize_func.process_table(table_context)
                    struct_in += linear_table + ' '
                
                if len(gold_result)==0:
                    seq_out='None'
                else:
                    seq_out = self.tab_processor.process_output(gold_result)
                extend_data.update({"struct_in": struct_in.lower(),
                                    "text_in": question.lower(),
                                    "seq_out": seq_out.lower(),
                                    "table_context": table_contexts})
                self.extended_data.append(extend_data)
                
            if args.dataset.use_cache:
                torch.save(self.extended_data, cache_path)

    def __getitem__(self, index) -> T_co:
        return self.extended_data[index]

    def __len__(self):
        return len(self.extended_data)
