import os
import torch
import random
import re
from copy import deepcopy
from typing import List, Dict

from datasets.dataset_dict import DatasetDict
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co

from third_party.miscs.bridge_content_encoder import get_database_matches

from tqdm import tqdm



"""
    Wrap the raw dataset into the seq2seq one.
    And the raw dataset item is formatted as
    {
    "query": query,
    "question_id": sample["nt"],
    "question": ' '.join(sample["nl"]),
    "table_id": sample["tbl"],
    "db_path": f"./third_party/squall/tables/db/{sample['tbl']}.db",
    "header_names": raw_header,
    "column_names": column_names,
    "ori_column_names": ori_column_names,
    "answer_text": sample["tgt"],
    }
    """


class Constructor(object):
    def __init__(self, args):
        self.args = args

    def to_seq2seq(self, raw_datasets: DatasetDict, cache_root: str):
        if not len(raw_datasets) == 3:
            raise AssertionError("Train, Dev, Test sections of dataset expected.")
        train_dataset = TrainDataset(self.args, raw_datasets['train'], cache_root)
        dev_dataset = DevDataset(self.args, raw_datasets['validation'], cache_root)
        test_dataset = TestDataset(self.args, raw_datasets['test'], cache_root)

        return train_dataset, dev_dataset, test_dataset


class TrainDataset(Dataset):
    def __init__(self, args, raw_datasets, cache_root):
        self.args = args
        self.raw_datasets = raw_datasets

        cache_path = os.path.join(cache_root, 'squall_train.cache')
        if os.path.exists(cache_path) and args.dataset.use_cache:
            self.extended_data = torch.load(cache_path)
        else:
            self.extended_data = []
            for raw_data in tqdm(self.raw_datasets):
                extend_data = deepcopy(raw_data)
                print(extend_data)
                assert 1==2

                extend_data.update(squall_add_serialized_schema(extend_data, args))

                question, seq_out = squall_pre_process_one_function(extend_data, args=self.args)
                extend_data.update({"struct_in": extend_data["serialized_schema"].strip(),
                                    "text_in": question,
                                    "seq_out": seq_out})
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

        cache_path = os.path.join(cache_root, 'squall_dev.cache')
        if os.path.exists(cache_path) and args.dataset.use_cache:
            self.extended_data = torch.load(cache_path)
        else:
            self.extended_data = []
            for raw_data in tqdm(self.raw_datasets):
                extend_data = deepcopy(raw_data)
                extend_data.update(squall_add_serialized_schema(extend_data, args))

                question, seq_out = squall_pre_process_one_function(extend_data, args=self.args)
                extend_data.update({"struct_in": extend_data["serialized_schema"].strip(),
                                    "text_in": question,
                                    "seq_out": seq_out})
                self.extended_data.append(extend_data)
            if args.dataset.use_cache:
                torch.save(self.extended_data, cache_path)

    def __getitem__(self, index) -> T_co:
        return self.extended_data[index]

    def __len__(self):
        return len(self.extended_data)
