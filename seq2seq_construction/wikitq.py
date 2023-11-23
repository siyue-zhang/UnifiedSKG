import copy
import os
from copy import deepcopy

import torch
from datasets import DatasetDict
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from transformers import AutoTokenizer
from tqdm import tqdm
from utils.processor import get_default_processor

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


"""
    Raw data are formatted as:
    {
        "id": datasets.Value("string"),
        "question": datasets.Value("string"),
        "table_id": datasets.Value("string"),
        "table": {"header": datasets.features.Sequence(datasets.Value("string")),
                  "rows": datasets.features.Sequence(datasets.features.Sequence(datasets.Value("string")))},
        "answer_text": datasets.features.Sequence(datasets.Value("string")),
    }
    """


class TrainDataset(Dataset):

    def __init__(self, args, raw_datasets, cache_root):
        # This tab processor is for table truncation and linearize.
        self.raw_datasets = raw_datasets

        cache_path = os.path.join(cache_root, 'wikitq_train.cache')
        if os.path.exists(cache_path) and args.dataset.use_cache:
            self.extended_data = torch.load(cache_path)
        else:
            if args.model.knowledge_usage == 'tapex':
                tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large')
            else:
                tokenizer = AutoTokenizer.from_pretrained(args.bert.location, use_fast=False)
            self.tab_processor = get_default_processor(max_cell_length=15,
                                                       tokenizer=tokenizer,
                                                       max_input_length=args.seq2seq.table_truncation_max_length)

            self.extended_data = []
            expansion = args.seq2seq.expansion if args.seq2seq.expansion else 1
            for expand_id in range(expansion):
                for raw_data in tqdm(self.raw_datasets):
                    extend_data = deepcopy(raw_data)
                    question = extend_data["question"]
                    table = extend_data['table']
                    # table = {'header': ['Year', 'Division', 'League', 'Regular Season', 
                    # 'Playoffs', 'Open Cup', 'Avg. Attendance'], 'rows': [['2001', 
                    # '2', 'USL A-League', '4th, Western', 'Quarterfinals', 
                    # 'Did not qualify', '7,169'], ['2002', '2', 'USL A-League', 
                    # '2nd, Pacific', '1st Round', 'Did not qualify', '6,260'], 
                    # ['2003', '2', 'USL A-League', '3rd, Pacific', 'Did not qualify', 
                    # 'Did not qualify', '5,871'], ['2004', '2', 'USL A-League', 
                    # '1st, Western', 'Quarterfinals', '4th Round', '5,628'], 
                    # ['2005', '2', 'USL First Division', '5th', 'Quarterfinals',
                    #  '4th Round', '6,028'], ['2006', '2', 'USL First Division', 
                    # '11th', 'Did not qualify', '3rd Round', '5,575'], 
                    # ['2007', '2', 'USL First Division', '2nd', 'Semifinals',
                    #  '2nd Round', '6,851'], ['2008', '2', 'USL First Division',
                    #  '11th', 'Did not qualify', '1st Round', '8,567'], ['2009', 
                    # '2', 'USL First Division', '1st', 'Semifinals', '3rd Round',
                    #  '9,734'], ['2010', '2', 'USSF D-2 Pro League', '3rd, USL (3rd)',
                    #  'Quarterfinals', '3rd Round', '10,727']]}
                    gold_result = extend_data['answer_text']

                    table_context = copy.deepcopy(table)
                    # modify a table internally
                    for truncate_func in self.tab_processor.table_truncate_funcs:
                        truncate_func.truncate_table(table_context, question, gold_result)
                    # linearize a table into a string
                    linear_table = self.tab_processor.table_linearize_func.process_table(table_context)
                    seq_out = self.tab_processor.process_output(gold_result)

                    extend_data.update({"struct_in": linear_table.lower(),
                                        "text_in": question.lower(),
                                        "seq_out": seq_out.lower(),
                                        "table_context": table_context})
                    self.extended_data.append(extend_data)
            if args.dataset.use_cache:
                torch.save(self.extended_data, cache_path)

    def __getitem__(self, index) -> T_co:
        return self.extended_data[index]

    def __len__(self):
        return len(self.extended_data)


class DevDataset(Dataset):

    def __init__(self, args, raw_datasets, cache_root):
        # This tab processor is for table truncation and linearize.
        self.raw_datasets = raw_datasets

        cache_path = os.path.join(cache_root, 'wikitq_dev.cache')
        if os.path.exists(cache_path) and args.dataset.use_cache:
            self.extended_data = torch.load(cache_path)
        else:
            if args.model.knowledge_usage == 'tapex':
                tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large')
            else:
                tokenizer = AutoTokenizer.from_pretrained(args.bert.location, use_fast=False)
            self.tab_processor = get_default_processor(max_cell_length=15,
                                                       tokenizer=tokenizer,
                                                       max_input_length=args.seq2seq.table_truncation_max_length)

            self.extended_data = []
            for raw_data in tqdm(self.raw_datasets):
                extend_data = deepcopy(raw_data)
                question = extend_data["question"]
                table = extend_data['table']
                gold_result = extend_data['answer_text']

                table_context = copy.deepcopy(table)
                # modify a table internally
                for truncate_func in self.tab_processor.table_truncate_funcs:
                    truncate_func.truncate_table(table_context, question, gold_result)
                # linearize a table into a string
                linear_table = self.tab_processor.table_linearize_func.process_table(table_context)
                seq_out = self.tab_processor.process_output(gold_result)

                extend_data.update({"struct_in": linear_table.lower(),
                                    "text_in": question.lower(),
                                    "seq_out": seq_out.lower(),
                                    "table_context": table_context})
                self.extended_data.append(extend_data)
            if args.dataset.use_cache:
                torch.save(self.extended_data, cache_path)

    def __getitem__(self, index) -> T_co:
        return self.extended_data[index]

    def __len__(self):
        return len(self.extended_data)


class TestDataset(Dataset):

    def __init__(self, args, raw_datasets, cache_root):
        # This tab processor is for table truncation and linearize.
        self.raw_datasets = raw_datasets

        cache_path = os.path.join(cache_root, 'wikitq_test.cache')
        if os.path.exists(cache_path) and args.dataset.use_cache:
            self.extended_data = torch.load(cache_path)
        else:
            if args.model.knowledge_usage == 'tapex':
                location = 'facebook/bart-large'
            else:
                location = args.bert.location
            tokenizer = AutoTokenizer.from_pretrained(location, use_fast=False)
            self.tab_processor = get_default_processor(max_cell_length=15,
                                                       tokenizer=tokenizer,
                                                       max_input_length=args.seq2seq.table_truncation_max_length)
            
            self.extended_data = []
            for raw_data in tqdm(self.raw_datasets):
                extend_data = deepcopy(raw_data)
                question = extend_data["question"]
                table = extend_data['table']
                gold_result = extend_data['answer_text']

                table_context = copy.deepcopy(table)
                # modify a table internally
                for truncate_func in self.tab_processor.table_truncate_funcs:
                    truncate_func.truncate_table(table_context, question, gold_result)
                # linearize a table into a string
                linear_table = self.tab_processor.table_linearize_func.process_table(table_context)
                seq_out = self.tab_processor.process_output(gold_result)

                extend_data.update({"struct_in": linear_table.lower(),
                                    "text_in": question.lower(),
                                    "seq_out": seq_out.lower(),
                                    "table_context": table_context})
                self.extended_data.append(extend_data)
            if args.dataset.use_cache:
                torch.save(self.extended_data, cache_path)

    def __getitem__(self, index) -> T_co:
        return self.extended_data[index]

    def __len__(self):
        return len(self.extended_data)
