import logging
import os
import time

import torch
import datasets
import transformers
from transformers import (
    HfArgumentParser,
    set_seed,
    EarlyStoppingCallback,
    AutoModelForSeq2SeqLM
)
from transformers.trainer_utils import get_last_checkpoint
from collections import OrderedDict
import utils.tool
from utils.configue import Configure
from utils.dataset import TokenizedDataset
from utils.trainer import EvaluateFriendlySeq2SeqTrainer
from utils.training_arguments import WrappedSeq2SeqTrainingArguments
from seq2seq_construction.squall import squall_add_serialized_schema, squall_pre_process_one_function
from copy import deepcopy
import json

cfg='Salesforce/_Omnitab_large_finetune_squall_tableqa.cfg'
load_weights_from='output/Omnitab_large_finetune_squall_tableqa2/checkpoint-4700'
section='dev'
idx=0
generation_num_beams=5
max_train_samples=1
max_eval_samples=None

args = Configure.Get(cfg)

from filelock import FileLock
import nltk
with FileLock(".lock") as lock:
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)

if not args.arg_paths:
    cache_root = os.path.join('output', 'cache')
    os.makedirs(cache_root, exist_ok=True)
    raw_datasets_split: datasets.DatasetDict = datasets.load_dataset(path=args.dataset.loader_path,
                                                                    cache_dir=args.dataset.data_store_path)
    seq2seq_dataset_split: tuple = utils.tool.get_constructor(args.seq2seq.constructor)(args).to_seq2seq(
        raw_datasets_split, cache_root)
else:
    cache_root = os.path.join('output', 'cache')
    os.makedirs(cache_root, exist_ok=True)
    meta_tuning_data = {}
    for task, arg_path in args.arg_paths:
        task_args = Configure.Get(arg_path)
        # new
        task_args.bert, task_args.model = args.bert, args.model
        if max_train_samples:
            task_args.max_train_samples = max_train_samples
        if max_eval_samples:
            task_args.max_eval_samples = max_eval_samples

        print('task_args.bert.location:', task_args.bert.location)
        task_raw_datasets_split: datasets.DatasetDict = datasets.load_dataset(
            path=task_args.dataset.loader_path,
            cache_dir=task_args.dataset.data_store_path)

        task_seq2seq_dataset_split: tuple = utils.tool.get_constructor(task_args.seq2seq.constructor)(task_args).\
            to_seq2seq(task_raw_datasets_split, cache_root)

        meta_tuning_data[arg_path] = task_seq2seq_dataset_split

    seq2seq_dataset_split: tuple = utils.tool.get_constructor(args.seq2seq.constructor)(args).\
        to_seq2seq(meta_tuning_data)    

evaluator = utils.tool.get_evaluator(args.evaluate.tool)(args)
model = utils.tool.get_model(args.model.name)(args)
model_tokenizer = model.tokenizer


seq2seq_train_dataset, seq2seq_eval_dataset, seq2seq_test_dataset = None, None, None
if len(seq2seq_dataset_split) == 2:
    seq2seq_train_dataset, seq2seq_eval_dataset = seq2seq_dataset_split
elif len(seq2seq_dataset_split) == 3:
    seq2seq_train_dataset, seq2seq_eval_dataset, seq2seq_test_dataset = seq2seq_dataset_split
else:
    raise ValueError("Other split not support yet.")

if section=='dev':
    dataset = seq2seq_eval_dataset
else:
    dataset = seq2seq_test_dataset

print(dataset)