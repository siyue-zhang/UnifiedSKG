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


def main() -> None:

    parser = HfArgumentParser((WrappedSeq2SeqTrainingArguments,))
    training_args, = parser.parse_args_into_dataclasses()
    set_seed(training_args.seed)
    args = Configure.Get(training_args.cfg)
    print(args.id)
    assert 1==2
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
            if training_args.max_train_samples:
                task_args.max_train_samples = training_args.max_train_samples
            if training_args.max_eval_samples:
                task_args.max_eval_samples = training_args.max_eval_samples

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

    # We wrap the "string" seq2seq data into "tokenized tensor".
    train_dataset = TokenizedDataset(args, training_args, model_tokenizer,
                                     seq2seq_train_dataset) if seq2seq_train_dataset else None
    eval_dataset = TokenizedDataset(args, training_args, model_tokenizer,
                                    seq2seq_eval_dataset) if seq2seq_eval_dataset else None
    test_dataset = TokenizedDataset(args, training_args, model_tokenizer,
                                    seq2seq_test_dataset) if seq2seq_test_dataset else None
    
    trainer = EvaluateFriendlySeq2SeqTrainer(
        args=training_args,
        model=model,
        evaluator=evaluator,
        # We name it "evaluator" while the hugging face call it "Metric",
        # they are all f(predictions: List, references: List of dict) = eval_result: dict
        tokenizer=model_tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        eval_examples=seq2seq_eval_dataset,
    )

    # Load model weights (for --do_train=False or post finetuning).
    if training_args.load_weights_from:
        state_dict = torch.load(os.path.join(training_args.load_weights_from, transformers.WEIGHTS_NAME), map_location="cpu")
        trainer.model.load_state_dict(state_dict, strict=True)
        # release memory
        del state_dict


 
    # encoding = model_tokenizer(seq_in, return_tensors="pt")

    # # let the model generate an answer autoregressively
    # outputs = model.generate(**encoding)
    # print('outputs: ', outputs)  

    # # decode back to text
    # predicted_answer = model_tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    # print('predicted_answer: ', predicted_answer)    
 
if __name__ == "__main__":
   
    main(
        cfg='Salesforce/_T5_large_finetune_squall.cfg',
        load_weights_from='output/T5_large_finetune_squall/checkpoint-900',
        id="nt-4491",
        output_dir="./output/",
        generation_num_beams=5,
        max_train=1
    )
    
    