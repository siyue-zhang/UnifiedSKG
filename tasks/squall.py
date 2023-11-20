# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""SQUALL: Lexical-level Supervised Table Question Answering Dataset."""


import json
import re, os
import datasets
from datasets.tasks import QuestionAnsweringExtractive


logger = datasets.logging.get_logger(__name__)


_CITATION = """\
@inproceedings{Shi:Zhao:Boyd-Graber:Daume-III:Lee-2020,
	Title = {On the Potential of Lexico-logical Alignments for Semantic Parsing to {SQL} Queries},
	Author = {Tianze Shi and Chen Zhao and Jordan Boyd-Graber and Hal {Daum\'{e} III} and Lillian Lee},
	Booktitle = {Findings of EMNLP},
	Year = {2020},
}
"""

_DESCRIPTION = """\
To explore the utility of fine-grained, lexical-level supervision, authors \
introduce SQUALL, a dataset that enriches 11,276 WikiTableQuestions \ 
English-language questions with manually created SQL equivalents plus \ 
alignments between SQL and question fragments.
"""

_URL = "https://raw.githubusercontent.com/tzshi/squall/main/data/"
_URLS = {
    "squall": _URL + "squall.json",
    "wtq-test": _URL + "wtq-test.json",
    "dev-0": _URL +  "dev-0.ids",
    "dev-1": _URL +  "dev-1.ids",
    "dev-2": _URL +  "dev-2.ids",
    "dev-3": _URL +  "dev-3.ids",
    "dev-4": _URL +  "dev-4.ids",
}

# class SquallConfig(datasets.BuilderConfig):
#     """BuilderConfig for Squall."""

#     def __init__(self, **kwargs):
#         """BuilderConfig for Squall.
#         Args:
#           **kwargs: keyword arguments forwarded to super.
#         """
#         super(SquallConfig, self).__init__(**kwargs)


class Squall(datasets.GeneratorBasedBuilder):
    """SQUALL: Lexical-level Supervised Table Question Answering Dataset."""

    # BUILDER_CONFIGS = [
    #     SquallConfig(name = '0'),
    #     SquallConfig(name = '1'),
    #     SquallConfig(name = '2'),
    #     SquallConfig(name = '3'),
    #     SquallConfig(name = '4')
    # ]
    
    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "nt": datasets.Value("string"),
                    "tbl": datasets.Value("string"),
                    "columns":
                        {
                            "raw_header": datasets.features.Sequence(datasets.Value("string")),
                            "tokenized_header": datasets.features.Sequence(datasets.features.Sequence(datasets.Value("string"))),
                            "column_suffixes": datasets.features.Sequence(datasets.features.Sequence(datasets.Value("string"))),
                            "column_dtype": datasets.features.Sequence(datasets.Value("string")),
                            "example": datasets.features.Sequence(datasets.Value("string"))
                        },
                    "nl": datasets.features.Sequence(datasets.Value("string")),
                    "nl_pos": datasets.features.Sequence(datasets.Value("string")),
                    "nl_ner": datasets.features.Sequence(datasets.Value("string")),
                    "nl_incolumns": datasets.features.Sequence(datasets.Value("bool_")),
                    "nl_incells": datasets.features.Sequence(datasets.Value("bool_")),
                    "columns_innl": datasets.features.Sequence(datasets.Value("bool_")),
                    "tgt": datasets.Value("string"),
                    "sql": {
                        "sql_type": datasets.features.Sequence(datasets.Value("string")), 
                        "value": datasets.features.Sequence(datasets.Value("string")), 
                        "span_indices": datasets.features.Sequence(datasets.features.Sequence(datasets.Value("int32")))
                    },
                    "nl_ralign": {
                        "aligned_sql_token_type":datasets.features.Sequence(datasets.Value("string")),
                        "aligned_sql_token_info":datasets.features.Sequence(datasets.Value("string")),
                    },
                    "align":{
                        "nl_indices": datasets.features.Sequence(datasets.features.Sequence(datasets.Value("int32"))),
                        "sql_indices": datasets.features.Sequence(datasets.features.Sequence(datasets.Value("int32")))
                    }
                }
            ),
            # No default supervised_keys (as we have to pass both question
            # and context as input).
            supervised_keys=None,
            homepage="https://github.com/tzshi/squall/",
            citation=_CITATION,
            task_templates=[
                QuestionAnsweringExtractive(
                    question_column="nl", context_column="columns", answers_column="tgt"
                )
            ],
        )

    def _split_generators(self, dl_manager):
        urls_to_download = {
            "squall": _URLS["squall"],
            "wtq-test": _URLS["wtq-test"],
            "dev-0": _URLS["dev-0"],
            "dev-1": _URLS["dev-1"],
            "dev-2": _URLS["dev-2"],
            "dev-3": _URLS["dev-3"],
            "dev-4": _URLS["dev-4"],
        }

        downloaded_files = dl_manager.download_and_extract(urls_to_download)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, 
                gen_kwargs={"split_key": "train", "filepath": downloaded_files}),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION, 
                gen_kwargs={"split_key": "dev", "filepath": downloaded_files}),
            datasets.SplitGenerator(
                name=datasets.Split.TEST, 
                gen_kwargs={"split_key": "test", "filepath": downloaded_files}),
        ]

    def _generate_examples(self, split_key, filepath):
        """This function returns the examples in the raw (text) form."""
        logger.info("generating examples from = %s", filepath)

        squall_full = filepath["squall"]
        dev_ids = filepath["dev-1"]
        test = filepath["wtq-test"]

        # transform the original squall data structure (list of things) to dict 
        def transform(sample, sample_key, keys):
            cols = {}
            n_col = len(sample[sample_key])
            for k in range(len(keys)):
                tmp = []
                for j in range(n_col):
                    tmp.append(sample[sample_key][j][k])
                cols[keys[k]] = tmp
            return cols

        columns_keys = ["raw_header", "tokenized_header", "column_suffixes", "column_dtype", "example"]
        sql_keys = ["sql_type", "value", "span_indices"]

        with open(squall_full, encoding="utf-8") as f:
            squall_full_data = json.load(f)
    
        NUM_MAPPING = {
            'half': 0.5,
            'one': 1,
            'two': 2,
            'three': 3,
            'four': 4,
            'five': 5,
            'six': 6,
            'seven': 7,
            'eight': 8,
            'nine': 9,
            'ten': 10,
            'eleven': 11,
            'twelve': 12,
            'twenty': 20,
            'thirty': 30,
            'once': 1,
            'twice': 2,
            'first': 1,
            'second': 2,
            'third': 3,
            'fourth': 4,
            'fifth': 5,
            'sixth': 6,
            'seventh': 7,
            'eighth': 8,
            'ninth': 9,
            'tenth': 10,
            'hundred': 100,
            'thousand': 1000,
            'million': 1000000,
            'jan': 1,
            'feb': 2,
            'mar': 3,
            'apr': 4,
            'may': 5,
            'jun': 6,
            'jul': 7,
            'aug': 8,
            'sep': 9,
            'oct': 10,
            'nov': 11,
            'dec': 12,
            'january': 1,
            'february': 2,
            'march': 3,
            'april': 4,
            'june': 6,
            'july': 7,
            'august': 8,
            'september': 9,
            'october': 10,
            'november': 11,
            'december': 12,
        }

        def parse_number(s):
            if s in NUM_MAPPING:
                return NUM_MAPPING[s]
            s = s.replace(',', '')
            # https://stackoverflow.com/questions/4289331/python-extract-numbers-from-a-string
            ret = re.findall(r"[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", s)
            if len(ret) > 0:
                return ret[0]
            return None

        for instance in squall_full_data:
            has_number = False
            numbers = []
            for x in instance["nl"]:
                numbers.append(parse_number(x))
                if numbers[-1] is not None:
                    has_number = True
            instance["numbers"] = numbers
            instance["has_number"] = has_number

        if split_key != 'test':
            with open(dev_ids) as f:
                dev_ids = json.load(f)
            if split_key == "train":
                set = [x for x in squall_full_data if x["tbl"] not in dev_ids]
            else:
                set = [x for x in squall_full_data if x["tbl"] in dev_ids]
        else:
            with open(test, encoding="utf-8") as f:
                set = json.load(f)

        for idx, sample in enumerate(set):
            # transform columns
            cols = transform(sample, "columns", columns_keys)
            if split_key == 'test':
                query = ''
            else:
                # transform sql
                sqls = transform(sample, "sql", sql_keys)
                query = ' '.join(sqls['value'])
                
            raw_header = cols['raw_header']
            raw_header = ['unknown' if element == '' else element for element in raw_header]
            raw_header = [raw_header[i]+f'_{i}' for i in range(len(raw_header))]
            column_suffixes = cols['column_suffixes']
            column_names = []
            ori_column_names = []
            for idx, h in enumerate(column_suffixes):
                column_names.append(raw_header[idx])
                ori_column_names.append(f'c{idx+1}')
                for suf in h:
                    column_names.append(raw_header[idx]+'_'+suf)
                    ori_column_names.append(f'c{idx+1}_{suf}')
            
            yield idx, {
                "query": query,
                "question_id": sample["nt"],
                "question": ' '.join(sample["nl"]),
                "table_id": sample["tbl"],
                "db_path": f"/scratch/sz4651/Projects/UnifiedSKG/third_party/squall/tables/db/{sample['tbl']}.db",
                "header_names": raw_header,
                "column_names": column_names,
                "ori_column_names": ori_column_names,
                "answer_text": sample["tgt"],
            }
