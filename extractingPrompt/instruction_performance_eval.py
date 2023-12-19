"""
======================================================================
3.INSTRUCTION_PERFORMANCE_EXPERIMENT ---

To evaluate the differences of performances under hand-crafted instructions
and generated instructions.

    Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
    Copyright © 2023, ZiLiang, all rights reserved.
    Created: 19 November 2023
======================================================================
"""


# ------------------------ Code --------------------------------------

# normal import
import json
from typing import List, Tuple, Dict
import random
from pprint import pprint as ppp

import torch
from tqdm import tqdm
from test_llama2_extracting import InferPromptExtracting

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline
)

import json
import logging
print = logging.info

logging.basicConfig(format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

torch.cuda.empty_cache()


def one_prompt_one_task_one_model(gen_pipeline, prompt,
                                  task_name, save_pth):

    logging.warning("This code has not been debug yet.")
    # models to be evaluted
    model_ls = [
        "lmsys/vicuna-7b-v1.5-16k",
        "microsoft/phi-1_5",
        "NousResearch/Llama-2-7b-chat-hf",
        "Qwen/Qwen-7B-Chat",
        "01-ai/Yi-6B",
        "mistralai/Mistral-7B-Instruct-v0.1",
        "openchat/openchat_3.5"]

    # tasks for evaluation
    tasks_name = ["valid_parentheses", "bool_logic",
                  "un_multi", "squad_v2",
                  "sst2", "wnli", "rte",
                  "mnli", "cola", "qqp",
                  "qnli", "mrpc",]

    glue_ds = ["ax", "cola", "mnli",
               "mnli_matched",
               "mnli_mismatched", "mrpc",
               "qnli", "qqp", "rte", "sst2",
               "stsb", "wnli",]

    tasks_we_used = [
        "cola", "mnli",
        "mrpc",
        "qnli", "qqp", "rte", "sst2",
        "wnli",]

    task_label_map = {
        "cola": {"1": "acceptable", "0": "unacceptable"},
        "mnli": {"1": "neutral", "0": "entailment", "2": "contradiction"},
        "mrpc": {"1": "equivalent", "2": "not_equivalent"},
        "qnli": {"1": "not_entailment", "0": "entailment"},
        "qqp": {"1": "duplicate", "0": "not_duplicate"},
        "rte": {"1": "not_entailment", "0": "entailment"},
        "sst2": {"1": "positive", "0": "negative"},
        "wnli": {"0": "not_entailment", "1": "entailment"},
    }
    task_key_map = {
        "mrpc": ["sentence1", "sentence2"],
        "qnli": ["question", "answer"],
        "qqp": ["question1", "question2"],
        "rte": ["sentence1", "sentence2"],
        "wnli": ["sentence1", "sentence2"],

    }
    single_input_tasks = ["cola", "sst2",]
    double_input_tasks = ["mrpc", "qnli", "qqp", "rte", "wnli",]

    # # evaluation dataset as well as its labels.
    # for t in tasks_name:
    #     if t not in glue_ds:
    #         print(t)

    # unseen_tasks = ["valid_parentheses", "bool_logic",
    #                 "un_multi", "squad_v2",]

    # obtain huggingface dataset paths
    assert task_name in tasks_we_used
    dataset = load_dataset("glue", task_name)
    res_ls = []
    if task_name in single_input_tasks:
        for d in dataset["validation"]:
            inps = d["sentence"]
            label = d["label"]
            label = task_label_map[task_name][str(label)]
            # print("--------------------")
            # print(prompt)
            # print("--------------------")
            # print(inps)
            res = gen_pipeline("Instruction: " + prompt +
                               " User: "+inps+" Assistant: ")
            res_ls.append((res, label))

    elif task_name == "mnli":
        for d in dataset["validation_matched"]:
            inps = d["premise"]+"SEP"+d["hypothesis"]
            label = d["label"]
            label = task_label_map[task_name][str(label)]
            res = gen_pipeline("Instruction: " + prompt +
                               " User: "+inps+" Assistant: ")
            res_ls.append((res, label))
    elif task_name in double_input_tasks:
        for d in dataset["validation"]:
            inps = d[task_key_map[task_name][0]]+"SEP" +\
                d[task_key_map[task_name][1]]
            label = d["label"]
            label = task_label_map[task_name][str(label)]
            res = gen_pipeline("Instruction: " + prompt +
                               f" User: The sentence is '{inps}'. "
                               + " Assistant: ")
            res_ls.append((res, label))
    else:
        logging.error(f"task name: {task_name} not found.")
    with open(save_pth, 'w', encoding='utf8') as f:
        json.dump(res_ls, f, ensure_ascii=False, indent=4)

    return res_ls


# running entry
if __name__ == "__main__":
    # main()
    print("EVERYTHING DONE.")
