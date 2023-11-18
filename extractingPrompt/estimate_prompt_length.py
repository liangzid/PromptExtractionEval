"""
======================================================================
ESTIMATE_PROMPT_LENGTH --- 
varying prompt length for experiments.

    Author: Zi Liang <frost.liang@polyu.edu.hk>
    Copyright © 2023, ZiLiang, all rights reserved.
    Created: 18 November 2023
======================================================================
"""


# ------------------------ Code --------------------------------------

# normal import
import json
from typing import List, Tuple, Dict
import random
from pprint import pprint as ppp

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline
)

import torch
from datasets import load_dataset


def estimate_sequence_len():
    prompt_dataset = "liangzid/prompts"
    model_name = "NousResearch/Llama-2-7b-chat-hf"

    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              trust_remote_code=True)
    prompts = load_dataset(prompt_dataset)["train"].to_list()
    length_ls = []
    for p in prompts:
        pp = p["text"]
        inps = tokenizer(pp).input_ids
        # print(inps)
        length_ls.append(len(inps))
    return length_ls


def main():
    res=estimate_sequence_len()
    print(res)


# running entry
if __name__ == "__main__":
    main()
    print("EVERYTHING DONE.")
