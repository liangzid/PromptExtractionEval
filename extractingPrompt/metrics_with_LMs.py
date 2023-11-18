"""
======================================================================
METRICS_WITH_LMS ---

High-level evluation, about the quality of LMs.

    Author: Zi Liang <frost.liang@polyu.edu.hk>
    Copyright © 2023, ZiLiang, all rights reserved.
    Created: 17 November 2023
======================================================================
"""


# ------------------------ Code --------------------------------------

# normal import
import json
from typing import List, Tuple, Dict
import random
from pprint import pprint as ppp
# import pickle
# import os
# from os.path import join, exists
# from collections import Counter,OrderedDict
# from bisect import bisect
# from copy import deepcopy
# import pickle

# import sys
# # sys.path.append("./")
# from tqdm import tqdm

# import numpy as np

# import argparse
# import logging


# deep learning related import
# import numpy as np

import torch
# from torch.utils.tensorboard import SummaryWriter
# from torch.nn import DataParallel
# from torch.utils.data import Dataset, DataLoader
# from torch.nn import CrossEntropyLoss
# import torch.nn as nn
# from torch.nn.parallel import DistributedDataParallel
# from torchvision.transforms import Compose as ComposeTransformation
# import tensorboardX

# ## transformers related import
# from transformers import T5Tokenizer,T5ForConditionalGeneration
# from transformers import BertTokenizer
# from transformers import pipeline
# import transformers

from transformers import pipeline
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline
)


def acceptability(gens):
    model_name = "Abirate/bert_fine_tuned_cola"

    cola = pipeline("text-classification", model=model_name,from_tf=True)

    scores = []
    for g in gens:
        res = cola(g)
        scores.append(res[0]["score"])
    return sum(scores)/len(scores)


def semantic_rating_gpt3point5_turbo(gens, ps):
    # not used. We have no now.
    instruction = "Given sentences A and B, please rate the semantic similarity between A and B on a scale of 1 to 5, where 5 represents nearly perfect match, 4 represents generally matching, 3 represents partial match, 2 represents few areas matching, 1 represents semantic mismatch in most areas, and 0 represents almost no match. Your first word in the answer must be a number between 1 and 5:"

    scores = []
    for i in range(len(gens)):
        query = instruction+f" Sentence A: {gens[i]} Setence B: {ps[i]}"
        resp = ""
        score = float(int(resp[0]))
        scores.append(score)
    return sum(scores)/len(scores)


def semantic_rating_llama2_chat_7b(gens, ps):
    instruction = "Given sentences A and B, please rate the semantic similarity between A and B on a scale of 1 to 5, where 5 represents nearly perfect match, 4 represents generally matching, 3 represents partial match, 2 represents few areas matching, 1 represents semantic mismatch in most areas, and 0 represents almost no match. Your first word in the answer must be a number between 1 and 5:"

    model_name = "NousResearch/Llama-2-7b-chat-hf"

    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # quantization_config=quant_config,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    text_gen = pipeline(task="text-generation",
                        model=model,
                        tokenizer=tokenizer,
                        max_length=2047)

    scores = []
    for i in range(len(gens)):
        query = instruction +\
            f" ### User: Sentence A: {gens[i]} Setence B: {ps[i]}" +\
            " ### System: "
        resp = text_gen(query)[0]["generated_text"]
        if "1" in resp:
            score = 1
        elif "2" in resp:
            score = 2
        elif "3" in resp:
            score = 3
        elif "4" in resp:
            score = 4
        elif "5" in resp:
            score = 5
        else:
            print("ERROR. NO score returned.")
            print(f"Raw response: {resp}")
            return -1
        # score=float(int(resp[0]))
        scores.append(score)
    return sum(scores)/len(scores)


def information_cover_gpt3point5_turbo(gens, ps):
    # not used. We have no now.
    instruction = "Given sentences A and B, please determine whether sentence A covers all the information contained in sentence B and rate it on a scale of 1 to 5, where 5 represents complete coverage of information, 4 represents substantial coverage, 3 represents partial coverage, 2 represents minimal coverage, 1 represents lack of coverage for most information, and 0 represents no coverage at all. Your first word in the answer must be a number between 1 and 5:"

    scores = []
    for i in range(len(gens)):
        query = instruction+f" Sentence A: {gens[i]} Setence B: {ps[i]}"
        resp = ""
        score = float(int(resp[0]))
        scores.append(score)
    return sum(scores)/len(scores)


def information_cover_llama2_chat_7b(gens, ps):
    instruction = "Given sentences A and B, please determine whether sentence A covers all the information contained in sentence B and rate it on a scale of 1 to 5, where 5 represents complete coverage of information, 4 represents substantial coverage, 3 represents partial coverage, 2 represents minimal coverage, 1 represents lack of coverage for most information, and 0 represents no coverage at all. Your first word in the answer must be a number between 1 and 5:"

    model_name = "NousResearch/Llama-2-7b-chat-hf"

    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # quantization_config=quant_config,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    text_gen = pipeline(task="text-generation",
                        model=model,
                        tokenizer=tokenizer,
                        max_length=2047)

    scores = []
    for i in range(len(gens)):
        query = instruction +\
            f" ### User: Sentence A: {gens[i]} Setence B: {ps[i]}" +\
            " ### System: "
        resp = text_gen(query)[0]["generated_text"]
        if "1" in resp:
            score = 1
        elif "2" in resp:
            score = 2
        elif "3" in resp:
            score = 3
        elif "4" in resp:
            score = 4
        elif "5" in resp:
            score = 5
        else:
            print("ERROR. NO score returned.")
            print(f"Raw response: {resp}")
            return -1
        # score=float(int(resp[0]))
        scores.append(score)
    return sum(scores)/len(scores)


def perplexity_llama2_7b(gens):

    device = "cuda:0"
    model_name = "NousResearch/Llama-2-7b-chat-hf"

    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # quantization_config=quant_config,
        device_map=device,
        trust_remote_code=True,
    )

    inps = tokenizer(gens, return_tensors="pt", padding=True,
                     truncation=True)
    inps_ids = inps.input_ids.to(device)
    att_msk = inps.attention_mask.to(device)

    with torch.no_grad():
        loss = model(inps_ids, att_msk, label=inps_ids).loss

    return torch.exp(loss)


def main():
    # test metrics without OpenAI's inferface

    ps = ["Given sentences A and B, please determine whether sentence A covers all the information contained in sentence B and rate it on a scale of 1 to 5, where 5 represents complete coverage of information, 4 represents substantial coverage, 3 represents partial coverage, 2 represents minimal coverage, 1 represents lack of coverage for most information, and 0 represents no coverage at all. Your first word in the answer must be a number between 1 and 5:"]

    gens = ["Geven two sentence, give me a rating of their information coverage, varing from 1 to 5."]

    # print(acceptability(gens))
    print(semantic_rating_llama2_chat_7b(gens, ps))
    print(information_cover_llama2_chat_7b(gens, ps))
    print(perplexity_llama2_7b(gens))

    # open AI related work
    # print(semantic_rating_gpt3point5_turbo(gens,ps))
    # print(information_cover_gpt3point5_turbo)


# running entry
if __name__ == "__main__":
    main()
    print("EVERYTHING DONE.")
