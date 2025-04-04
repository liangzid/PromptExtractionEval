"""
======================================================================
STATISTIC_DATASET ---

Obtain the statistical information of the dataset.

    Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
    Copyright © 2025, ZiLiang, all rights reserved.
    Created: 29 March 2025
======================================================================
"""


# ------------------------ Code --------------------------------------

# normal import
import json
from typing import List, Tuple, Dict
import random
from pprint import pprint as ppp


from datasets import load_dataset

from transformers import AutoTokenizer


def obtainStatisticalInfo():
    dataset_name = "liangzid/PEAD"

    dataset = load_dataset(dataset_name, split="train")

    textls = dataset["text"]

    model_name = "meta-llama/Llama-3.1-8B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        )

    tokenlss = []
    len_token_ls = []

    res_dict = {
        "<2**4": 0.,
        "<2**5": 0.,
        "<2**6": 0.,
        "<2**7": 0.,
        "<2**8": 0.,
        "<2**9": 0.,
        "<2**10": 0.,
        "<2**11": 0.,
        "<2**12": 0.,
        ">2**12": 0.,
    }

    for t in textls:
        tokenls = tokenizer.encode(
            t,
            return_tensors="pt",
            )
        # print(tokenls)
        tokenls=tokenls[0]

        tokenlss.append(tokenls)
        length = len(tokenls)
        if length < 2**4:
            res_dict["<2**4"] += 1
        elif length < 2**5:
            res_dict["<2**5"] += 1
        elif length < 2**6:
            res_dict["<2**6"] += 1
        elif length < 2**7:
            res_dict["<2**7"] += 1
        elif length < 2**8:
            res_dict["<2**8"] += 1
        elif length < 2**9:
            res_dict["<2**9"] += 1
        elif length < 2**10:
            res_dict["<2**10"] += 1
        elif length < 2**11:
            res_dict["<2**11"] += 1
        elif length < 2**12:
            res_dict["<2**12"] += 1
        else:
            res_dict[">2**12"] += 1
        len_token_ls.append(len(tokenls))

    print(len_token_ls)
    print(f"Res Dict: {res_dict}")

    normalized_res_dict={}
    for key in res_dict:
        normalized_res_dict[key]=res_dict[key]/len(textls)
    print(f"Normalized Res Dict: {normalized_res_dict}")
    


if __name__ == "__main__":
    obtainStatisticalInfo()
