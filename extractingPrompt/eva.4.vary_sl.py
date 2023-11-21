"""
======================================================================
EVA.4.VARY_SL --- 

    Author: Zi Liang <frost.liang@polyu.edu.hk>
    Copyright © 2023, ZiLiang, all rights reserved.
    Created: 21 November 2023
======================================================================
"""


# ------------------------ Code --------------------------------------

# normal import
import json
from typing import List, Tuple, Dict
import random
from pprint import pprint as ppp

from collections import OrderedDict
import os

from metrics import ngram_recall_evaluate
from metrics import fuzzy_match_recall


target_dir = "./vary_sl/"

model_ls = [
    "phi-1_5",
    # "Llama-2-7b-chat-hf",
]

model_directly_res = {}

for m in model_ls:
    fpth = target_dir+m+"-res.json"
    with open(fpth, 'r', encoding='utf8') as f:
        datas = json.load(f, object_pairs_hook=OrderedDict)

    # evaluation
    for query in datas.keys():
        q_dict = {}
        for interval in datas[query].keys():
            data = datas[query][interval]
            ins, gens = zip(*data)

            gram_matchrate_dict = {}
            ratio_matchrate_dict = {}
            for n in range(3, 15):
                gram_matchrate_dict[n] = ngram_recall_evaluate(gens, ins, n=n)
            for ratio in range(60, 101, 10):
                ratio_matchrate_dict[ratio] = fuzzy_match_recall(gens, ins,
                                                                 ratio=ratio)
            ppp(gram_matchrate_dict)
            ppp(ratio_matchrate_dict)
            q_dict[interval] = {"ngram": gram_matchrate_dict,
                                "fuzzy": ratio_matchrate_dict}

            for sample in data:
                label = sample[0]
                extracted = sample[1]
                if "\n" in extracted:
                    es = list(set(extracted.split("\n")))
        model_directly_res[query] = q_dict

ppp(model_directly_res)


# running entry
if __name__ == "__main__":
    # main()
    print("EVERYTHING DONE.")
