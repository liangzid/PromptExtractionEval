"""
======================================================================
EVA.2.VARY_MODEL_SIZE_EVALUATION ---

Evaluation script of 2.model_size_prompt_extraction_experiments.py

    Author: Zi Liang <frost.liang@polyu.edu.hk>
    Copyright © 2023, ZiLiang, all rights reserved.
    Created: 20 November 2023
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


target_dir = "./pythia_p_model_res/"

model_ls = [
    "70m-deduped",
    "160m-deduped",
    "410m-deduped",
    "1.4b-deduped",
    "1b-deduped",
    "2.8b-deduped",
    "6.9b-deduped",
    "12b-deduped",
]

model_directly_res={}

for m in model_ls:
    fpth = target_dir+"model_size-----"+m+"smallres.json"
    with open(fpth, 'r', encoding='utf8') as f:
        data = json.load(f, object_pairs_hook=OrderedDict)

    # evaluation
    ins, gens = zip(*data)

    gram_matchrate_dict={}
    ratio_matchrate_dict={}
    for n in range(3,8):
        gram_matchrate_dict[n]=ngram_recall_evaluate(gens,ins,n=n)
    for ratio in range(60,101,10):
        ratio_matchrate_dict[ratio] = fuzzy_match_recall(gens, ins,
                                                         ratio=ratio)
    ppp(gram_matchrate_dict)
    ppp(ratio_matchrate_dict)
    model_directly_res[m]={"ngram":gram_matchrate_dict,
                           "fuzzy":ratio_matchrate_dict}

    for sample in data:
        label = sample[0]
        extracted = sample[1]
        if "\n" in extracted:
            es = list(set(extracted.split("\n")))

ppp(model_directly_res)

# running entry
if __name__ == "__main__":
    # main()
    print("EVERYTHING DONE.")
