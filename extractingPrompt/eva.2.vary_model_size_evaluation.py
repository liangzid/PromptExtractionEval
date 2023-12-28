"""
======================================================================
EVA.2.VARY_MODEL_SIZE_EVALUATION ---

Evaluation script of 2.model_size_prompt_extraction_experiments.py

    Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
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
    "1b-deduped",
    "1.4b-deduped",
    "2.8b-deduped",
    "6.9b-deduped",
    "12b-deduped",
]

# target_dir = "./opt_varying_modelsize/"

# model_ls = [
#     "125m",
#     "350m",
#     "1.3b",
#     "2.7b",
#     "6.7b",
#     "13b",
# ]



model_directly_res = {}

# from collections import OrderedDict
# with open(target_dir+"overall.json", 'r', encoding='utf8') as f:
#     all_data = json.load(f, object_pairs_hook=OrderedDict)

big_res = {}
for m in model_ls:
    # Evaluation Explicit Attacks
    # from collections import OrderedDict
    with open(target_dir+f"{m}.json", 'r',encoding='utf8') as f:
        data_dict=json.load(f,object_pairs_hook=OrderedDict)
    data = data_dict["E"]
    explicit_dict = {}
    for ap in data.keys():
        ins, gens = zip(* data[ap])

        gram_matchrate_dict = {}
        ratio_matchrate_dict = {}
        for n in range(3, 13, 3):
            gram_matchrate_dict[n] = ngram_recall_evaluate(gens, ins, n=n)
        for ratio in range(70, 101, 10):
            ratio_matchrate_dict[ratio] = fuzzy_match_recall(gens, ins,
                                                             ratio=ratio)

        explicit_dict[ap] = {"ngram": gram_matchrate_dict,
                             "fuzzy": ratio_matchrate_dict}

    # data = all_data[m]["I"]
    data = data_dict["I"]
    implicit_dict = {}
    for ap in data.keys():
        ins, gens = zip(* data[ap])

        gram_matchrate_dict = {}
        ratio_matchrate_dict = {}
        for n in range(3, 13, 3):
            gram_matchrate_dict[n] = ngram_recall_evaluate(gens, ins, n=n)
        for ratio in range(70, 101, 10):
            ratio_matchrate_dict[ratio] = fuzzy_match_recall(gens, ins,
                                                             ratio=ratio)

        implicit_dict[ap] = {"ngram": gram_matchrate_dict,
                             "fuzzy": ratio_matchrate_dict}
    big_res[m] = {"E": explicit_dict, "I": implicit_dict}


with open(target_dir+"scores.json", 'w', encoding='utf8') as f:
    json.dump(big_res, f, ensure_ascii=False, indent=4)


# running entry
if __name__ == "__main__":
    # main()
    print("EVERYTHING DONE.")
