"""
======================================================================
EVA.1.CLOSEAI --- 

    Author: Zi Liang <frost.liang@polyu.edu.hk>
    Copyright © 2023, ZiLiang, all rights reserved.
    Created: 19 December 2023
======================================================================
"""


# ------------------------ Code --------------------------------------
# normal import
import json
from typing import List, Tuple, Dict
import random
from pprint import pprint as ppp
import os
import numpy as np
import sys
sys.path.append("../")

from eva_1_models_table import statistic_scores

from collections import OrderedDict

from metrics import ngram_recall_evaluate, fuzzy_match_recall


def eva_2(aptype="#E",pth_prefix="./model_eval_res/gluprompt_val_"):
    model_ls = [
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-1106",
        "gpt-4-0314",
        "gpt-4-0613",
        "gpt-4-1106-preview",
    ]

    model_res_dict = {}
    for m in model_ls:
        # suffix = m.split("/")[1]
        pth = pth_prefix+m+aptype+".json"
        with open(pth, 'r', encoding='utf8') as f:
            data = json.load(f, object_pairs_hook=OrderedDict)

        n_ls = list(range(3, 13, 3))
        r_ls = list(range(70, 101, 10))

        n_res_dict_ls = {}
        r_res_dict_ls = {}
        for n in n_ls:
            n_res_dict_ls[n] = []
        for r in r_ls:
            r_res_dict_ls[r] = []

        for ap in data:
            inp_ps, genps = zip(* data[ap])
            for n in n_ls:
                n_res_dict_ls[n].append(
                    ngram_recall_evaluate(genps, inp_ps, n=n)
                )
            for r in r_ls:
                r_res_dict_ls[r].append(
                    fuzzy_match_recall(genps, inp_ps, ratio=r)
                )
        model_res_dict[m] = {
            "ngram": n_res_dict_ls,
            "fuzzy": r_res_dict_ls,
        }
    with open(f"./model_eval_res/close_ai_scores_aptype{aptype}.json",
              'w', encoding='utf8') as f:
        json.dump(model_res_dict, f, ensure_ascii=False, indent=4)
    return model_res_dict


if __name__=="__main__":
    statistic_scores(eva_1=eva_2)
