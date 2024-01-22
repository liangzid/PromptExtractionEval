"""
======================================================================
EVA.1.MODELS_TABLE ---

Evaluate the scores of different LMs.

    Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
    Copyright © 2023, ZiLiang, all rights reserved.
    Created: 28 November 2023
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

from collections import OrderedDict

from metrics import ngram_recall_evaluate, fuzzy_match_recall


def eva_1(aptype="#E"):
    model_ls = [
        "lmsys/vicuna-7b-v1.5",
        # "microsoft/phi-1_5",
        # "NousResearch/Llama-2-7b-chat-hf",
        # "Qwen/Qwen-7B-Chat",
        # "mistralai/Mistral-7B-Instruct-v0.1",
        # "openchat/openchat_3.5",
    ]

    model_res_dict = {}
    for m in model_ls:
        suffix = m.split("/")[1]
        pth = "./model_eval_res/gluprompt_val_"+suffix+aptype+".json"
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
    with open(f"./model_eval_res/scores_aptype{aptype}.json",
              'w', encoding='utf8') as f:
        json.dump(model_res_dict, f, ensure_ascii=False, indent=4)
    return model_res_dict


def statistic_scores(pth1="./model_eval_res/scores_aptype#E.json",
                     pth2="./model_eval_res/scores_aptype#I.json",
                     eva_1=eva_1,
                     ):
    resE = eva_1(aptype="#E")
    resI = eva_1(aptype="#I")
    # else:
    #     # from collections import OrderedDict
    #     with open(pth1, 'r', encoding='utf8') as f:
    #         resE = json.load(f, object_pairs_hook=OrderedDict)
    #     with open(pth2, 'r', encoding='utf8') as f:
    #         resI = json.load(f, object_pairs_hook=OrderedDict)

    print("Now calculate the averaged score and the interval of models")
    eaveraged_models_res_dict = {}
    einterval_models_res_dict = {}
    iaveraged_models_res_dict = {}
    iinterval_models_res_dict = {}

    n_ls = list(range(3, 13, 3))
    r_ls = list(range(70, 101, 10))

    for m in resE:
        eaveraged_models_res_dict[m] = {"ngram": {},
                                        "fuzzy": {}}
        einterval_models_res_dict[m] = {"ngram": {},
                                        "fuzzy": {}}
        for n in n_ls:
            els = resE[m]["ngram"][n]
            v = sum(els)/len(els)
            eaveraged_models_res_dict[m]["ngram"][n] = v
            einterval_models_res_dict[m]["ngram"][n] = np.std(els,ddof=1)
        for r in r_ls:
            els = resE[m]["fuzzy"][r]
            v = sum(els)/len(els)
            eaveraged_models_res_dict[m]["fuzzy"][r] = v
            einterval_models_res_dict[m]["fuzzy"][r] = np.std(els,ddof=1)

        # interval list.
        iaveraged_models_res_dict[m] = {"ngram": {},
                                        "fuzzy": {}}
        iinterval_models_res_dict[m] = {"ngram": {},
                                        "fuzzy": {}}
        for n in n_ls:
            els = resI[m]["ngram"][n]
            v = sum(els)/len(els)
            iaveraged_models_res_dict[m]["ngram"][n] = v
            iinterval_models_res_dict[m]["ngram"][n] = np.std(els,ddof=1)
        for r in r_ls:
            els = resI[m]["fuzzy"][r]
            v = sum(els)/len(els)
            iaveraged_models_res_dict[m]["fuzzy"][r] = v
            iinterval_models_res_dict[m]["fuzzy"][r] = np.std(els,ddof=1)

    # Now format and print the results.
    print("EEEEEEEEEEEEEEEEEEEEEEE")
    print("-----Mean Scores----------")
    ppp(eaveraged_models_res_dict)
    print("-----interval Scores----------")
    ppp(einterval_models_res_dict)

    print("IIIIIIIIIIIIIIIIIIII")
    print("-----Mean Scores----------")
    ppp(iaveraged_models_res_dict)
    print("-----interval Scores----------")
    ppp(iinterval_models_res_dict)


# running entry
if __name__ == "__main__":
    statistic_scores()
    print("EVERYTHING DONE.")
