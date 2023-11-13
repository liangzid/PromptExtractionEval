"""
======================================================================
EVALUATE_ATTACKS ---

Evaluate whether there are duplications or not in generated texts.

    Author: Zi Liang <frost.liang@polyu.edu.hk>
    Copyright © 2023, ZiLiang, all rights reserved.
    Created: 13 November 2023
======================================================================
"""


# ------------------------ Code --------------------------------------

# normal import
import json
from typing import List, Tuple, Dict
import random
from pprint import pprint as ppp


def evaluate_consisting(gen_text, datas, n=3, stride=1):

    if "### Assistant" in gen_text:
        gen_text=gen_text.replace("### Assistant","")
    # 2. execute_fuzzy_ngram_match
    match_dict = {}
    grams = to_ngram(gen_text, n, stride)
    for g in grams:
        res = ngram_match(g, datas)
        if res != []:
            match_dict[g] = res

    ppp(f"match situation with n:{n}============")
    # ppp(match_dict)
    print(match_dict.keys())
    return match_dict


def ngram_match(g, samples_ls):

    matched_ls = []
    for s in samples_ls:
        if g in s:
            matched_ls.append(s)
    return matched_ls


def to_ngram(t, n=3, stride=1):
    tls = t.split(" ")
    n_gram_ls = []
    bgin_idx = 0
    while bgin_idx+n < len(tls):
        n_gram_ls.append(" ".join(tls[bgin_idx:bgin_idx+n]))
        bgin_idx += stride
    return n_gram_ls


def evaluate_success_rate(gens, task="contract_sections", n=3, stride=1):

    # 1. parse train corpus
    prefix = "/home/liangzi/code/attackFineTunedModels/data/"
    task_name_maps = {"contract_sections": "ContractSections",
                      "contract_types": "ContractTypes",
                      "us_crimes": "CrimeCharges", }
    train_pth = prefix+task_name_maps[task]+"___fewshot_dataset.json"
    train_pth += "____huggingface_format_train.jsonl"

    datas = []
    with open(train_pth, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for l in lines:
            l = l[:-1]  # to remove "\n"
            data = json.loads(l)["text"]
            datas.append(data)

    attack_hit = 0.
    sample_hitted = 0.
    attack_num = len(gens)
    sample_num = len(datas)

    all_attacked_ls = []

    for gen in gens:
        res = evaluate_consisting(gen, datas, n, stride)
        if res != {}:
            attack_hit += 1
            for x in res.keys():
                all_attacked_ls.extend(res[x])
    all_attacked_ls = list(set(all_attacked_ls))
    sample_hitted = len(all_attacked_ls)

    print(f"attack success rate: {attack_hit/attack_num}")
    print(f"training data hitted rate: {sample_hitted/sample_num}")
    return attack_hit/attack_num, sample_hitted/sample_num


# running entry
if __name__ == "__main__":
    print("EVERYTHING DONE.")
