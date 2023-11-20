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

## normal import 
import json
from typing import List,Tuple,Dict
import random
from pprint import pprint as ppp

from collections import OrderedDict
import os

from metrics import ngram_recall_evaluate
from metrics import fuzzy_match_recall


target_dir="./pythia_p_model_res/"

model_ls=[
    "70m-deduped",
    "160m-deduped",
    "410m-deduped",
    "1.4b-deduped",
    "1b-deduped",
    "2.8b-deduped",
    "6.9b-deduped",
    "12b-deduped",
    ]

for m in model_ls:
    fpth=target_dir+"model_size-----"+m+"smallres.json"
    with open(fpth, 'r',encoding='utf8') as f:
        data=json.load(f,object_pairs_hook=OrderedDict)

    ## evaluation
    ins,gens=zip(*data)

    s_3ngram=ngram_recall_evaluate(gens,ins,n=3)
    s_4ngram=ngram_recall_evaluate(gens,ins,n=4)
    s_5ngram=ngram_recall_evaluate(gens,ins,n=5)
    s_6ngram=ngram_recall_evaluate(gens,ins,n=6)
    s_7ngram=ngram_recall_evaluate(gens,ins,n=7)
    s_8ngram=ngram_recall_evaluate(gens,ins,n=8)
    fuzzy_rate=fuzzy_match_recall(gens,ins,ratio=80)
    fuzzy_rate=fuzzy_match_recall(gens,ins,ratio=90)
    fuzzy_rate=fuzzy_match_recall(gens,ins,ratio=100)
    
    for sample in data:
        label=sample[0]
        extracted=sample[1]
        if "\n" in extracted:
            es=list(set(extracted.split("\n")))


## running entry
if __name__=="__main__":
    # main()
    print("EVERYTHING DONE.")


