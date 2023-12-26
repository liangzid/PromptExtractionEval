"""
======================================================================
METRICS ---

Evalute the quality of extracted instructions

    Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
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
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline
)

global_tokenizer = AutoTokenizer.from_pretrained(
    "NousResearch/Llama-2-7b-chat-hf")


def _to_ngram(t, n=3, stride=1):
    t = t.replace("\n\n", " ")
    t = t.replace("\n", " ")
    tls = global_tokenizer.tokenize(t)
    # print(tls)
    # tls = t.split(" ")
    n_gram_ls = []
    bgin_idx = 0
    while bgin_idx+n < len(tls):
        # n_gram_ls.append(" ".join(tls[bgin_idx:bgin_idx+n]))
        ids=global_tokenizer.convert_tokens_to_ids(tls[bgin_idx:bgin_idx+n])
        text=global_tokenizer.decode(ids)
        n_gram_ls.append(text)
        bgin_idx += stride
    return n_gram_ls


to_ngram = _to_ngram


def _ngram_match(gen_p, original_p, n=3, stride=1):
    frags = _to_ngram(gen_p, n=n, stride=stride)
    for frag in frags:
        if frag in original_p:
            return 1
    return 0


def ngram_recall_evaluate(gens, ps, n=3, stride=1):
    assert len(gens) == len(ps)

    hit_num = 0.
    for i, g in enumerate(gens):
        hit_num += _ngram_match(g, ps[i], n, stride)

    # print(f"{n}-gram match_rate:{hit_num/len(gens)}")
    return round(hit_num/len(gens), 3)


def total_match_recall(gens, ps):
    pass


def fuzzy_match_recall(gens, ps, ratio=80):
    from thefuzz import fuzz
    hit_num = 0.
    for i, g in enumerate(gens):
        
        if fuzz.partial_ratio(g, ps[i]) >= ratio:
        # if fuzz.ratio(g, ps[i]) >= ratio:
            # print("++++++++")
            # print(ps[i],g)
            # print("++++++++")
            hit_num += 1
    # print(f"fuzzy hit-rate:{hit_num/len(gens)}")
    return round(hit_num/len(gens), 3)

# note that BLEU-4 is dataset-level evaluation, not sentence-level.


def blue4(gen_p_ls, p_ls):
    p_ls = [[p] for p in p_ls]
    from bleu4 import corpus_bleu
    res = corpus_bleu(gen_p_ls, p_ls)
    print(f"BELU Results: {res}")
    # print("CODE NEED TO DEBUG")
    return res[0][0]


def BERTscore(gens, ps):
    import bert_score as bs
    p, r, f1 = bs.score(gens, ps, lang="en", verbose=True)

    # then average this score into the same one.
    p = torch.mean(p)
    r = torch.mean(r)
    f1 = torch.mean(f1)
    return p, r, f1


def main():
    gens = ["please do this! Can you do this? yes, I can!",
            "What day is it today?",
            "can you understand me?",
            "A C match B C."]

    ps = ["please do not do this! You cannot do that!",
          "What date is it today?",
          "It is difficult to follow you.",
          "A C match B C.",]

    for i in range(3, 18, 3):
        print(ngram_recall_evaluate(gens, ps, n=i))
    print(blue4(gens, ps))
    print(BERTscore(gens, ps))


# running entry
if __name__ == "__main__":
    main()
    print("EVERYTHING DONE.")
