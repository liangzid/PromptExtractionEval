"""
======================================================================
ALPHA_GAMMA_TOPK_SEARCH_REBUTTAL ---

Search Top-K Alpha and Gamma Values.

    Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
    Copyright © 2025, ZiLiang, all rights reserved.
    Created: 31 March 2025
======================================================================
"""


# ------------------------ Code --------------------------------------
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3" 

# normal import
import json
from typing import List, Tuple, Dict
import random
from pprint import pprint as ppp

from transformers import (
    AutoModelForCausalLM,
    AutoModel,
    PhiModel,
    PhiForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline
)
from tqdm import tqdm

from attention_visualize import filter_targeted_samples

from heapq import nlargest



def obtainTopKFromeDoubleKeyDict(score_dict, K=15):
    all_entries = []
    for b in score_dict:
        for c in score_dict[b]:
            all_entries.append((score_dict[b][c], b, c))

    top_entries = nlargest(K, all_entries)

    return [(b, c) for (d, b, c) in top_entries]


def main_infer():
    file_pth = "./vary_sl/Llama-2-7b-chat-hf-res.json"
    model_name = "NousResearch/Llama-2-7b-chat-hf"

    try:
        searchTopKGammaCur(file_pth, model_name, "cuda:0")
    except Exception as e:
        raise e
        print("error: ", e)

    try:
        model_name = "meta-llama/Llama-2-7b-hf"
        searchTopKGammaCur(file_pth, model_name, "cuda:0")
    except Exception as e:
        print("error: ", e)

    pth = "./vary_sl/phi-1_5-res.json"
    model_name = "microsoft/phi-1_5"
    try:
        searchTopKGammaCur(pth, model_name, "cuda:0")
    except Exception as e:
        print("error: ", e)


def searchTopKGammaCur(filepth, model_name, device="cuda:3",):

    # --- 1. Prepare test cases
    posls, negls = filter_targeted_samples(
        filepth,
        2,
    )

    # --- 2. load models
    if "hi" in model_name:
        model = PhiForCausalLM.from_pretrained(
            model_name,
            device_map=device,
            trust_remote_code=True,
            output_attentions=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,
            trust_remote_code=True,
            output_attentions=True,
        )

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    from attention_visualize import compute_metric_of_attentions

    for i, pos in tqdm(enumerate(posls), sec="Samples"):
        is_negative = False
        if i > 2:
            break

        text = f"Instruction: {pos[0]} User: {pos[1]} Assistant: {pos[2]}"
        inps_p_tokens = tokenizer.tokenize(pos[0])
        text_tokens = tokenizer.tokenize(text)

        input_ids = tokenizer(
            text,
            return_tensors="pt",
            truncation=True
        ).input_ids.to(device)

        attentions = model.forward(input_ids,
                                   # attention_mask=attention_mask,
                                   output_attentions=True).attentions

        score_dict = {}
        for nl in tqdm(range(24)):
            for nh in tqdm(range(32)):
                # for nl,nh in selected_layer_head_pairs:
                if nl not in score_dict:
                    score_dict[nl] = {}
                per_att = attentions[nl][:, nh, :,
                                         :].squeeze().cpu().detach()
                # per_att = per_att*inps.attention_mask
                per_att = per_att.numpy()

                sl = per_att.shape[1]

                res, end_p, bgn_genp, \
                    end_genp = compute_metric_of_attentions(text_tokens,
                                                            inps_p_tokens,
                                                            per_att,
                                                            is_negtive=is_negative
                                                            )
                newlen = min(sl, end_genp+2)
                per_att = per_att[:newlen, :newlen]
                score_dict[nl][nh] = res["alpha_n"]

        # 3. obtain the topK's indices within in `score_dict`.
        index_pair_ls = obtainTopKFromeDoubleKeyDict(score_dict, K=15)

        print("-----------------------------------------------------")
        print(f"{index_pair_ls=}")


if __name__ == "__main__":
    main_infer()
