"""
======================================================================
7.ATTENTION_VISUALIZE ---

Visualization of the attention matrices, to 

    Author: Zi Liang <frost.liang@polyu.edu.hk>
    Copyright © 2023, ZiLiang, all rights reserved.
    Created: 24 November 2023
======================================================================
"""


# ------------------------ Code --------------------------------------

# normal import
import json
from typing import List, Tuple, Dict
import random
from pprint import pprint as ppp
import os

from datasets import load_dataset
import torch
import json
from typing import List, Tuple, Dict
import random
from pprint import pprint as ppp
from collections import OrderedDict

import torch
from tqdm import tqdm
from test_llama2_extracting import InferPromptExtracting
import json
import numpy as np
from matplotlib import pyplot as plt

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
import logging
from metrics import fuzzy_match_recall, ngram_recall_evaluate

torch.cuda.empty_cache()

# logging.basicConfig(format='%(asctime)s %(message)s',
#                     datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.DEBUG)


def filter_targeted_samples(pth, selected_num=12):

    pos_ls = []
    neg_ls = []

    # from collections import OrderedDict
    with open(pth, 'r', encoding='utf8') as f:
        data = json.load(f, object_pairs_hook=OrderedDict)
    for ap in data:
        intervals = data[ap]
        for inter in intervals:
            pps = intervals[inter]
            for pair in pps:
                p, gen_p = pair
                if fuzzy_match_recall([gen_p], [p], ratio=100) == 1:
                    if len(p.split(" ")) > 15:
                        continue
                    pos_ls.append((p, ap, gen_p))
                elif fuzzy_match_recall([gen_p], [p], ratio=20) == 0:
                    if len(p.split(" ")) > 150:
                        continue
                    neg_ls.append((p, ap, gen_p))

    random.seed(11242113)
    random.shuffle(pos_ls)
    random.shuffle(neg_ls)

    if len(pos_ls) > selected_num:
        pos_ls = pos_ls[selected_num:]
    if len(neg_ls) > selected_num:
        neg_ls = neg_ls[selected_num:]

    return pos_ls, neg_ls


def visualize_attention_matrix(model, tokenizer, text,
                               inps_p_tokens,
                               text_tokens,
                               device,
                               pth="res.pdf"):

    model.eval()
    inps = tokenizer(text,
                     return_tensors="pt",
                     truncation=True).to(device)
    inps.attention_mask = torch.triu(torch.ones(inps.input_ids.shape[1],
                                                inps.input_ids.shape[1],
                                                )).unsqueeze(0).to(device)

    print("-----------------")
    print(inps)
    print("-----------------")

    # outputs = model(ids, output_attentions=True)
    # print(outputs)
    attentions = model.forward(**inps,
                               output_attentions=True).attentions

    # # shape of attentions: [num_layers, batchsize, num_heads, sl, sl]
    # print(len(attentions))
    # attentions = (attentions[1][:, 30:, :, :], attentions[23][:, 30:, :, :])

    # print(attentions)

    # attention_weights = attentions[0].cpu().squeeze().detach().numpy()
    # attention_weights = np.transpose(attention_weights, (1, 2, 0))

    n_layer = len(attentions)  # 24
    n_head = attentions[0].shape[1]  # 3 2
    score_dict = {}
    for nl in tqdm(range(n_layer)):
        score_dict[nl] = {}
        for nh in range(n_head):
            per_att = attentions[nl][:, nh, :,
                                     :].squeeze().cpu().detach().numpy()

            score_dict[nl][nh] = compute_metric_of_attentions(text_tokens,
                                                              inps_p_tokens,
                                                              per_att,
                                                              )
            fig, axs = plt.subplots(1, 1, figsize=(7, 7))
            res = axs.imshow(per_att, cmap=plt.cm.Blues,
                             # interpolation="nearest"
                             )
            axs.set_xlabel('Attention From')
            axs.set_ylabel('Attention To')
            plt.colorbar(res, ax=axs)
            axs.title.set_text(f'Layer {nl+1} Head {nh+1}')
            plt.savefig(pth+f"layer{nl}_head{nh}.pdf",
                        pad_inches=0.1)
            print(f"Save to {pth}layer{nl}_head{nh}.pdf DONE.")
    with open(f"{pth}metricsRes_layer{nl}_head{nh}.json",
              'w', encoding='utf8') as f:
        json.dump(score_dict, f, ensure_ascii=False, indent=4)


def visualizeSampled(model, tokenizer, text,
                     inps_p_tokens,
                     text_tokens,
                     device,
                     pth="res.pdf"):

    model.eval()
    inps = tokenizer(text,
                     return_tensors="pt",
                     truncation=True).to(device)

    sl = inps.input_ids.shape[1]

    # attention_mask = torch.triu(torch.ones(sl,
    #                                        sl,
    #                                        )).to(device).unsqueeze(0).unsqueeze(0)
    # print(attention_mask, inps.input_ids)
    # print(attention_mask.shape, inps.input_ids.shape)
    input_ids=inps.input_ids
    attentions = model.forward(input_ids,
                               # attention_mask=attention_mask,
                               output_attentions=True).attentions

    # # shape of attentions: [num_layers, batchsize, num_heads, sl, sl]
    # print(len(attentions))
    # attentions = (attentions[1][:, 30:, :, :], attentions[23][:, 30:, :, :])

    selected_layer_head_pairs = [
        (5, 4),
        (5, 13),
        (6, 9),
        (6, 16),
        (7, 6),
        (8, 29),
    ]

    score_dict = {}
    for nl, nh in tqdm(selected_layer_head_pairs):
        if nl not in score_dict:
            score_dict[nl] = {}
        per_att = attentions[nl][:, nh, :,
                                 :].squeeze().cpu().detach()
        # per_att = per_att*inps.attention_mask
        per_att = per_att.numpy()

        score_dict[nl][nh] = compute_metric_of_attentions(text_tokens,
                                                          inps_p_tokens,
                                                          per_att,
                                                          )
        fig, axs = plt.subplots(1, 1, figsize=(7, 7))
        res = axs.imshow(per_att, cmap=plt.cm.Blues,
                         # interpolation="nearest"
                         )
        axs.set_xlabel('Attention From')
        axs.set_ylabel('Attention To')
        plt.colorbar(res, ax=axs)
        axs.title.set_text(f'Layer {nl+1} Head {nh+1}')
        plt.savefig(pth+f"layer{nl+1}_head{nh+1}.pdf",
                    pad_inches=0.1)
        print(f"Save to {pth}layer{nl+1}_head{nh+1}.pdf DONE.")
    with open(f"{pth}metricsRes_layer{nl+1}_head{nh+1}.json",
              'w', encoding='utf8') as f:
        json.dump(score_dict, f, ensure_ascii=False, indent=4)
    print(f"Save to `{pth}metricsRes_layer{nl+1}_head{nh+1}.json` DONE.")


# only for good cases
def compute_metric_of_attentions(tokens, inp_p_tokens, atts):
    """
    tokens: a list of token, len(tokens)=sl
    shape of atts: [sl, sl]
    """
    # at first, find the beginning token of prompts.

    idx_first_token_in_inps = -1
    for i, t in enumerate(tokens):
        if t == inp_p_tokens[0] and tokens[i+1] == inp_p_tokens[1]:
            idx_first_token_in_inps = i
            break
    assert idx_first_token_in_inps != -1
    idx_last_token_in_inps = -1
    for i, t in enumerate(tokens):
        if t == " User":
            idx_last_token_in_inps = i

    idxes_prompts = [idx_first_token_in_inps,
                     idx_last_token_in_inps]

    offset = 0
    while True:
        if tokens[idx_last_token_in_inps+offset] == inp_p_tokens[0]\
           and tokens[idx_last_token_in_inps+offset +
                      len(inp_p_tokens)-1] == inp_p_tokens[-1]:
            break
        offset += 1
        if offset > 10000:
            return -5

    # idxes_user_attacks = [-1, -1]
    # idxes_system_responses = [-1, -1]
    idxes_system_gen_p = [idx_last_token_in_inps+offset,
                          idx_last_token_in_inps+offset+len(inp_p_tokens)]
    print(f"idxes_prmpts: {idxes_prompts}")
    print(f"idxes_sysgen_prmpts: {idxes_system_gen_p}")

    # calculate \alpha_p
    alpha_p = 0.
    alpha_n = 0.

    beta_p = 0.
    gammar_p = 0.
    gammar_n = 0.
    num = 0
    for i, ii in enumerate(list(range(idxes_system_gen_p[0]+1,
                                      idxes_system_gen_p[1]))):
        idx_its_preivous_token_in_inps = idxes_prompts[0]+i-1
        att_valuep = atts[ii, idx_its_preivous_token_in_inps]
        alpha_p += att_valuep

        idx_its_current_token_in_inps = idxes_prompts[0]+i
        att_valuen = atts[ii, idx_its_current_token_in_inps]
        alpha_n += att_valuen

        gens_token_preivous = ii-1
        beta_p += atts[ii, gens_token_preivous]
        num += 1

        gammar_sum = sum(atts[ii,
                              idxes_prompts[0]:idxes_prompts[1]]
                         .tolist())

        gammar_p += att_valuep/gammar_sum
        gammar_n += att_valuen/gammar_sum

    alpha_p /= num
    alpha_n /= num
    beta_p = beta_p/num + alpha_p

    gammar_p /= num
    gammar_n /= num

    res = {"alpha_p": alpha_p, "alpha_n": alpha_n,
           "beta_p": beta_p,
           "gammar_p": gammar_p, "gammar_n": gammar_n}
    return res


def main1():
    pth = "./vary_sl/Llama-2-7b-chat-hf-res.json"
    model_name = "NousResearch/Llama-2-7b-chat-hf"

    # pth = "./vary_sl/phi-1_5-res.json"
    # model_name = "microsoft/phi-1_5"

    # device = "cuda:0"
    device = "auto"

    poss, negs = filter_targeted_samples(pth, selected_num=30)

    # model = AutoModelForCausalLM.from_pretrained(
    #     model_name,
    #     device_map=device,
    #     trust_remote_code=True,
    #     output_attentions=True,
    # )

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

    # model = AutoModelForCausalLM.from_pretrained(

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if not os.path.exists("./attention_viz"):
        os.makedirs("./attention_viz")

    with open("./attention_viz/samples.json", 'w', encoding='utf8') as f:
        json.dump([poss, negs], f, ensure_ascii=False, indent=4)

    for i, pos in tqdm(enumerate(poss), desc="Samples"):
        if i > 0:
            break
        text = f"Instruction: {pos[0]} User: {pos[1]} Assistant: {pos[2]}"
        inps_p_tokens = tokenizer.tokenize(pos[0])
        text_tokens = tokenizer.tokenize(text)
        # print(ids)
        pth = f"./attention_viz/Sampled__POSITIVE_{i}_img---"
        # visualize_attention_matrix(model,
        #                            tokenizer,
        #                            text,
        #                            inps_p_tokens,
        #                            text_tokens,
        #                            "cuda:0",
        #                            pth=pth,
        #                            )
        visualizeSampled(model,
                         tokenizer,
                         text,
                         inps_p_tokens,
                         text_tokens,
                         "cuda:0",
                         pth=pth,
                         )

    for i, neg in tqdm(enumerate(negs), desc="Samples"):
        if i > 0:
            break
        text = f"Instruction: {neg[0]} User: {neg[1]} Assistant: {neg[2]}"
        inps_p_tokens = tokenizer.tokenize(neg[0])
        text_tokens = tokenizer.tokenize(text)
        # print(ids)
        pth = f"./attention_viz/Sampled__NEGIVE_{i}_img---"
        # visualize_attention_matrix(model,
        #                            tokenizer,
        #                            text,
        #                            inps_p_tokens,
        #                            text_tokens,
        #                            "cuda:0",
        #                            pth=pth,
        #                            )

        visualizeSampled(model,
                         tokenizer,
                         text,
                         inps_p_tokens,
                         text_tokens,
                         "cuda:0",
                         pth=pth,
                         )


# running entry
if __name__ == "__main__":
    # main()
    main1()
    print("EVERYTHING DONE.")
