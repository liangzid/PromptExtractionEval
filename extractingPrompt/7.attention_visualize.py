"""
======================================================================
7.ATTENTION_VISUALIZE ---

after `attention_visulaize.py`

    Author: Zi Liang <frost.liang@polyu.edu.hk>
    Copyright © 2023, ZiLiang, all rights reserved.
    Created: 24 November 2023
======================================================================
"""


# ------------------------ Code --------------------------------------

# normal import
from attention_visualize import compute_metric_of_attentions
from attention_visualize import filter_targeted_samples
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


def draw_2x8():
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

    model.eval()

    selected_layer_head_pairs = [
        (7, 6),
        (3, 31),
        # (6, 12),
        # (3, 26),
    ]

    fig, axs = plt.subplots(2, 8, figsize=(30, 7))
    fig.subplots_adjust(wspace=0.01, hspace=0.7)
    fs = 20

    for i in range(2):
        # for positive
        pos = poss[i]
        device = "cuda:0"
        is_negtive = False
        text = f"Instruction: {pos[0]} User: {pos[1]} Assistant: {pos[2]}"
        inps_p_tokens = tokenizer.tokenize(pos[0])
        text_tokens = tokenizer.tokenize(text)

        input_ids = tokenizer(text,
                              return_tensors="pt",
                              truncation=True).input_ids.to(device)

        attentions = model.forward(input_ids,
                                   # attention_mask=attention_mask,
                                   output_attentions=True).attentions
        for j, (nl, nh) in enumerate(selected_layer_head_pairs):
            per_att = attentions[nl][:, nh, :, :].squeeze().cpu().detach()

            sl = per_att.shape[1]

            res, end_p, bgn_genp, \
                end_genp = compute_metric_of_attentions(text_tokens,
                                                        inps_p_tokens,
                                                        per_att,
                                                        is_negtive=is_negtive
                                                        )
            newlen = min(sl, end_genp+2)
            per_att = per_att[:newlen, :newlen]

            res = axs[0, i*2+j].imshow(per_att,
                                       cmap=plt.cm.Blues,
                                       )
            axs[0, i*2+j].set_xlabel('Attentions From', fontsize=fs)
            axs[0, i*2+j].set_ylabel('Attentions To', fontsize=fs)
            # plt.colorbar(res, ax=axs[i, j])
            axs[0, i*2+j].title.set_text(
                f'Pos. Cases {i+1}\n Layer {nl+1} Head {nh+1}')
            axs[0, i*2+j].title.set_fontsize(fs)

            per_att = per_att[bgn_genp:newlen, :newlen-bgn_genp]

            res = axs[1, i*2+j].imshow(per_att,
                                       cmap=plt.cm.Blues,
                                       extent=[
                                           0, newlen-bgn_genp,
                                           bgn_genp, newlen,
                                               ]
                                       )
            axs[1, i*2+j].set_xlabel('Attentions From', fontsize=fs)
            axs[1, i*2+j].set_ylabel('Attentions To', fontsize=fs)
            axs[1, i*2+j].title.set_text(
                'Generated Part\n Zoom in')
            axs[1, i*2+j].title.set_fontsize(fs)

        # for negtive
        neg = negs[i]
        device = "cuda:1"
        is_negtive = True
        text = f"Instruction: {neg[0]} User: {neg[1]} Assistant: {neg[2]}"
        inps_p_tokens = tokenizer.tokenize(neg[0])
        text_tokens = tokenizer.tokenize(text)

        input_ids = tokenizer(text,
                              return_tensors="pt",
                              truncation=True).input_ids.to(device)

        attentions = model.forward(input_ids,
                                   # attention_mask=attention_mask,
                                   output_attentions=True).attentions
        for j, (nl, nh) in enumerate(selected_layer_head_pairs):
            per_att = attentions[nl][:, nh, :, :].squeeze().cpu().detach()

            sl = per_att.shape[1]

            res, end_p, bgn_genp, \
                end_genp = compute_metric_of_attentions(text_tokens,
                                                        inps_p_tokens,
                                                        per_att,
                                                        is_negtive=is_negtive
                                                        )
            newlen = min(sl, end_genp+2)
            per_att = per_att[:newlen, :newlen]

            res = axs[0, 4+i*2+j]\
                .imshow(per_att,
                        cmap=plt.cm.Blues,
                        )
            axs[0, i*2+j+4].set_xlabel('Attentions From', fontsize=fs)
            axs[0, i*2+j+4].set_ylabel('Attentions To', fontsize=fs)
            # plt.colorbar(res, ax=axs[i, j])
            axs[0, i*2+j+4].title.set_text(
                f'Neg. Cases {i+1}\n Layer {nl+1} Head {nh+1}')
            axs[0, i*2+j+4].title.set_fontsize(fs)

            per_att = per_att[bgn_genp:newlen, :newlen-bgn_genp]

            res = axs[1, 4+i*2+j].imshow(per_att,
                                         cmap=plt.cm.Blues,
                                         extent=[
                                             0, newlen-bgn_genp,
                                             bgn_genp, newlen,
                                             ]
                                         )
            axs[1, 4+i*2+j].set_xlabel('Attentions From', fontsize=fs)
            axs[1, 4+i*2+j].set_ylabel('Attentions To', fontsize=fs)
            # axs[1, 4+i*2+j].title.set_text(
                # f'Neg. Cases {i+1} Local Zoom in\n of Layer {nl+1} Head {nh+1}')
            axs[1, 4+i*2+j].title.set_text(
                f'Generated Part\n Zoom in')
            axs[1, 4+i*2+j].title.set_fontsize(fs)

    plt.savefig(f"./attention_viz/visualization_2x8.pdf",
                pad_inches=0.1)

    print(f"Save to XXX DONE.")


def draw_multiple_imgs():

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

    model.eval()

    selected_layer_head_pairs = [
        (7, 6),
        (3, 31),
        # (6, 12),
        # (3, 26),
    ]

    fig, axs = plt.subplots(3, 4, figsize=(15, 10.5))
    fig.subplots_adjust(wspace=0.01, hspace=0.5)
    fs = 13

    for i in range(2):
        # for positive
        pos = poss[i]
        device = "cuda:0"
        is_negtive = False
        text = f"Instruction: {pos[0]} User: {pos[1]} Assistant: {pos[2]}"
        inps_p_tokens = tokenizer.tokenize(pos[0])
        text_tokens = tokenizer.tokenize(text)

        input_ids = tokenizer(text,
                              return_tensors="pt",
                              truncation=True).input_ids.to(device)

        attentions = model.forward(input_ids,
                                   # attention_mask=attention_mask,
                                   output_attentions=True).attentions
        for j, (nl, nh) in enumerate(selected_layer_head_pairs):
            per_att = attentions[nl][:, nh, :, :].squeeze().cpu().detach()

            sl = per_att.shape[1]

            res, end_p, bgn_genp, \
                end_genp = compute_metric_of_attentions(text_tokens,
                                                        inps_p_tokens,
                                                        per_att,
                                                        is_negtive=is_negtive
                                                        )
            newlen = min(sl, end_genp+2)
            per_att = per_att[:newlen, :newlen]

            res = axs[i, j].imshow(per_att,
                                   cmap=plt.cm.Blues,
                                   )
            axs[i, j].set_xlabel('Attention From', fontsize=fs)
            axs[i, j].set_ylabel('Attention To', fontsize=fs)
            # plt.colorbar(res, ax=axs[i, j])
            axs[i, j].title.set_text(
                f'Pos. Cases {i+1} Layer {nl+1} Head {nh+1}')
            axs[i, j].title.set_fontsize(fs)

        # for negtive
        neg = negs[i]
        device = "cuda:1"
        is_negtive = True
        text = f"Instruction: {neg[0]} User: {neg[1]} Assistant: {neg[2]}"
        inps_p_tokens = tokenizer.tokenize(neg[0])
        text_tokens = tokenizer.tokenize(text)

        input_ids = tokenizer(text,
                              return_tensors="pt",
                              truncation=True).input_ids.to(device)

        attentions = model.forward(input_ids,
                                   # attention_mask=attention_mask,
                                   output_attentions=True).attentions
        for j, (nl, nh) in enumerate(selected_layer_head_pairs):
            per_att = attentions[nl][:, nh, :, :].squeeze().cpu().detach()

            sl = per_att.shape[1]

            res, end_p, bgn_genp, \
                end_genp = compute_metric_of_attentions(text_tokens,
                                                        inps_p_tokens,
                                                        per_att,
                                                        is_negtive=is_negtive
                                                        )
            newlen = min(sl, end_genp+2)
            per_att = per_att[:newlen, :newlen]

            res = axs[i, len(selected_layer_head_pairs)+j]\
                .imshow(per_att,
                        cmap=plt.cm.Blues,
                        )
            axs[i, j+2].set_xlabel('Attention From', fontsize=fs)
            axs[i, j+2].set_ylabel('Attention To', fontsize=fs)
            # plt.colorbar(res, ax=axs[i, j])
            axs[i, j+2].title.set_text(
                f'Neg. Cases {i+1} Layer {nl+1} Head {nh+1}')
            axs[i, j+2].title.set_fontsize(fs)

    # for positive
    pos = poss[1]
    device = "cuda:0"
    is_negtive = False
    text = f"Instruction: {pos[0]} User: {pos[1]} Assistant: {pos[2]}"
    inps_p_tokens = tokenizer.tokenize(pos[0])
    text_tokens = tokenizer.tokenize(text)

    input_ids = tokenizer(text,
                          return_tensors="pt",
                          truncation=True).input_ids.to(device)

    attentions = model.forward(input_ids,
                               # attention_mask=attention_mask,
                               output_attentions=True).attentions
    for j, (nl, nh) in enumerate(selected_layer_head_pairs):
        per_att = attentions[nl][:, nh, :, :].squeeze().cpu().detach()

        sl = per_att.shape[1]

        res, end_p, bgn_genp, \
            end_genp = compute_metric_of_attentions(text_tokens,
                                                    inps_p_tokens,
                                                    per_att,
                                                    is_negtive=is_negtive
                                                    )
        newlen = min(sl, end_genp+2)
        per_att = per_att[:newlen, :newlen]
        per_att = per_att[bgn_genp:newlen, :newlen-bgn_genp]

        res = axs[2, j].imshow(per_att,
                               cmap=plt.cm.Blues,
                               extent=[bgn_genp, newlen, newlen, bgn_genp]
                               )
        axs[2, j].set_xlabel('Attention From', fontsize=fs)
        axs[2, j].set_ylabel('Attention To', fontsize=fs)
        # plt.colorbar(res, ax=axs[i, j])
        axs[2, j].title.set_text(
            f'Pos. Cases {2} Local Zoom in\n of Layer {nl+1} Head {nh+1}')
        axs[2, j].title.set_fontsize(fs)

    # for negtive
    neg = negs[1]
    device = "cuda:1"
    is_negtive = True
    text = f"Instruction: {neg[0]} User: {neg[1]} Assistant: {neg[2]}"
    inps_p_tokens = tokenizer.tokenize(neg[0])
    text_tokens = tokenizer.tokenize(text)

    input_ids = tokenizer(text,
                          return_tensors="pt",
                          truncation=True).input_ids.to(device)

    attentions = model.forward(input_ids,
                               # attention_mask=attention_mask,
                               output_attentions=True).attentions
    for j, (nl, nh) in enumerate(selected_layer_head_pairs):
        per_att = attentions[nl][:, nh, :, :].squeeze().cpu().detach()

        sl = per_att.shape[1]

        res, end_p, bgn_genp, \
            end_genp = compute_metric_of_attentions(text_tokens,
                                                    inps_p_tokens,
                                                    per_att,
                                                    is_negtive=is_negtive
                                                    )
        newlen = min(sl, end_genp+2)
        per_att = per_att[:newlen, :newlen]
        per_att = per_att[bgn_genp:newlen, :newlen-bgn_genp]

        res = axs[2, len(selected_layer_head_pairs)+j]\
            .imshow(per_att,
                    cmap=plt.cm.Blues,
                    extent=[bgn_genp, newlen, newlen, bgn_genp]
                    )
        # axs[2, j+2].set_xticks(range(bgn_genp,newlen,5),range(bgn_genp,newlen,5),)
        axs[2, j+2].set_xlabel('Attention From', fontsize=fs)
        axs[2, j+2].set_ylabel('Attention To', fontsize=fs)
        # plt.colorbar(res, ax=axs[i, j])
        axs[2, j+2].title.set_text(
            f'Neg. Cases {1+1} Local Zoom in\n of Layer {nl+1} Head {nh+1}')
        axs[2, j+2].title.set_fontsize(fs)

    plt.savefig(f"./attention_viz/visualization_4x4.pdf",
                pad_inches=0.1)

    print(f"Save to XXX DONE.")


if __name__ == "__main__":
    # draw_multiple_imgs()
    draw_2x8()
