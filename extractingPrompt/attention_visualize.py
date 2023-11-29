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


def visualize_attention_matrix(model, tokenizer, text, device,
                               pth="res.pdf"):

    model.eval()
    inps = tokenizer(text,
                     return_tensors="pt",
                     truncation=True).to(device)

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
    for nl in range(n_layer):
        for nh in range(n_head):
            fig, axs = plt.subplots(1, 1, figsize=(7, 7))
            per_att = attentions[nl][:, nh, :,
                                     :].squeeze().cpu().detach().numpy()
            res = axs.imshow(per_att, cmap=plt.cm.Blues,
                             # interpolation="nearest"
                             )
            axs.set_xlabel('Attention From')
            axs.set_ylabel('Attention To')
            plt.colorbar(res, ax=axs[nl][nh])
            axs.title.set_text(f'Layer {nl+1} Head {nh+1}')
            plt.savefig(pth+f"layer{nl}_head{nh}.pdf",
                        pad_inches=0.1)
            print(f"Save to {pth}layer{nl}_head{nh}.pdf DONE.")


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

    for i, pos in enumerate(poss):
        text = f"Instruction: {pos[0]} User: {pos[1]} Assistant: {pos[2]}"
        # print(ids)
        pth = f"./attention_viz/{i}_img---"
        visualize_attention_matrix(model,
                                   tokenizer,
                                   text,
                                   "cuda:0",
                                   pth=pth,
                                   )


# running entry
if __name__ == "__main__":
    # main()
    main1()
    print("EVERYTHING DONE.")
