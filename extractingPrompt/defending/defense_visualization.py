"""
======================================================================
DEFENSE_VISUALIZATION ---

Visualize the results after defending.

    Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
    Copyright © 2023, ZiLiang, all rights reserved.
    Created: 19 December 2023
======================================================================
"""

# ------------------------ Code --------------------------------------
import sys
sys.path.append("../")
from attention_visualize import compute_metric_of_attentions
from attention_visualize import visualizeSampled, visualizeSampled2
from attention_visualize import filter_targeted_samples
from ppl_high2_confusingBeginnings import defense_reshape
import json
from typing import List, Tuple, Dict
import random
from pprint import pprint as ppp
from tqdm import tqdm
import os
from matplotlib import pyplot as plt
from collections import OrderedDict
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


def visualize_cases_experi(methodls=["none", "fakeone", "donot", "prefix"],
                           spth="./attention_viz/cases---"):

    name_map = OrderedDict({
        "none": "Vanilla",
        "donot": "Directly Defense",
        "prefix": "Repeated Prefix",
        "fakeone": "Fake Prompt",
        "insert": "Random Insertion",
        "locallook": "Directly Defense",
    })
    methodls = list(name_map.keys())
    negls = [False,]*len(methodls)
    pth = "../vary_sl/Llama-2-7b-chat-hf-res.json"
    model_name = "NousResearch/Llama-2-7b-chat-hf"
    device = "auto"

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

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    text_gen = pipeline(task="text-generation",
                        model=model,
                        tokenizer=tokenizer,
                        max_length=512)

    poss, negs = filter_targeted_samples(pth, selected_num=30)
    # print(poss)
    pos_dict = {}
    inpp, ap, resp = zip(*poss)
    for i, method in enumerate(methodls):
        if method == "none":
            pos_dict[method] = poss[0]
        else:
            newinpp = defense_reshape(inpp[:1], method=method)[0][0]
            nap = ap
            q = f"Instruction: {newinpp} User: {nap} Assistant: "
            resp = text_gen(q)
            resp = resp[0]["generated_text"]
            resp = resp.split(q)[1]
            pos_dict[method] = (newinpp, nap, resp)
            if newinpp not in resp and inpp[0] not in resp:
                negls[i] = True

    if not os.path.exists("./attention_viz"):
        os.makedirs("./attention_viz")

    with open("./attention_viz/samples.json", 'w', encoding='utf8') as f:
        json.dump(pos_dict, f, ensure_ascii=False, indent=4)

    model.eval()
    fig, axs = plt.subplots(1, 6, figsize=(26, 3.5))
    iii = 0
    for key, pos in pos_dict.items():
        device = "cuda:0"
        text = f"Instruction: {pos[0]} User: {pos[1]} Assistant: {pos[2]}"
        inps_p_tokens = tokenizer.tokenize(pos[0])
        text_tokens = tokenizer.tokenize(text)

        input_ids = tokenizer(text,
                              return_tensors="pt",
                              truncation=True).input_ids.to(device)

        attentions = model.forward(input_ids,
                                   # attention_mask=attention_mask,
                                   output_attentions=True).attentions

        selected_layer_head_pairs = [
            (7, 6),
            (6, 12),
            (3, 26),
            (3, 31),
        ][0]
        score_dict = {}
        nl, nh = selected_layer_head_pairs

        if nl not in score_dict:
            score_dict[nl] = {}
        per_att = attentions[nl][:, nh, :,
                                 :].squeeze().cpu().detach()
        # per_att = per_att*inps.attention_mask
        per_att = per_att.numpy()

        sl = per_att.shape[1]
        is_negtive = negls[iii]

        res, end_p, bgn_genp, \
            end_genp = compute_metric_of_attentions(text_tokens,
                                                    inps_p_tokens,
                                                    per_att,
                                                    is_negtive=is_negtive
                                                    )
        axs[iii].axvline(x=end_p, color="green", linestyle="-")
        axs[iii].axhline(y=bgn_genp, color="red", linestyle="dotted")

        newlen = min(sl, end_genp+2)
        per_att = per_att[:newlen, :newlen]
        score_dict[nl][nh] = res

        res = axs[iii].imshow(per_att,
                              cmap=plt.cm.Blues,
                              )

        lw = 0.8

        fs=23
        axs[iii].set_xlabel('Attention From', fontsize=fs)
        axs[iii].set_ylabel('Attention To', fontsize=fs)
        # plt.colorbar(res, ax=axs[iii])
        axs[iii].title.set_text(f'{name_map[key]}', fontsize=fs)
        iii += 1
    # plt.show()
    plt.savefig(spth+"4-attention-visualize.pdf", pad_inches=0.1)


# running entry
if __name__ == "__main__":
    # main()
    visualize_cases_experi()
    print("EVERYTHING DONE.")
