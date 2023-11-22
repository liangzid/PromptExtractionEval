"""
======================================================================
CONSTRUCT_VARYING_SEQUENCE_LENGTH ---

Constructing Dataset varying Sequence Length

    Author: Zi Liang <frost.liang@polyu.edu.hk>
    Copyright © 2023, ZiLiang, all rights reserved.
    Created: 21 November 2023
======================================================================
"""


# ------------------------ Code --------------------------------------

# normal import
import json
from typing import List, Tuple, Dict
import random
from pprint import pprint as ppp

from collections import OrderedDict

from datasets import load_dataset
import torch
import json
from typing import List, Tuple, Dict
import random
from pprint import pprint as ppp

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline
)
import logging

logging.basicConfig(format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.DEBUG)


def obtain_prompts_in_different_interval():
    """
    XXX
    ------
    : 
    ------
    result:
    """

    # we estimate the sequence length based splited word, rather than
    # tokenizer, since there exist differences of tokenizer in different LLMs.
    t = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf")

    # first load the prompts from datasets
    glue_prompts = load_dataset("liangzid/prompts")["train"].to_list()
    prompt_ls = []
    for xxx in glue_prompts:
        prompt_ls.append(xxx["text"])

    # # then load awsome chatgpt list.
    # awsome_prompts = load_dataset(
    #     "fka/awesome-chatgpt-prompts")["train"].to_list()
    # for xxx in awsome_prompts:
    #     prompt_ls.append(xxx["prompt"])

    # 2. now obtain the sequence length distribution of prompt list.
    seqlen_ls = []
    for x in prompt_ls:
        # seqlen_ls.append(len(x.split(" ")))
        seqlen_ls.append(len(t(x).input_ids))

    # construct the distribution dict
    from collections import Counter

    cc = Counter(seqlen_ls)
    scc = dict(sorted(cc.items(), key=lambda x: x[1]))
    ppp(scc)

    interval_ls = [0, 16, 32, 64, 128, 256, 512, 768, 1024]
    intervals = []
    for i in range(len(interval_ls)-1):
        intervals.append([interval_ls[i], interval_ls[i+1]])

    per_interval_sample_num = 12

    random.seed(20231122)
    random.shuffle(prompt_ls)
    sampled_prompt_dict = OrderedDict()
    sampled_x_map = OrderedDict()

    for p in prompt_ls:
        l_p = len(t(p).input_ids)
        for inter in intervals:
            if l_p < inter[1] and l_p >= inter[0]:
                key = f"{inter[0]}-{inter[1]}"
                if key not in sampled_prompt_dict:
                    sampled_prompt_dict[key] = [p]
                    sampled_x_map[key] = [l_p]
                elif len(sampled_prompt_dict[key]) < per_interval_sample_num:
                    sampled_prompt_dict[key].append(p)
                    sampled_x_map[key].append(l_p)
                break

    # now we got the sampled_prompt_dict.
    for k, v in sampled_prompt_dict.items():
        print(k, len(v))  # check the number of it.

    final_dict = OrderedDict()
    for k, v in sampled_prompt_dict.items():
        avg_len = round(sum(sampled_x_map[k])/len(sampled_x_map[k]), 2)
        final_dict[str(avg_len)] = v

    ppp(final_dict)
    with open("Sampled_prompt_dict.json", 'w', encoding='utf8') as f:
        json.dump(final_dict, f, ensure_ascii=False, indent=4)

    print("Save DONEE.")




# running entry
if __name__ == "__main__":
    # main()
    obtain_prompts_in_different_interval()
    print("EVERYTHING DONE.")
