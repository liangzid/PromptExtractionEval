"""
======================================================================
FUNCTION_CALLINGS_DATASET_CONSTRUCTION ---

Instructions with "Function Callings."

    Author: Zi Liang <frost.liang@polyu.edu.hk>
    Copyright © 2023, ZiLiang, all rights reserved.
    Created: 22 November 2023
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

from tqdm import tqdm

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


def arrange_func_callings():
    t = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf")

    # first load the prompts from datasets
    prompts = load_dataset("iohadrubin/api_guru")["train"].to_list()
    # print(prompts[0])
    pls = []
    for p in prompts:
        p = p["text"]
        p = json.loads(p)
        # print(p)
        p = json.dumps(p)
        # print(p)
        pls.append(p)
    prompts = pls
    # print(f"Dataset length (before): {len(prompts)}")
    # prompts=list(set(prompts))
    # print(f"Dataset length (after): {len(prompts)}")
    random.shuffle(prompts)
    # prompts = prompts[:50]

    sub_prompts_stage1 = []
    seqlen_ls = []
    for x in tqdm(prompts):
        seqlen_ls.append(len(x.split(" ")))
        # seqlen_ls.append(len(t(x).input_ids))
        if len(x.split(" ")) < 1000:
            sub_prompts_stage1.append(x)

    # construct the distribution dict
    from collections import Counter

    cc = Counter(seqlen_ls)
    scc = dict(sorted(cc.items(), key=lambda x: x[1]))
    ppp(scc)

    with open("selected_APIs_inStage1.json", 'w', encoding='utf8') as f:
        json.dump(sub_prompts_stage1, f, ensure_ascii=False, indent=4)


def selection_stage2():
    t = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf")

    from collections import OrderedDict
    with open("selected_APIs_inStage1.json", 'r', encoding='utf8') as f:
        data = json.load(f, object_pairs_hook=OrderedDict)

    print(f"Original Selected Prompt Nums: {len(data)}")

    newls = []
    for d in tqdm(data):
        if len(t(d).input_ids) < 1024:
            newls.append(d)

    with open("APIs_short1024_cases.json", 'w', encoding='utf8') as f:
        json.dump(newls, f, ensure_ascii=False, indent=4)

    print(f"New selected Nums: {len(newls)}")


def analysis_and_postprocess_stage3():
    t = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf")
    from collections import OrderedDict
    with open("APIs_short1024_cases.json", 'r', encoding='utf8') as f:
        data = json.load(f, object_pairs_hook=OrderedDict)

    # statistic its length (i.e. num of tokens)
    length_ls = [len(t(x).input_ids) for x in data]

    from collections import Counter
    c = Counter(length_ls)
    c = dict(sorted(c.items(), key=lambda x: x[0]))
    print(c)

    intervals = list(range(200, 1001, 100))
    # print(intervals)
    ins = []
    for i in range(len(intervals)-1):
        ins.append([intervals[i], intervals[i+1]])
    print(ins)

    interval_dict = OrderedDict()
    len_dict = OrderedDict()
    j = 0
    for i, d in enumerate(data):
        for interval in ins:
            if length_ls[i] < interval[1] and length_ls[i] >= interval[0]:
                if interval[0] not in interval_dict:
                    interval_dict[interval[0]] = []
                    len_dict[interval[0]] = []
                interval_dict[interval[0]].append(d)
                len_dict[interval[0]].append(length_ls[i])
                break
    print(len_dict)
    # then show it.
    new_dict = OrderedDict()
    for k in len_dict:
        avg_len = round(sum(len_dict[k])/len(len_dict[k]), 3)
        print(k, len(len_dict[k]))
        new_dict[str(avg_len)] = interval_dict[k]

    print(new_dict)
    print(new_dict.keys())
    print("-----------------")
    print(len_dict)

    normal_dict = obtain_vanilla_prompts_for_function_calling_comparison()
    finally_res = {"normal": normal_dict,
                   "funcall": new_dict, }

    # from collections import OrderedDict

    with open("./function_callling_all_prompts.json",
              'w', encoding='utf8') as f:
        json.dump(finally_res, f, ensure_ascii=False, indent=4)


def obtain_vanilla_prompts_for_function_calling_comparison():
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

    # interval_ls = [0, 16, 32, 64, 128, 256, 512, 768, 1024]
    interval_ls = [200, 300, 400, 500, 600, 700, 800, 900, 1000]
    intervals = []
    for i in range(len(interval_ls)-1):
        intervals.append([interval_ls[i], interval_ls[i+1]])

    per_interval_sample_num = 12

    random.seed(1850)
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
    # with open("Sampled_prompt_dict.json", 'w', encoding='utf8') as f:
    #     json.dump(final_dict, f, ensure_ascii=False, indent=4)
    # print("Save DONEE.")

    return final_dict


# running entry
if __name__ == "__main__":
    # arrange_func_callings()
    # selection_stage2()
    analysis_and_postprocess_stage3()
    # main()
    print("EVERYTHING DONE.")
