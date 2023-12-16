"""
======================================================================
UNBALANCED_CORPUS_WITH_RAW_DISTRIBUTION_PREPROCESS ---

Different from `preprocess_legalLAMA.py`, this file focuses on constructing
fewshowt dataset more similar to real-world exmaple distributions.

    Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
    Copyright © 2023, ZiLiang, all rights reserved.
    Created: 14 November 2023
======================================================================
"""


# ------------------------ Code --------------------------------------

# normal import
import json
from typing import List, Tuple, Dict
import random
from pprint import pprint as ppp


from datasets import load_dataset
from collections import OrderedDict


def getDistributeContractTypes(num_new_dataset=206,
                               save_pth="./contract_types_raw_dist_raw.json"):

    task = "contract_types"
    dataset_name = "lexlms/legal_lama"

    handle_tasks = ["contract_sections",
                    "contract_types",
                    "use_crimes"]

    dataset = load_dataset(dataset_name, name=task)
    dataset = dataset["test"]
    dataset = dataset.shuffle(seed=20231114,)

    print(f"Length of dataset: {len(dataset)}")

    dataset_dict = {}

    # obtain a key distribution
    for s in dataset:
        t = s['text']
        if "<mask> " in t:
            t = t.replace("<mask> ", "")
        if "<mask>" in t:
            t = t.replace("<mask>", "")
        s["text"] = t

        if s['label'] not in dataset_dict.keys():
            dataset_dict[s["label"]] = [s["text"]]
        else:
            dataset_dict[s["label"]].append(s["text"])

    # calculate the distribution
    label_ls = list(dataset_dict.keys())
    nums_ls = []
    for l in label_ls:
        nums_ls.append(len(dataset_dict[l]))
    print(f"distribution of different labels: {nums_ls}")

    print(f">>> NUM of the new dataset: {num_new_dataset}")
    num_per_labels_dict = {}
    for i, l in enumerate(label_ls):
        num_per_labels_dict[l] = int(nums_ls[i]/len(dataset)*num_new_dataset)
    print(f"New distribution of data: {num_per_labels_dict}")

    all_num=0.
    for k,v in num_per_labels_dict.items():
        all_num+=v
    print(f"Selected number: {all_num}")


    ts = {}
    vals = {}
    tes = {}
    for s in dataset:
        t = s["text"]
        l = s["label"]
        if l not in ts:
            ts[l] = []
            vals[l] = []
            tes[l] = []
        if num_per_labels_dict[l] > len(ts[l]):
            ts[l].append(t)
        elif num_per_labels_dict[l] > len(vals[l]):
            vals[l].append(t)
        elif num_per_labels_dict[l] > len(tes[l]):
            tes[l].append(t)
        else:
            pass

    with open(save_pth, 'w', encoding='utf8') as f:
        json.dump({"train": ts,
                   "val": vals,
                   "test": tes,
                   "keys": list(ts.keys())},
                  f, ensure_ascii=False, indent=4)

    print("save done. Save to {}".format(save_pth))


def main():
    # getDistributeContractTypes(100)
    getDistributeContractTypes()
    # getDistributeContractTypes(300)
    from preprocess_legalLAMA import transferToOpenAIFormats
    prefix = "./contract_types_raw_dist_raw.json"
    transferToOpenAIFormats(prefix,"contract_types")
    from format_to_huggingface_dataset import openai_format_to_hugginface_llama_format as tohuggingface

    p1 = prefix+"____openAI_format_train.jsonl"
    p2 = prefix+"____huggingface_format_train.jsonl"
    tohuggingface(p1, p2)

    p1 = prefix+"____openAI_format_val.jsonl"
    p2 = prefix+"____huggingface_format_val.jsonl"
    tohuggingface(p1, p2)

    p1 = prefix+"____openAI_format_test.jsonl"
    p2 = prefix+"____huggingface_format_test.jsonl"
    tohuggingface(p1, p2)


# running entry
if __name__ == "__main__":
    main()
    print("EVERYTHING DONE.")
