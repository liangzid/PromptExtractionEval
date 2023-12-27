"""
======================================================================
2.DROPS_OF_DEFENDING ---

evaluate the performance drops of defending prompts

    Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
    Copyright © 2023, ZiLiang, all rights reserved.
    Created: 16 December 2023
======================================================================
"""

# ------------------------ Code --------------------------------------

import sys
sys.path.append("../")
import os
from pprint import pprint as ppp
import random
from typing import List, Tuple, Dict
import json
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline
)
from collections import OrderedDict
import numpy as np
from datasets import load_dataset
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score
from ppl_high2_confusingBeginnings import defense_reshape
from instruction_performance_eval import one_prompt_one_task_one_model as o3
from tqdm import tqdm

# normal import
task_label_map = {
    "cola": {"1": "acceptable", "0": "unacceptable"},
    "mnli": {"1": "neutral", "0": "entailment", "2": "contradiction"},
    "mrpc": {"1": "equivalent", "2": "not_equivalent"},
    "qnli": {"1": "not_entailment", "0": "entailment"},
    "qqp": {"1": "duplicate", "0": "not_duplicate"},
    "rte": {"1": "not_entailment", "0": "entailment"},
    "sst2": {"1": "positive", "0": "negative"},
    "wnli": {"0": "not_entailment", "1": "entailment"},
}


def myeval(task, res):
    predict_ls = []
    label_ls = []
    submap = task_label_map[task]
    sm_r = {v: k for k, v in submap.items()}

    for res_sent, lbl in res:
        hit = 0
        for k in sm_r:
            if k == lbl and lbl in res_sent:
                hit = 1.
            if k != lbl and lbl in res_sent:
                hit = 0.

        predict_ls.append(hit)
        label_ls.append(float(sm_r[lbl]))

    metric_ls = [accuracy_score, precision_score, recall_score, f1_score]
    scores = []
    for m in metric_ls:
        scores.append(m(label_ls, predict_ls))
    return scores

    # then we do evaluation to calculate acc, P, R, and F1


def oneDefense_oneTask_MultipleOriginalPrompts(
    pipeline,
    defense_method="prefix",
        task="cola",
        pls=[],
):

    save_dir = "./glue_res/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    all_res = []
    for iii, p in tqdm(enumerate(pls), desc=f"Task: {task}  defense: {defense_method}"):
        tmppth = save_dir+f"task___{task}-pindex___{iii}.json"
        res = o3(pipeline, p, task, save_pth=tmppth)
        scores = myeval(task, res)
        all_res.append(scores)

    # now split the dataset.
    accls, prels, recls, f1ls = zip(*all_res)
    mean_acc = mean(accls)
    mean_pre = mean(prels)
    mean_rec = mean(recls)
    mean_f1 = mean(f1ls)

    std_acc = std(accls)
    std_pre = std(prels)
    std_rec = std(recls)
    std_f1 = std(f1ls)

    return [mean_acc, mean_pre, mean_rec, mean_f1,
            std_acc, std_pre, std_rec, std_f1]


def mulDefen_mulTask(model_name="NousResearch/Llama-2-7b-chat-hf",
                     device="auto"):

    # construct the generation pipeline
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # quantization_config=quant_config,
        device_map=device,
        # load_in_8bit=True,
        trust_remote_code=True,
        offload_folder="offload",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    # max_length = 128
    max_new_tokens = 15

    text_gen = pipeline(task="text-generation",
                        model=model,
                        tokenizer=tokenizer,
                        max_new_tokens=max_new_tokens)

    # obtain the training datasets.
    prompt_dataset = "liangzid/nlu_prompts"
    split = "train"
    temp_prompts = load_dataset(prompt_dataset)[split].to_list()

    # set experiment tasks
    tasks_we_used = [
        # "cola",

        # "qnli",
        "qqp",
        "rte",
        "sst2",
        "wnli",]
    defenses_methods = ["prefix", "fakeone",
                        "insert", "donot", "locallook"]

    overall_res = OrderedDict()
    for ttt in tasks_we_used:
        overall_res[ttt] = {}
        subset = temp_prompts[0][ttt]  # the prompt list
        if len(subset) > 5:
            subset = subset[:5]

        # ress = oneDefense_oneTask_MultipleOriginalPrompts(
        #     text_gen,
        #     "vanilla",
        #     ttt,
        #     subset
        # )
        # overall_res[ttt]["vanilla"] = ress

        for ddd in defenses_methods:
            newprompts, _ = defense_reshape(subset, method=ddd)
            # print("Newprompts", newprompts)
            ress = oneDefense_oneTask_MultipleOriginalPrompts(
                text_gen,
                ddd,
                ttt,
                newprompts
            )
            overall_res[ttt][ddd] = ress
            # break
        with open(f"./glue_res/FOR-TASK{ttt}.json",
                  'w', encoding='utf8') as f:
            json.dump(overall_res[ttt],
                      f, ensure_ascii=False, indent=4)
    with open(f"./glue_res/overall-performance-res.json",
              'w', encoding='utf8') as f:
        json.dump(overall_res, f, ensure_ascii=False, indent=4)
    print("Save Done..")


def mean(ls):
    return sum(ls)/len(ls)


def std(ls):
    return np.std(ls, ddof=1)


def main():
    mulDefen_mulTask()

    # running entry
if __name__ == "__main__":
    main()
    print("EVERYTHING DONE.")
