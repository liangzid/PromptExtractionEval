"""
======================================================================
MAIN1 --- 

    Author: Zi Liang <frost.liang@polyu.edu.hk>
    Copyright © 2023, ZiLiang, all rights reserved.
    Created: 21 December 2023
======================================================================
"""


# ------------------------ Code --------------------------------------

from datasets import load_dataset
from collections import OrderedDict
import json


def main():

    # a dict of all tasks
    glue_datasets = load_dataset("liangzid/nlu_prompts",)["train"][0]
    promptss = []
    for task in glue_datasets:
        promptss.extend(glue_datasets[task])
    print(f"Length of GLUE prompts: {len(promptss)}")

    # func_callings
    func_callings = "../selected_APIs_inStage1.json"

    # from collections import OrderedDict
    with open(func_callings, 'r', encoding='utf8') as f:
        fcls = json.load(f, object_pairs_hook=OrderedDict)
    print(f"Length of function callings: {len(fcls)}")

    # leaked prompts
    # run ./dirty_dir/add_gpts_leaked.py to obtain the leaked corpus.
    leaked_prompt_path = "../dirty_dir/leaked_prompts.jsonl"
    leaked_ps = []
    with open(leaked_prompt_path,
              'r', encoding='utf8') as f:
        lines = f.readlines()
        for l in lines:
            if l.endswith("\n"):
                l = l[:-1]
            data = json.loads(l)
            data = data["text"]
            leaked_ps.append(data)

    print(f"Length of leaked prompts: {len(leaked_ps)}")
    # awesome prompts
    dataset_name = "fka/awesome-chatgpt-prompts"

    dataset = load_dataset(dataset_name)["train"]

    awe_ls = []
    for line in dataset:
        awe_ls.append(line["prompt"])

    print(f"Length of awesome prompts: {len(awe_ls)}")

    dataset_dict = {"glue": glue_datasets,
                    "api": fcls,
                    "leaked": leaked_ps,
                    "simulated": awe_ls}
    with open("./OVERALL_DATA_BENCHMARK.json",
              'w', encoding='utf8') as f:
        json.dump(dataset_dict, f, ensure_ascii=False, indent=4)
    print("Save DONE.")


# running entry
if __name__ == "__main__":
    main()
    print("EVERYTHING DONE.")
