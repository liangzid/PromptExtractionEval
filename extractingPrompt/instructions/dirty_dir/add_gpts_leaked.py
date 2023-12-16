"""
======================================================================
ADD_GPTS_LEAKED --- 

    Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
    Copyright © 2023, ZiLiang, all rights reserved.
    Created: 27 November 2023
======================================================================
"""


# ------------------------ Code --------------------------------------

# normal import
import os
import json
from typing import List, Tuple, Dict
import random
from pprint import pprint as ppp


from datasets import load_dataset


glue_prompts = load_dataset("liangzid/prompts")["train"].to_list()
prompt_ls = []
for xxx in glue_prompts:
    prompt_ls.append(xxx["text"])


# new evaluation dataset.
prefix_dir = "./GPTs-prompts/"

new_prompts = []
for filename in os.listdir(prefix_dir):
    pth = prefix_dir+filename
    if os.path.isfile(pth) and ".md" in pth and "README" not in pth:
        with open(pth, 'r', encoding="utf8") as f:
            data = f.read()
            if "```\n" in data:
                data1 = data.split("```\n")[0]
                data = data.split(data1)[1]
                data = data.replace("```", "")
                if data.endswith("\n```"):
                    data=data[:-4]
        new_prompts.append(data)

# save as jsonl
with open("gpts_prompts.jsonl", 'w', encoding='utf8') as f:
    for np in new_prompts:
        nps = json.dumps({"text": np}, ensure_ascii=False)
        f.write(nps+"\n")

print("save done.")


# running entry
if __name__ == "__main__":
    print("EVERYTHING DONE.")
