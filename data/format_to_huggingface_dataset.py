"""
======================================================================
FORMAT_TO_HUGGINGFACE_DATASET ---

Format existing formats into huggingface's dataset style, and update it.

    Author: Zi Liang <frost.liang@polyu.edu.hk>
    Copyright © 2023, ZiLiang, all rights reserved.
    Created: 12 November 2023
======================================================================
"""


# ------------------------ Code --------------------------------------

# normal import
import json
from typing import List, Tuple, Dict
import random
from pprint import pprint as ppp


def openai_format_to_hugginface_llama_format(pth, save_pth):

    with open(pth, "r", encoding="utf8") as f1:
        with open(save_pth, "w", encoding="utf8") as f2:
            lines = f1.readlines()
            newlines = []
            for l in lines:
                l = l[:-1]  # to remove "\n"
                d = json.loads(l)
                d = d["messages"]
                text = ""
                for turn in d:
                    if turn["role"] == "system":
                        continue
                    elif turn["role"] == "user":
                        text += "### Human: "+turn["content"]
                    elif turn["role"] == "assistant":
                        text += "### Assistant: "+turn["content"]
                    else:
                        print(f"ERROR. unseen roles: {turn['role']}")
                        return -1
                newlines.append(json.dumps({"text": text})+"\n")
            for nl in newlines:
                f2.write(nl)
            print(f"save to {save_pth} DONE.")


def main():
    # prefix = "./ContractSections___fewshot_dataset.json"
    # p1 = prefix+"____openAI_format_train.jsonl"
    # p2 = prefix+"____huggingface_format_train.jsonl"
    # openai_format_to_hugginface_llama_format(p1, p2)

    # prefix = "./ContractSections___fewshot_dataset.json"
    # p1 = prefix+"____openAI_format_val.jsonl"
    # p2 = prefix+"____huggingface_format_val.jsonl"
    # openai_format_to_hugginface_llama_format(p1, p2)

    # prefix = "./ContractSections___fewshot_dataset.json"
    # p1 = prefix+"____openAI_format_test.jsonl"
    # p2 = prefix+"____huggingface_format_test.jsonl"
    # openai_format_to_hugginface_llama_format(p1, p2)

    prefix = "./ContractTypes___fewshot_dataset.json"
    p1 = prefix+"____openAI_format_train.jsonl"
    p2 = prefix+"____huggingface_format_train.jsonl"
    openai_format_to_hugginface_llama_format(p1, p2)

    prefix = "./ContractTypes___fewshot_dataset.json"
    p1 = prefix+"____openAI_format_val.jsonl"
    p2 = prefix+"____huggingface_format_val.jsonl"
    openai_format_to_hugginface_llama_format(p1, p2)

    prefix = "./ContractTypes___fewshot_dataset.json"
    p1 = prefix+"____openAI_format_test.jsonl"
    p2 = prefix+"____huggingface_format_test.jsonl"
    openai_format_to_hugginface_llama_format(p1, p2)

    prefix = "./CrimeCharges___fewshot_dataset.json"
    p1 = prefix+"____openAI_format_train.jsonl"
    p2 = prefix+"____huggingface_format_train.jsonl"
    openai_format_to_hugginface_llama_format(p1, p2)

    prefix = "./CrimeCharges___fewshot_dataset.json"
    p1 = prefix+"____openAI_format_val.jsonl"
    p2 = prefix+"____huggingface_format_val.jsonl"
    openai_format_to_hugginface_llama_format(p1, p2)

    prefix = "./CrimeCharges___fewshot_dataset.json"
    p1 = prefix+"____openAI_format_test.jsonl"
    p2 = prefix+"____huggingface_format_test.jsonl"
    openai_format_to_hugginface_llama_format(p1, p2)



# running entry
if __name__ == "__main__":
    main()
    print("EVERYTHING DONE.")
