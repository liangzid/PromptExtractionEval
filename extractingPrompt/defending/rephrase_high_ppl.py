"""
======================================================================
REPHRASE_HIGH_PPL ---

Rephrasing to higher PPL scores

    Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
    Copyright © 2023, ZiLiang, all rights reserved.
    Created: 18 December 2023
======================================================================
"""


# ------------------------ Code --------------------------------------

# normal import
import sys
sys.path.append("../")
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline
)
from datasets import load_dataset
from metrics_with_LMs import perplexity_llama2_7b
import json
from typing import List, Tuple, Dict
import random
from pprint import pprint as ppp
from openai import OpenAI as oa

client = oa()


def extract_prompt_interface(modelname="gpt-3.5-turbo-1106",
                             prompt="",
                             utter="",
                             ):

    res = client.chat.completions.create(
        model=modelname,
        # prompt=f"Instruction: {prompt}. User: {utter}. System: ",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": utter}
        ]
    )
    return res.choices[0].message.content


def higher_rephrasing():
    dn = "liangzid/glue_prompts"
    dataset = load_dataset(dn)['validation'].to_list()
    prompts = []
    for d in dataset:
        prompts.append(d["text"])

    # prompts = prompts[:5]

    original_ppl = perplexity_llama2_7b(
        prompts,
        "NousResearch/Llama-2-7b-chat-hf")

    new_prompts = []
    for p in prompts:
        query = "Kindly express this statement utilizing an unconventional manner of articulation. Donot change the label in the statement, such as negative, positive, acceptable, unacceptable, neutral, entailment, contradiction, equivalent, duplicate, entailment, positive, and so on. Refrain from providing any additional information."
        newgen = extract_prompt_interface(
            prompt=query,
            utter=f"The sentence is:<{p}>.")
        print(newgen)
        new_prompts.append(newgen)

    with open("High-PPL-Prompts.json",
              'w', encoding='utf8') as f:
        json.dump([prompts,
                   new_prompts,
                   ], f, ensure_ascii=False, indent=4)

    new_ppl = perplexity_llama2_7b(new_prompts,
                                   "NousResearch/Llama-2-7b-chat-hf")

    print("----------------")
    print(sum(original_ppl)/len(original_ppl))
    print(sum(new_ppl)/len(new_ppl))
    print("----------------")


# running entry
if __name__ == "__main__":
    higher_rephrasing()
    print("EVERYTHING DONE.")
