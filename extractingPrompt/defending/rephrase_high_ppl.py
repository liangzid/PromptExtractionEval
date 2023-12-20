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
import json
import sys
from typing import List, Tuple, Dict
import random
from pprint import pprint as ppp

sys.path.append("../")
from metrics_with_LMs import perplexity_llama2_7b


from datasets import load_dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline
)


def higher_rephrasing():
    dn = "liangzid/glue_prompts"
    dataset = load_dataset(dn)['validation'].to_list()
    prompts = []
    for d in dataset:
        prompts.append(d["text"])

    prompts=prompts[:5]

    original_ppl = perplexity_llama2_7b(prompts,
                                        "NousResearch/Llama-2-7b-chat-hf")

    # load model
    model_name = "NousResearch/Llama-2-7b-chat-hf"
    device = "auto"

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

    new_prompts = []
    for p in prompts:
        query = f"User: Please rephrase this sentence into a unfamiliar way: \"{p}\".\nAssistant: "
        newgen = text_gen(query, do_sample=False)[0]["generated_text"]
        newgen = newgen.split(query)[1]
        print(newgen)
        new_prompts.append(newgen)

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
