"""
======================================================================
INFER ---

Running attacking and inference for fine-tuned llama models.

    Author: Zi Liang <frost.liang@polyu.edu.hk>
    Copyright © 2023, ZiLiang, all rights reserved.
    Created: 12 November 2023
======================================================================
"""


# ------------------------ Code --------------------------------------

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline
)

data_name = "liangzid/legalLAMA_sft_llama_contractualsections"
ckpt = f"./save_models/llama2-7b-ckpt--1112-{data_name}"

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False
)

# Model
base_model = AutoModelForCausalLM.from_pretrained(
    ckpt,
    quantization_config=quant_config,
    device_map={"": 0}
)
# base_model.config.use_cache = False
# base_model.config.pretraining_tp = 1

llama_tokenizer = AutoTokenizer.from_pretrained(ckpt,
                                                trust_remote_code=True,
                                                )
# Generate Text
query = "How do I use the OpenAI API?"
text_gen = pipeline(task="text-generation", model=base_model,
                    tokenizer=llama_tokenizer,
                    max_length=1024)
print("llama checkpoint load done.")


def attack(query):
    # output = text_gen(f"<s>[INST] {query} [/INST]")
    output = text_gen(f"### Human: {query}")
    x = output[0]['generated_text']
    print(x)
    return x


def main():
    attack("hello, are you here?")


# running entry
if __name__ == "__main__":
    main()
    print("EVERYTHING DONE.")
