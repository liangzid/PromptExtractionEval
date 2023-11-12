"""
======================================================================
MAIN --- 

test the finetuning of LLaMA with LORA.
reference: https://colab.research.google.com/drive/1vk8i01apaSp59GVV2yInxOV15QwCwMrg#scrollTo=n5kH6OUjdubg

    Author: Zi Liang <frost.liang@polyu.edu.hk>
    Copyright © 2023, ZiLiang, all rights reserved.
    Created: 12 November 2023
======================================================================
"""


# ------------------------ Code --------------------------------------

# !pip install -q  torch peft==0.4.0 bitsandbytes==0.40.2 transformers==4.31.0 trl==0.4.7 accelerate

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline
)
from peft import LoraConfig
from trl import SFTTrainer

# Dataset
print("download datasets...")
# data_name = "mlabonne/guanaco-llama2-1k"
data_name = "liangzid/legalLAMA_sft_llama_contractualsections"
training_data = load_dataset(data_name, split="train")

# Model and tokenizer names
print("download pre-trained models")
base_model_name = "NousResearch/Llama-2-7b-chat-hf"
refined_model = f"./save_models/llama2-7b-ckpt--1112-{data_name}"

# Tokenizer
llama_tokenizer = AutoTokenizer.from_pretrained(base_model_name,
                                            trust_remote_code=True)
llama_tokenizer.pad_token = llama_tokenizer.eos_token
llama_tokenizer.padding_side = "right"  # Fix for fp16

# Quantization Config
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False
)

# Model
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=quant_config,
    device_map={"": 0}
)
base_model.config.use_cache = False
base_model.config.pretraining_tp = 1

# LoRA Config
peft_parameters = LoraConfig(
    lora_alpha=16, # scale, \alpha/rank=\zelta W
    lora_dropout=0.1,
    r=8, # rank reduced to
    bias="none",
    task_type="CAUSAL_LM"
)

# Training Params
train_params = TrainingArguments(
    output_dir="./train_results",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=25,
    logging_steps=25,
    learning_rate=8e-5,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    # max_steps=10000,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="tensorboard"
)

# Trainer
fine_tuning = SFTTrainer(
    model=base_model,
    train_dataset=training_data,
    peft_config=peft_parameters,
    dataset_text_field="text",
    tokenizer=llama_tokenizer,
    max_seq_length=1024,
    args=train_params
)

# Training
fine_tuning.train()

# Save Model
fine_tuning.model.save_pretrained(refined_model)
llama_tokenizer.save_pretrained(refined_model)


# # Generate Text
# query = "How do I use the OpenAI API?"
# text_gen = pipeline(task="text-generation", model=refined_model, tokenizer=llama_tokenizer, max_length=200)
# output = text_gen(f"<s>[INST] {query} [/INST]")
# print(output[0]['generated_text'])
