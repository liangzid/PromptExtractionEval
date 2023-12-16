"""
======================================================================
DOWNLOAD_LLMS --- 

This file only used to download models to cache.

    Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
    Copyright © 2023, ZiLiang, all rights reserved.
    Created: 17 November 2023
======================================================================
"""


# ------------------------ Code --------------------------------------

model_ls = ["lmsys/vicuna-7b-v1.5-16k",
            "microsoft/phi-1_5",
            "NousResearch/Llama-2-7b-chat-hf",
            "Qwen/Qwen-7B-Chat",
            "01-ai/Yi-6B",
            "mistralai/Mistral-7B-Instruct-v0.1",
            "openchat/openchat_3.5"]

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline
)
import concurrent.futures

def load(m):
    try:
        model = AutoModelForCausalLM.from_pretrained(
            m,
            # quantization_config=quant_config,
            device_map="cuda:0"
        )
    except:
        print("ERROR occured.")
    return 0


# 创建 ThreadPoolExecutor 对象
with concurrent.futures.ThreadPoolExecutor() as executor:
    # 提交每个任务到线程池
    futures = [executor.submit(load, x) for x in model_ls]
    # 获取结果
    results = [future.result() for future in futures]

# 打印结果
print(results)

## running entry
if __name__=="__main__":
    print("EVERYTHING DONE.")


