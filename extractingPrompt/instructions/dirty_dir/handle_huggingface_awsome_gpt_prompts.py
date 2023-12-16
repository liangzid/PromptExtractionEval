"""
======================================================================
HANDLE_HUGGINGFACE_AWSOME_GPT_PROMPTS --- 

    Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
    Copyright © 2023, ZiLiang, all rights reserved.
    Created: 16 November 2023
======================================================================
"""









from datasets import load_dataset




dataset=load_dataset("fka/awesome-chatgpt-prompts")

d=dataset["train"]["prompt"]
print(d)

import json
with open("awsome-overall.json", 'w',encoding='utf8') as f:
    json.dump(d,f,ensure_ascii=False,indent=4)
    
