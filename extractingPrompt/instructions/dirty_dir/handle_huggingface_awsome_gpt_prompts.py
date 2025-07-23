"""
======================================================================
HANDLE_HUGGINGFACE_AWSOME_GPT_PROMPTS --- 

    Author: XXXXXXXXXXXXXX
    Copyright © 20xx, XXX, all rights reserved.
    Created: 16 November 20xx
======================================================================
"""









from datasets import load_dataset




dataset=load_dataset("fka/awesome-chatgpt-prompts")

d=dataset["train"]["prompt"]
print(d)

import json
with open("awsome-overall.json", 'w',encoding='utf8') as f:
    json.dump(d,f,ensure_ascii=False,indent=4)
    
