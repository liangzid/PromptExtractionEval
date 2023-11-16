"""
======================================================================
COLLECT_PUSHTO_HF --- 

    Author: Zi Liang <frost.liang@polyu.edu.hk>
    Copyright © 2023, ZiLiang, all rights reserved.
    Created: 16 November 2023
======================================================================
"""


# ------------------------ Code --------------------------------------

## normal import 
import json
from typing import List,Tuple,Dict
import random
from pprint import pprint as ppp
# import pickle
# import os
# from os.path import join, exists
# from collections import Counter,OrderedDict
# from bisect import bisect
# from copy import deepcopy
# import pickle

# import sys
# # sys.path.append("./")
# from tqdm import tqdm

# import numpy as np

# import argparse
# import logging

from datasets import load_dataset


from collections import OrderedDict
with open("./awsome-overall.json", 'r',encoding='utf8') as f:
    data1=json.load(f,object_pairs_hook=OrderedDict)
    
# from collections import OrderedDict
with open("./promptbench_overall.json", 'r',encoding='utf8') as f:
    data2=json.load(f,object_pairs_hook=OrderedDict)
    
overall=[]
overall.extend(data1)
overall.extend(data2)

## now we should handle the dataset.

prefix="./prompts___"

with open(prefix+"overall.jsonl", 'w',encoding='utf8') as f:
    for x in overall:
        x={"text":x}
        s=json.dumps(x,ensure_ascii=False)
        f.write(s+"\n")

with open(prefix+"promptBench.jsonl","w",encoding="utf8") as f:
    for x in data1:
        x={"text":x}
        s=json.dumps(x,ensure_ascii=False)
        f.write(s+"\n")
    
with open(prefix+"awsome.jsonl","w",encoding="utf8") as f:
    for x in data2:
        x={"text":x}
        s=json.dumps(x,ensure_ascii=False)
        f.write(s+"\n")



## running entry
if __name__=="__main__":
    # main()
    print("EVERYTHING DONE.")


