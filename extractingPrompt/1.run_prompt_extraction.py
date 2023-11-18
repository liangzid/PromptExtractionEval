"""
======================================================================
1.RUN_PROMPT_EXTRACTION --- 

    Author: Zi Liang <frost.liang@polyu.edu.hk>
    Copyright © 2023, ZiLiang, all rights reserved.
    Created: 17 November 2023
======================================================================
"""


# ------------------------ Code --------------------------------------

from test_llama2_extracting import InferPromptExtracting

import torch
torch.cuda.empty_cache()
x = InferPromptExtracting(prompt_dataset="liangzid/glue_prompts",
                          split="validation",
                          device="cuda:0",
                          is_parallel=True)
x.experiment_prompts_and_saves(save_pth="test_save_1.json")

# running entry
if __name__ == "__main__":
    print("EVERYTHING DONE.")
