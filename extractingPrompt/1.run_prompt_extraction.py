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
                          device="auto",
                          is_parallel=False)
x.experiment_prompts_and_saves(save_pth="all_models_short32_prompts_1.json")

print(">>>>>>>>>>>>Frist part of generation DONE.")


torch.cuda.empty_cache()

x = InferPromptExtracting(prompt_dataset="liangzid/glue_prompts",
                          split="train",
                          device="auto",
                          is_parallel=False)
x.experiment_prompts_and_saves(save_pth="all_models_zeroshot_prompts_1.json")

print(">>>>>>>>>>>>Second part of generation DONE.")

torch.cuda.empty_cache()

x = InferPromptExtracting(prompt_dataset="liangzid/prompts",
                          split="train",
                          device="auto",
                          is_parallel=False)
x.experiment_prompts_and_saves(save_pth="all_models___overallPrompts_1.json")

print(">>>>>>>>>>>>Third part of generation DONE.")

# running entry
if __name__ == "__main__":
    print("EVERYTHING DONE.")
