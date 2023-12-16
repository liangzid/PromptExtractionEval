"""
======================================================================
HANDLE_FINETUNING ---

Handle fine-tuning procedure with OpenAI interfaces.

reference: https://platform.openai.com/docs/guides/fine-tuning/preparing-your-dataset

    Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
    Copyright © 2023, ZiLiang, all rights reserved.
    Created: 11 November 2023
======================================================================
"""


# ------------------------ Code --------------------------------------

# normal import
import json
from typing import List, Tuple, Dict
import random
from pprint import pprint as ppp

from openai import OpenAI


class OpenAIClient:
    def __init__(self,):
        self.client = OpenAI()
        print("NOTED: this part of code has not been debug yet.")

    def execute_finetuneJob(self, train_pth, val_pth, use_val_path=0):

        print("Now uploading dataset file.")
        if use_val_path > 0:
            print("we should use val files.")
            val_res = self.client.files.create(
                file=open(val_pth, "rb"),
                purpose="fine-tune",
            )
            print(f"validation set details: {val_res}")

        f_res = self.client.files.create(
            file=open(train_pth, "rb"),
            purpose="fine-tune",
        )
        print("uploading done.")
        print(f"file uploading response: {f_res}")

        file_id = f_res["id"]
        print(f"file id:{file_id}")
        model_name = "gpt-3.5-turbo"
        print("now create the fine-tuning job.")
        if use_val_path > 0:
            ft_res = self.client.fine_tuning.jobs.create(
                training_file=file_id,
                model=model_name,
                hyperparameters={
                    "n_epochs": 3,
                },
                validation_file=val_res["id"]
            )
        else:
            ft_res = self.client.fine_tuning.jobs.create(
                training_file=file_id,
                model=model_name,
                hyperparameters={
                    "n_epochs": 3,
                }
            )

        print(f"fine-tuning situation: {ft_res}")
        return ft_res


def see_finetunes(self, jobid=None):
    res = self.client.fine_tuning.jobs.list()
    print("-------------------------")
    print(res)

    if jobid is not None:
        # client.fine_tuning.jobs.cancel(jobid)
        res = self.client.fine_tuning.jobs.retrieve(jobid)
        print("-=-=-=-=-=-=-=-=-=-=-=-=")
        print(res)
        ress = self.client.fine_tuning.jobs.list_events(
            fine_tuning_job_id=jobid,
            limit=200,
        )
        ppp("LIST of EVENTS: ========================")
        ppp(ress)


def infer(self, model_id, utterance, N_gen=1, temperature=1, top_p=1):
    c = self.client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system",
                "content": "Hello, I'm your law assistant. What can I do for you?"},
            {"role": "user", "content": utterance},
        ],
        n=N_gen,
        temperature=temperature,
        top_p=top_p,
    )
    print(c)
    return [x.message["content"] for x in c.choices]


# running entry
if __name__ == "__main__":
    # main()
    print("EVERYTHING DONE.")
