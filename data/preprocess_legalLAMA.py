"""
======================================================================
PREPROCESS_LEGALLAMA ---

extract subtasks based on LegalLAMA benchmark.

    Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
    Copyright © 2023, ZiLiang, all rights reserved.
    Created: 10 November 2023
======================================================================
"""


# ------------------------ Code --------------------------------------

# normal import
import json
from typing import List, Tuple, Dict
import random
from pprint import pprint as ppp

from datasets import load_dataset
from collections import OrderedDict


dataset_name = "lexlms/legal_lama"

handle_tasks = ["contract_sections",
                "contract_types",
                "use_crimes"]


def handle_contractSections(save_pth="./ContractSections___fewshot_dataset.json",per_num=5):
    task = "contract_sections"

    dataset = load_dataset(dataset_name, name=task)
    dataset = dataset["test"]
    dataset = dataset.shuffle(seed=8263,)

    """
    shapes as:
    test: Dataset({
        features: ['text', 'label', 'category'],
        num_rows: 1527

   })

    it consists of 20 labels. We select 5 sample per label in train set,
    and 5 for validation, and 5 for test, respectively.
    """

    section_keyls = []
    train_set = {}
    val_set = {}
    test_set = {}

    per_num = per_num
    for sample in dataset:
        t = sample["text"]

        if "10. <mask>:\n" in t:
            t = t.replace("10. <mask>:\n", "")
        if "<mask>" in t:
            t.replace("<mask>", "")
        if sample["label"] not in section_keyls:

            section_keyls.append(sample["label"])
            train_set[sample["label"]] = [t]
            val_set[sample["label"]] = []
            test_set[sample["label"]] = []
        else:
            if len(train_set[sample["label"]]) < per_num:
                train_set[sample["label"]].append(t)
            elif len(val_set[sample["label"]]) < per_num:
                val_set[sample["label"]].append(t)
            elif len(test_set[sample["label"]]) < per_num:
                test_set[sample["label"]].append(t)
            else:
                pass

    # add number check to filter some rare categories.
    checked_rare_keys = []
    for k in train_set.keys():
        if len(train_set[k]) < per_num:
            checked_rare_keys.append(k)
            print(
                f">> Category {k} only {len(train_set[k])} found. Now delete it.")

    print(f"Overall delete keys: {checked_rare_keys}")
    for k in checked_rare_keys:
        del train_set[k]
        del val_set[k]
        del test_set[k]
        section_keyls.remove(k)

    with open(save_pth, 'w', encoding='utf8') as f:
        json.dump({"train": train_set,
                   "val": val_set,
                   "test": test_set,
                   "keys": section_keyls},
                  f, ensure_ascii=False, indent=4)
    print("save done. Save to {}".format(save_pth))


def handle_contractTypes(save_pth="./ContractTypes___fewshot_dataset.json",per_num=6):
    task = "contract_types"

    dataset = load_dataset(dataset_name, name=task)
    dataset = dataset["test"]
    dataset = dataset.shuffle(seed=2023,)

    """
    shapes as:
    DatasetDict({
        test: Dataset({
            features: ['text', 'label', 'category'],
            num_rows: 1062
        })
    })

    it consists of 15 labels. We select 6 sample per label in train set,
    and 6 for validation, and 6 for test, respectively.
    """

    section_keyls = []
    train_set = {}
    val_set = {}
    test_set = {}

    per_num = per_num
    for sample in dataset:
        t = sample["text"]
        if "<mask> " in t:
            t = t.replace("<mask> ", "")
        if "<mask>" in t:
            t = t.replace("<mask>", "")
        sample['text'] = t
        if sample["label"] not in section_keyls:
            section_keyls.append(sample["label"])
            train_set[sample["label"]] = [sample["text"]]
            val_set[sample["label"]] = []
            test_set[sample["label"]] = []
        else:
            if len(train_set[sample["label"]]) < per_num:
                train_set[sample["label"]].append(sample["text"])
            elif len(val_set[sample["label"]]) < per_num:
                val_set[sample["label"]].append(sample["text"])
            elif len(test_set[sample["label"]]) < per_num:
                test_set[sample["label"]].append(sample["text"])
            else:
                pass

    # add number check to filter some rare categories.
    checked_rare_keys = []
    for k in train_set.keys():
        if len(train_set[k]) < per_num:
            checked_rare_keys.append(k)
            print(
                f">> Category {k} only {len(train_set[k])} found. Now delete it.")

    print(f"Overall delete keys: {checked_rare_keys}")
    for k in checked_rare_keys:
        del train_set[k]
        del val_set[k]
        del test_set[k]
        section_keyls.remove(k)

    with open(save_pth, 'w', encoding='utf8') as f:
        json.dump({"train": train_set,
                   "val": val_set,
                   "test": test_set,
                   "keys": section_keyls},
                  f, ensure_ascii=False, indent=4)

    print("save done. Save to {}".format(save_pth))


def handle_crimeCharges(save_pth="./CrimeCharges___fewshot_dataset.json"):
    task = "us_crimes"

    dataset = load_dataset(dataset_name, name=task)
    dataset = dataset["test"]
    dataset = dataset.shuffle(seed=1110,)

    """
    shapes as:
    DatasetDict({
        test: Dataset({
            features: ['text', 'label', 'category'],
            num_rows: 1062
        })
    })

    it consists of 59 labels. We select 5 sample per label in train set,
    and 6 for validation, and 6 for test, respectively.
    """

    section_keyls = []
    train_set = {}
    val_set = {}
    test_set = {}

    per_num = 5
    for sample in dataset:

        t = sample["text"]
        if "<mask>" in t:
            t = t.replace("<mask>", "it")

        if sample["label"] not in section_keyls:
            section_keyls.append(sample["label"])
            train_set[sample["label"]] = [sample["text"]]
            val_set[sample["label"]] = []
            test_set[sample["label"]] = []
        else:
            if len(train_set[sample["label"]]) < per_num:
                train_set[sample["label"]].append(sample["text"])
            elif len(val_set[sample["label"]]) < per_num:
                val_set[sample["label"]].append(sample["text"])
            elif len(test_set[sample["label"]]) < per_num:
                test_set[sample["label"]].append(sample["text"])
            else:
                pass

    # add number check to filter some rare categories.
    checked_rare_keys = []
    for k in train_set.keys():
        if len(train_set[k]) < per_num:
            checked_rare_keys.append(k)
            print(
                f">> Category {k} only {len(train_set[k])} found. Now delete it.")

    print(f"Overall delete keys: {checked_rare_keys}")
    for k in checked_rare_keys:
        del train_set[k]
        del val_set[k]
        del test_set[k]
        section_keyls.remove(k)

    with open(save_pth, 'w', encoding='utf8') as f:
        json.dump({"train": train_set,
                   "val": val_set,
                   "test": test_set,
                   "keys": section_keyls},
                  f, ensure_ascii=False, indent=4)
    print("save done. Save to {}".format(save_pth))


def transferToOpenAIFormats(pth, subset_name, greeting_key="2", save_pth_prefix=None):
    if save_pth_prefix is None:
        save_pth_prefix = pth+"____openAI_format_"

    # from collections import OrderedDict
    with open(pth, 'r', encoding='utf8') as f:
        data = json.load(f, object_pairs_hook=OrderedDict)
    ts = data["train"]
    vs = data["val"]
    tes = data["test"]
    ks = data["keys"]

    nts = []
    nvs = []
    ntes = []

    example_text = "In New York, no provision of the law specifically and expressly addresses itself to it.” However, the Martin Act, which is incorporated in article 23-A of the General Business Law, states that it is “illegal and prohibited for any person * * * to promote, offer or grant participation in a chain distributor scheme.” (General Business Law § 359-fff [1].) The Act, which clearly includes chain-linked pyramid schemes, defines a chain distributor scheme as: “a sales device whereby a person, upon condition that he make an investment, is *857granted a license or right to solicit or recruit for profit or economic gain one or more additional persons who are also granted such license or right upon condition of making an investment and may further perpetuate the chain of persons who are granted such license or right upon such condition.” (General Business Law § 359-fff [2].)"

    greetings = {"1": f"Hey! I'm your intelligent law assitant. Now I support three tasks, `contractural section name generation`, `contract types inference`, and `crimes charges`! Give me a task name and your texts, I will give you the results!\n Example: `Help me to do crimes charges for this case: {example_text}`",
                 "2": f"Hello, I'm your law assistant. What can I do for you?", }

    ntasks = {"contract_sections": "contractural section name generation",
              "contract_types": "contract types inference",
              "us_crimes": "crimes charges", }
    vtasks = {"contract_sections": " gnerate the section name of this part of constract",
              "contract_types": "give me the types of this contract",
              "us_crimes": "judge the crime of this legal case", }

    ntask_name = ntasks[subset_name]
    vtask_name = vtasks[subset_name]

    for ky in ts.keys():
        ss = ts[ky]
        for s in ss:
            users_similated_prompts = {"1": f"{ntask_name}: {s}",
                                       "2": f"Hello. Could you please help me" +f"to do {ntask_name}? My legal case is: {s}",
                                       "3": f"Please {vtask_name}: {s}",
                                       "4": f"{s}  {vtask_name} for me."}
            agent_similateds = {"1": f"It seems like to be {ky}.",
                                "2": f"{ky}.",
                                "3": f"{ky}. Are there any other thing I can help you with?"}
            user_key = str(random.randint(1, 4))
            resp_key = str(random.randint(1, 3))
            nts.append({"messages": [
                {"role": "system",
                 "content": greetings[greeting_key]},
                {"role": "user",
                 "content": users_similated_prompts[user_key]},
                {"role": "assistant",
                 "content": agent_similateds[resp_key]},
            ]})
    with open(save_pth_prefix+"train.jsonl", 'w', encoding='utf8') as f:
        for x in nts:
            sss = json.dumps(x, ensure_ascii=False)+"\n"
            # print(sss)
            f.write(sss)
        print(f"save done. Save to {save_pth_prefix}train.jsonl")

    for ky in vs.keys():
        ss = vs[ky]
        for s in ss:
            users_similated_prompts = {"1": f"{ntask_name}: {s}",
                                       "2": f"Hello. Could you please help me" +
                                       f"to do {ntask_name}? My legal case is: {s}",
                                       "3": f"Please {vtask_name}: {s}",
                                       "4": f"{s}  {vtask_name} for me."}
            agent_similateds = {"1": f"It seems like to be {ky}.",
                                "2": f"{ky}.",
                                "3": f"{ky}. Are there any other thing I can help you with?"}
            user_key = str(random.randint(1, 4))
            resp_key = str(random.randint(1, 3))
            nvs.append({"messages": [
                {"role": "system",
                 "content": greetings[greeting_key]},
                {"role": "user",
                 "content": users_similated_prompts[user_key]},
                {"role": "assistant",
                 "content": agent_similateds[resp_key]},
            ]})
    with open(save_pth_prefix+"val.jsonl", 'w', encoding='utf8') as f:
        for x in nvs:
            sss = json.dumps(x, ensure_ascii=False)+"\n"
            f.write(sss)
        print(f"save done. Save to {save_pth_prefix}val.jsonl")

    for ky in tes.keys():
        ss = tes[ky]
        for s in ss:
            users_similated_prompts = {"1": f"{ntask_name}: {s}",
                                       "2": f"Hello. Could you please help me to do {ntask_name}? My legal case is:{s}",
                                       "3": f"Please {vtask_name}: {s}",
                                       "4": f"{s}  {vtask_name} for me."}
            agent_similateds = {"1": f"It seems like to be {ky}.",
                                "2": f"{ky}.",
                                "3": f"{ky}. Are there any other thing I can help you with?"}
            user_key = str(random.randint(1, 4))
            resp_key = str(random.randint(1, 3))
            ntes.append({"messages": [
                {"role": "system",
                 "content": greetings[greeting_key]},
                {"role": "user",
                 "content": users_similated_prompts[user_key]},
                {"role": "assistant",
                 "content": agent_similateds[resp_key]},
            ]})
    with open(save_pth_prefix+"test.jsonl", 'w', encoding='utf8') as f:
        for x in ntes:
            sss = json.dumps(x, ensure_ascii=False)+"\n"
            f.write(sss)
        print(f"save done. Save to {save_pth_prefix}test.jsonl")


def main():
    # handle_contractSections()
    # handle_contractTypes()
    # handle_crimeCharges()

    # pnum=1
    # handle_contractTypes(per_num=pnum,save_pth="./contracttypes_small15.json")

def main2():
    # transferToOpenAIFormats("./ContractSections___fewshot_dataset.json",
    #                         "contract_sections",)
    # transferToOpenAIFormats("./ContractTypes___fewshot_dataset.json",
    #                        "contract_types")
    # transferToOpenAIFormats("./CrimeCharges___fewshot_dataset.json",
    #                         "us_crimes")

    # transferToOpenAIFormats("./contracttypes_small15.json",
    #                         "contract_types")


# running entry
if __name__ == "__main__":
    main()
    main2()
    print("EVERYTHING DONE.")
