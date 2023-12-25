"""
======================================================================
OPENAI_INTERFACE --- 

    Author: Zi Liang <frost.liang@polyu.edu.hk>
    Copyright © 2023, ZiLiang, all rights reserved.
    Created: 25 December 2023
======================================================================
"""


# ------------------------ Code --------------------------------------

from openai import OpenAI as oa
client = oa()


def extract_prompt_interface(modelname="gpt-3.5-turbo-1106",
                             prompt="",
                             utter="",
                             ):

    res = client.chat.completions.create(
        model=modelname,
        # prompt=f"Instruction: {prompt}. User: {utter}. System: ",
        messages=[
            {"role": "system", "content": "Instruction: " + prompt},
            {"role": "user", "content": utter}
        ]
    )
    return res.choices[0].message.content


def test_openai(modelname="gpt-3.5-turbo-1106",
                ):
    res = client.chat.completions.create(
        model=modelname,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system",
                "content": "You are a helpful assistant designed to output JSON."},
            {"role": "user", "content": "Who won the world series in 2020?"}
        ]
    )
    return res.choices[0].message.content
