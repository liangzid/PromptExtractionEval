"""
======================================================================
CHATDEEPSEEK --- 

    Author: XXXXXXXXXXXx
    Copyright © 2025, XXXXX, all rights reserved.
    Created: 28 March 2025
======================================================================
"""


# ------------------------ Code --------------------------------------

from openai import OpenAI
import json
import os

apikey=os.environ["DEEPSEEK_API_KEY"]

client = OpenAI(
    api_key=apikey,
    base_url="https://api.deepseek.com",
)


def onetimequery(
    system_prompt,
    user_prompt,
):

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
    )
    adict = response.choices[0].message.content
    print(adict)
    return adict
