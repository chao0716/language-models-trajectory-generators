# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 15:54:48 2024

@author: chaoz
"""
from openai import OpenAI



client = OpenAI()

completion = client.chat.completions.create(
  model="gpt-4o-mini",
  messages=[
    {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
    {"role": "user", "content": "Compose a poem that explains the concept of recursion in programming."}
  ]
)

print(completion.choices[0].message)
#%%