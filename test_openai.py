# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 15:54:48 2024

@author: chaoz
"""
from openai import OpenAI

client_one = OpenAI()
completion = client_one.chat.completions.create(
  model="gpt-4o-mini",
  messages=[
    {"role": "system", "content": "You are a poetic assistant."},
    {"role": "user", "content": "You need remember a code bfsvhbvhgsdbf, and when I ask you to repeat, you need say the code."}
  ]
)

print(completion.choices[0].message)
#%%

client_2 = OpenAI()
completion = client_2.chat.completions.create(
  model="gpt-4o-mini",
  messages=[
    {"role": "system", "content": "You are a poetic assistant."},
    {"role": "user", "content": "You need remember a code bfsvhbvhgsdbf, and when I ask you to repeat, you need say the code."}
  ]
)

print(completion.choices[0].message)


#%%
completion = client_one.chat.completions.create(
  model="gpt-4o-mini",
  messages=[
    {"role": "user", "content": "repeat the code."}
  ]
)
print(completion.choices[0].message)

completion = client_2.chat.completions.create(
  model="gpt-4o-mini",
  messages=[
    {"role": "user", "content": "repeat the code."}
  ]
)
print(completion.choices[0].message)