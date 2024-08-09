from PIL import Image
from lang_sam import LangSAM
from main_prompt import MAIN_PROMPT
from lsam_fun import display_image_with_masks, print_bounding_boxes, print_detected_phrases, print_logits, display_image_with_boxes
import sys

# ee_start_position = [0.0, 0.6, 0.55]
# command = 'place the apple in the bowl'

# new_prompt = MAIN_PROMPT.replace("[INSERT EE POSITION]", str(ee_start_position)).replace("[INSERT TASK]", command)


# def get_chatgpt_output(model, new_prompt, messages, role, file=sys.stdout):

#     print(role + ":", file=file)
#     print(new_prompt, file=file)
#     messages.append({"role":role, "content":new_prompt})

#     client = OpenAI()

#     completion = client.chat.completions.create(
#         model=model,
#         temperature=0,
#         messages=messages,
#         stream=True
#     )

#     print("assistant:", file=file)

#     new_output = ""

#     for chunk in completion:
#         chunk_content = chunk.choices[0].delta.content
#         finish_reason = chunk.choices[0].finish_reason
#         if chunk_content is not None:
#             print(chunk_content, end="", file=file)
#             new_output += chunk_content
#         else:
#             print("finish_reason:", finish_reason, file=file)

#     messages.append({"role":"assistant", "content":new_output})

#     return messages

# messages = []
# messages = get_chatgpt_output("gpt-4", new_prompt, messages, "system")

# from openai import OpenAI

# client = OpenAI()

# completion = client.chat.completions.create(
#   model="gpt-4o-mini",
#   messages=[
#     {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
#     {"role": "user", "content": "Compose a poem that explains the concept of recursion in programming."}
#   ]
# )

# print(completion.choices[0].message)