import openai
from lang_sam import LangSAM
import numpy as np
import traceback
from io import StringIO
from contextlib import redirect_stdout
from prompts.main_prompt import MAIN_PROMPT
from prompts.print_output_prompt import PRINT_OUTPUT_PROMPT
import config
import sys
from api import API
from utils import get_chatgpt_output, render_camera_in_sim, encode_image_to_base64, build_sim, record_audio, audio_to_text
from prompts.error_correction_prompt import ERROR_CORRECTION_PROMPT
from prompts.task_summary_prompt import TASK_SUMMARY_PROMPT
from prompts.task_failure_prompt import TASK_FAILURE_PROMPT
from prompts.command_from_audio import TASK_COMMAND_FROM_AUDIO
import os
import requests
import argparse
import re

parser = argparse.ArgumentParser(description="Main Program.")
parser.add_argument("-lm", "--language_model", choices=["gpt-4o"], default="gpt-4o", help="select language model")
parser.add_argument("-m", "--mode", choices=["text", "voice"], default="text", help="select mode to run")
parser.add_argument("-s", "--speaker", action = 'store_true', help="if let robot to speak")
args = parser.parse_args()

audio_name = "output.mp3"
record_audio(audio_name)
converted_text = audio_to_text(audio_name)
print("converted_text:", converted_text)
#%%
messages_voice_to_command = []
prompt_voice_to_command = TASK_COMMAND_FROM_AUDIO.replace("[INSERT AUDIO TEXT]", converted_text)
messages_voice_to_command = get_chatgpt_output(args.language_model, prompt_voice_to_command, messages_voice_to_command, "system")
# 使用正则表达式提取任务命令
command_match = re.search(r'<<command>>(.*?)</command>>', messages_voice_to_command[-1]["content"])
if command_match:
    task_command = command_match.group(1)
    print("Extracted Task Command:", task_command)
else:
    print("No command found in the output.")