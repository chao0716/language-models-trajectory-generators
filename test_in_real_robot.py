import pybullet as p
import pybullet_data
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
from utils import get_chatgpt_output, render_camera_in_sim, encode_image_to_base64
from prompts.error_correction_prompt import ERROR_CORRECTION_PROMPT
from prompts.task_summary_prompt import TASK_SUMMARY_PROMPT
from prompts.task_failure_prompt import TASK_FAILURE_PROMPT
import os
import requests
from class_robot import Robot
from class_camera import Realsense
# Initialize Robot
robot = Robot(ip='192.168.1.10')
robot.go_pos_not_effect_camera()
# Initialize Camera
d435 = Realsense()
initial_rgb_img, _ = d435.get_aligned_verts()
initial_img_base64 = encode_image_to_base64(initial_rgb_img)
# Go home
robot.go_home()
#%%
command = 'put the blue block on the top of red block'
language_model = 'gpt-4o'
LangSAM_model = LangSAM()
api = API(LangSAM_model, command, language_model, initial_img_base64)

detect_object = api.detect_object
execute_trajectory = api.execute_trajectory
open_gripper = api.open_gripper
close_gripper = api.close_gripper
completed_task = api.completed_task
failed_task = api.failed_task
check_task_completed = api.check_task_completed
#%%
messages_for_infer = []
error = False
new_prompt = MAIN_PROMPT.replace("[INSERT EE POSITION]", str(config.ee_start_position)).replace("[INSERT TASK]", command)
messages_for_infer = get_chatgpt_output(language_model, new_prompt, messages_for_infer, "system")
#%%
while not completed_task:
#%%
    new_prompt = ""

    if len(messages_for_infer[-1]["content"].split("```python")) > 1:

        code_block = messages_for_infer[-1]["content"].split("```python")

        block_number = 0

        for block in code_block:
            if len(block.split("```")) > 1:
                code = block.split("```")[0]
                block_number += 1
                try:
                    f = StringIO()
                    with redirect_stdout(f):
                        exec(code)
                except Exception:
                    error_message = traceback.format_exc()
                    new_prompt += ERROR_CORRECTION_PROMPT.replace("[INSERT BLOCK NUMBER]", str(block_number)).replace("[INSERT ERROR MESSAGE]", error_message)
                    new_prompt += "\n"
                    error = True
                else:
                    s = f.getvalue()
                    error = False
                    if s != "" and len(s) < 2000:
                        new_prompt += PRINT_OUTPUT_PROMPT.replace("[INSERT PRINT STATEMENT OUTPUT]", s)
                        new_prompt += "\n"
                        error = True

    if error:
        completed_task = False
        failed_task = False
    
    if not completed_task:

        if failed_task:

            #FAILED TASK! Generating summary of the task execution attempt
            new_prompt += TASK_SUMMARY_PROMPT
            new_prompt += "\n"

            #Generating ChatGPT output..."
            messages_for_infer = get_chatgpt_output(language_model, new_prompt, messages_for_infer, "user")

            #"RETRYING TASK..."
            new_prompt = MAIN_PROMPT.replace("[INSERT EE POSITION]", str(config.ee_start_position)).replace("[INSERT TASK]", command)
            new_prompt += "\n"
            new_prompt += TASK_FAILURE_PROMPT.replace("[INSERT TASK SUMMARY]", messages_for_infer[-1]["content"])

            messages_for_infer = []

            error = False

            #Generating ChatGPT output..." 
            messages_for_infer = get_chatgpt_output(language_model, new_prompt, messages_for_infer, "system")

            failed_task = False

        else:
            # if everything is fine and task is not completed
            messages_for_infer = get_chatgpt_output(language_model, new_prompt, messages_for_infer, "user")
            
#%%