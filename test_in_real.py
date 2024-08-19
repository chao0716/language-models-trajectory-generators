import openai
from lang_sam import LangSAM
import traceback
from io import StringIO
from contextlib import redirect_stdout
from prompts.main_prompt import MAIN_PROMPT
from prompts.print_output_prompt import PRINT_OUTPUT_PROMPT
from api import API
from utils import get_chatgpt_output, render_camera_in_sim, encode_image_to_base64, build_sim, record_audio, audio_to_text
from prompts.error_correction_prompt import ERROR_CORRECTION_PROMPT
from prompts.task_summary_prompt import TASK_SUMMARY_PROMPT
from prompts.task_failure_prompt import TASK_FAILURE_PROMPT
from prompts.command_from_audio import TASK_COMMAND_FROM_AUDIO
import os
import argparse
import re
from class_kinowa import Kinowa, Robot
from class_camera import Realsense
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
if __name__ == "__main__":
    
    ee_start_position = [0.0, 0, 0.2]
    openai.api_key = os.getenv("OPENAI_API_KEY")
    # Parse args
    parser = argparse.ArgumentParser(description="Main Program.")
    parser.add_argument("-lm", "--language_model", choices=["gpt-4o"], default="gpt-4o", help="select language model")
    parser.add_argument("-m", "--mode", choices=["text", "voice"], default="text", help="select mode to run")
    args = parser.parse_args()
    LangSAM_model = LangSAM()

    interface = Kinowa(ip="192.168.1.10")
    robot = Robot(interface)
    
    robot.gripper_open()    
    robot.gripper_close()
    robot.gripper_open()
    robot.go_safe_place()
    robot.go_safe_place()

    initial_rgb_img, depth_camera_coordinates = render_camera_in_sim()
    initial_img_base64 = encode_image_to_base64(initial_rgb_img)
    cv2.imwrite('initial_rgb_img.png', initial_rgb_img)
    
    #%%
    if args.mode == "text":
        command = input("Enter a command: ")
    elif args.mode == "voice":
        audio_name = "output.mp3"
        record_audio(audio_name)
        converted_text = audio_to_text(audio_name)
        print("Converted Text:", converted_text)
        messages_voice_to_command = []
        prompt_voice_to_command = TASK_COMMAND_FROM_AUDIO.replace("[INSERT AUDIO TEXT]", converted_text)
        messages_voice_to_command = get_chatgpt_output(args.language_model, prompt_voice_to_command, messages_voice_to_command, "system")
        # Extracted Command from Voice text
        command_match = re.search(r'<<command>>(.*?)</command>>', messages_voice_to_command[-1]["content"])
        if command_match:
            command = command_match.group(1)
            print("Extracted Task Command:", command)
        else:
            print("No command found in the output.")


    api = API(LangSAM_model, command, args.language_model, initial_rgb_img, depth_camera_coordinates, initial_img_base64, robot)
    detect_object = api.detect_object
    execute_trajectory = api.execute_trajectory
    open_gripper = api.open_gripper
    close_gripper = api.close_gripper
    completed_task = api.completed_task
    failed_task = api.failed_task
    check_task_completed = api.check_task_completed
    #%%
    messages_infer = []
    error = False
    prompt_infer = MAIN_PROMPT.replace("[INSERT EE POSITION]", str(ee_start_position)).replace("[INSERT TASK]", command)
    messages_infer = get_chatgpt_output(args.language_model, prompt_infer, messages_infer, "system")
    #%%
    while not completed_task:
#%%
        prompt_infer = ""
    
        if len(messages_infer[-1]["content"].split("```python")) > 1:
    
            code_block = messages_infer[-1]["content"].split("```python")
    
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
                        prompt_infer += ERROR_CORRECTION_PROMPT.replace("[INSERT BLOCK NUMBER]", str(block_number)).replace("[INSERT ERROR MESSAGE]", error_message)
                        prompt_infer += "\n"
                        error = True
                    else:
                        s = f.getvalue()
                        error = False
                        if s != "" and len(s) < 2000:
                            prompt_infer += PRINT_OUTPUT_PROMPT.replace("[INSERT PRINT STATEMENT OUTPUT]", s)
                            prompt_infer += "\n"
                            error = True
    
        if error:
            completed_task = False
            failed_task = False
        
        if not completed_task:
    
            if failed_task:
    
                #FAILED TASK! Generating summary of the task execution attempt
                prompt_infer += TASK_SUMMARY_PROMPT
                prompt_infer += "\n"
    
                #Generating ChatGPT output..."
                messages_infer = get_chatgpt_output(args.language_model, prompt_infer, messages_infer, "user")
    
                #"RETRYING TASK..."
                prompt_infer = MAIN_PROMPT.replace("[INSERT EE POSITION]", str(ee_start_position)).replace("[INSERT TASK]", command)
                prompt_infer += "\n"
                prompt_infer += TASK_FAILURE_PROMPT.replace("[INSERT TASK SUMMARY]", messages_infer[-1]["content"])
    
                messages_infer = []
    
                error = False
    
                #Generating ChatGPT output..." 
                messages_infer = get_chatgpt_output(args.language_model, prompt_infer, messages_infer, "system")
    
                failed_task = False
    
            else:
                # if everything is fine and task is not completed
                messages_infer = get_chatgpt_output(args.language_model, prompt_infer, messages_infer, "user")
#%%       
    # if args.mode == "text":
    #     command = input("Enter a command: ")
    # elif args.mode == "voice":
    #     audio_name = "output.mp3"
    #     record_audio(audio_name)
    #     converted_text = audio_to_text(audio_name)
    #     print("Converted Text:", converted_text)
    #     messages_voice_to_command = []
    #     prompt_voice_to_command = TASK_COMMAND_FROM_AUDIO.replace("[INSERT AUDIO TEXT]", converted_text)
    #     messages_voice_to_command = get_chatgpt_output(args.language_model, prompt_voice_to_command, messages_voice_to_command, "system")
    #     # Extracted Command from Voice text
    #     command_match = re.search(r'<<command>>(.*?)</command>>', messages_voice_to_command[-1]["content"])
    #     if command_match:
    #         command = command_match.group(1)
    #         print("Extracted Task Command:", command)
    #     else:
    #         print("No command found in the output.")
    # api.command = command
    # api.completed_task = False 
