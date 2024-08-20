from utils import render_camera_in_sim
import numpy as np
from PIL import Image
from utils import get_chatgpt_output, render_camera_in_sim, encode_image_to_base64
import os
import math
import requests
from utils import get_object_properties, get_object_position

class API:

    def __init__(self, langsam_model, command, language_model, initial_rgb_img, depth_camera_coordinates, initial_img_base64, robot):

        self.langsam_model = langsam_model
        self.completed_task = False
        self.failed_task = False
        self.command = command
        self.language_model = language_model
        self.initial_img_base64 = initial_img_base64
        self.initial_rgb_img = initial_rgb_img
        self.depth_camera_coordinates = depth_camera_coordinates
        self.robot = robot

    def task_completed(self):
        self.completed_task = True
        self.failed_task = False

    def task_failed(self):
        self.completed_task = False
        self.failed_task = True

    # Function to check if the task is completed using GPT-4o

    def check_task_completed(self):

        current_rgb_img, _ = render_camera_in_sim()
        current_img_base64 = encode_image_to_base64(current_rgb_img)

        api_key = os.getenv("OPENAI_API_KEY")
        # Prepare the payload for GPT-4o
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "You are tasked with determining whether a user command was completed successfully or not, based on the current environment observation after the execution of the task and initial environment observation."
                        },
                        {
                            "type": "text",
                            "text": f"The user command is: {self.command}"
                        },
                        {
                            "type": "text",
                            "text":
                            '''1. If the task was completed successfully, output
                            ```python
                            task_completed()
                            ```.
                            
                            2. If the task was not completed successfully, output
                            ```python
                            task_failed()
                            ```.
                            Do not define the task_completed and task_failed functions yourself.
                            '''
                        },
                        {
                            "type": "text",
                            "text": "The initial environment image is provided below."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{self.initial_img_base64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": "The current environment image is provided below."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{current_img_base64}"
                            }
                        }
                    ]
                }
            ],
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        response = requests.post(
            "https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        # Convert the response to a dictionary
        response_dict = response.json()

        # Extract the message content
        messages = response_dict['choices'][0]['message']['content']

        # Print the extracted content
        print(messages)

        code_block = messages.split("```python")
        
        task_completed = self.task_completed
        task_failed = self.task_failed
        for block in code_block:
            if len(block.split("```")) > 1:
                code = block.split("```")[0]
                print(code)
                exec(code)

    def detect_object(self, text_prompt):

        rgb_img, depth_camera_coordinates = self.initial_rgb_img, self.depth_camera_coordinates
        rgb_img = Image.fromarray(rgb_img.astype('uint8'), 'RGB')

        x = depth_camera_coordinates[:, :, 0]
        y = depth_camera_coordinates[:, :, 1]
        z = depth_camera_coordinates[:, :, 2]

        masks, boxes, phrases, logits = self.langsam_model.predict(
            rgb_img, text_prompt)
        # Initialize an empty dictionary
        mask_dict = {}

        if len(masks) == 0:
            print(
                f"No objects of the '{text_prompt}' prompt detected in the image.")
        else:
            # Convert masks to numpy arrays
            masks_np = [mask.squeeze().cpu().numpy() for mask in masks]

            for i, (mask_np, box, logit) in enumerate(zip(masks_np, boxes, logits)):
                # Convert logit to a scalar before rounding
                confidence_score = round(logit.item(), 2)
                # Change confidence_score if wrong object is detected
                if confidence_score < 0.3:
                    print(
                        f"No objects of the '{text_prompt}' prompt detected in the image.")
                else:
                    
                    length, width, orientation = get_object_properties(mask_np, x, y, z)
                    position = get_object_position(box, depth_camera_coordinates)
                    print("Position of " + text_prompt + str(i) + ":", list(
                        [np.around(position[1], 3),np.around(position[0], 3),  np.around(position[2], 3)]))

                    print("Dimensions:")
                    print("Width:", np.around(width, 3))
                    print("Length:", np.around(length, 3))
                    print("Height of " + text_prompt + str(i) + ":", np.around(position[2], 3))
                    print("Orientation along longer side (length):", np.around(math.radians(orientation), 3), "\n")

                    # Add the mask and corresponding label to the dictionary
                    mask_dict[text_prompt + str(i)] = mask_np

    # def detect_object(self, text_prompt):

    #     rgb_img, depth_camera_coordinates = self.initial_rgb_img, self.depth_camera_coordinates
    #     rgb_img = Image.fromarray(rgb_img.astype('uint8'), 'RGB')

    #     x = depth_camera_coordinates[:, :, 0]
    #     y = depth_camera_coordinates[:, :, 1]
    #     z = depth_camera_coordinates[:, :, 2]

    #     masks, boxes, phrases, logits = self.langsam_model.predict(
    #         rgb_img, text_prompt)
    #     # Initialize an empty dictionary
    #     mask_dict = {}

    #     if len(masks) == 0:
    #         print(
    #             f"No objects of the '{text_prompt}' prompt detected in the image.")
    #     else:
    #         # Convert masks to numpy arrays
    #         masks_np = [mask.squeeze().cpu().numpy() for mask in masks]

    #         for i, (mask_np, box, logit) in enumerate(zip(masks_np, boxes, logits)):
    #             # Convert logit to a scalar before rounding
    #             confidence_score = round(logit.item(), 2)
    #             # Change confidence_score if wrong object is detected
    #             if confidence_score < 0.5:
    #                 pass
    #             else:
    #                 x_min, y_min, x_max, y_max = box
    #                 # Ensure the coordinates are integers
    #                 x_min = int(x_min)
    #                 y_min = int(y_min)
    #                 x_max = int(x_max)
    #                 y_max = int(y_max)

    #                 # Calculate object dimensions in pixel units
    #                 object_width_px = x_max - x_min
    #                 object_length_px = y_max - y_min

    #                 # Find the corresponding 3D coordinates of the bounding box
    #                 x_min_world = x[y_min:y_max, x_min:x_max].min()
    #                 x_max_world = x[y_min:y_max, x_min:x_max].max()
    #                 y_min_world = y[y_min:y_max, x_min:x_max].min()
    #                 y_max_world = y[y_min:y_max, x_min:x_max].max()
    #                 z_min_world = z[y_min:y_max, x_min:x_max].min()
    #                 z_max_world = z[y_min:y_max, x_min:x_max].max()

    #                 # Calculate object dimensions in real-world units
    #                 object_width_real = x_max_world - x_min_world
    #                 object_length_real = y_max_world - y_min_world
    #                 object_height_real = (z_max_world + z_min_world)/2

    #                 # Calculate the center of the bounding box in pixel units
    #                 center_pixel_x = int(x_min + object_width_px / 2)
    #                 center_pixel_y = int(y_min + object_length_px / 2)

    #                 # Extract the corresponding real-world coordinates from the camera_coordinates array
    #                 center_real_x = x[center_pixel_y, center_pixel_x]
    #                 center_real_y = y[center_pixel_y, center_pixel_x]
    #                 center_real_z = z[center_pixel_y, center_pixel_x]

    #                 print("Position of " + text_prompt + str(i) + ":", list(
    #                     [np.around(center_real_y, 3),np.around(center_real_x, 3),  np.around(center_real_z, 3)]))

    #                 print("Dimensions:")
    #                 print("Width:", np.around(object_width_real, 3))
    #                 print("Length:", np.around(object_length_real, 3))
    #                 print("Height of " + text_prompt + str(i) + ":", np.around(center_real_z, 3))

    #                 # Calculating rotation in world frame
    #                 bounding_cubes_orientation_width = np.arctan2(
    #                     0, x_max - x_min)
    #                 bounding_cubes_orientation_length = np.arctan2(
    #                     y_max - y_min, 0)

    #                 if object_width_real < object_length_real:
    #                     # print("Orientation along shorter side (width):",
    #                     #       np.around(bounding_cubes_orientation_width, 3))
    #                     print("Orientation along longer side (length):", np.around(
    #                         bounding_cubes_orientation_length, 3), "\n")
    #                 else:
    #                     # print("Orientation along shorter side (length):",
    #                     #       np.around(bounding_cubes_orientation_length, 3))
    #                     print("Orientation along longer side (width):", np.around(
    #                         bounding_cubes_orientation_width, 3), "\n")

    #                 # Add the mask and corresponding label to the dictionary
    #                 mask_dict[text_prompt + str(i)] = mask_np

    def execute_trajectory(self, trajectory):

        for i in range (len(trajectory)):
            x = trajectory[i][0]
            y = trajectory[i][1]
            z = trajectory[i][2]
            yaw = math.degrees(trajectory[i][3])
            self.robot.move_world(x, y, z, yaw)
        print("Generated trajectory Executed")
        
    # def execute_trajectory(self, trajectory):

    #     x = trajectory[0][0]
    #     y = trajectory[0][1]
    #     z = trajectory[0][2]
    #     yaw = math.degrees(trajectory[0][3])
    #     self.robot.move_world(x, y, z, yaw)
        
    #     x = trajectory[0][0]
    #     y = trajectory[0][1]
    #     z = trajectory[0][2]
    #     yaw = math.degrees(trajectory[-1][3])
    #     self.robot.move_world(x, y, z, yaw)
        
    #     x = trajectory[-1][0]
    #     y = trajectory[-1][1]
    #     z = trajectory[-1][2]
    #     yaw = math.degrees(trajectory[-1][3])
    #     self.robot.move_world(x, y, z, yaw)
        
    #     print("Generated trajectory Executed")
        
        
    def open_gripper(self):
        self.robot.gripper_open()
        print("Gripper open")

    def close_gripper(self):
        self.robot.gripper_close()
        print("Gripper close")