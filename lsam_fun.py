import warnings
import numpy as np
import matplotlib.pyplot as plt
import requests
from PIL import Image
from io import BytesIO
from lang_sam import LangSAM
from utils import render_camera_in_sim
import cv2
from utils import get_object_properties
import math
from utils import get_object_position
def save_mask(mask_np, filename):
    mask_image = Image.fromarray((mask_np * 255).astype(np.uint8))
    mask_image.save(filename)

def display_image_with_masks(image, masks):
    num_masks = len(masks)

    fig, axes = plt.subplots(1, num_masks + 1, figsize=(15, 5))
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    for i, mask_np in enumerate(masks):
        axes[i+1].imshow(mask_np, cmap='gray')
        axes[i+1].set_title(f"Mask {i+1}")
        axes[i+1].axis('off')

    plt.tight_layout()
    plt.show()

def display_image_with_boxes(image, boxes, logits):
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.set_title("Image with Bounding Boxes")
    ax.axis('off')

    for box, logit in zip(boxes, logits):
        x_min, y_min, x_max, y_max = box
        confidence_score = round(logit.item(), 2)  # Convert logit to a scalar before rounding
        box_width = x_max - x_min
        box_height = y_max - y_min

        # Draw bounding box
        rect = plt.Rectangle((x_min, y_min), box_width, box_height, fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)

        # Add confidence score as text
        ax.text(x_min, y_min, f"Confidence: {confidence_score}", fontsize=8, color='red', verticalalignment='top')

    plt.show()

def print_bounding_boxes(boxes):
    print("Bounding Boxes:")
    for i, box in enumerate(boxes):
        print(f"Box {i+1}: {box}")

def print_detected_phrases(phrases):
    print("\nDetected Phrases:")
    for i, phrase in enumerate(phrases):
        print(f"Phrase {i+1}: {phrase}")

def print_logits(logits):
    print("\nConfidence:")
    for i, logit in enumerate(logits):
        print(f"Logit {i+1}: {logit}")
        

 #%% 
if __name__ == "__main__":      
    model = LangSAM()
    text_prompt=  "cola"
    
    rgb_img, depth_camera_coordinates = render_camera_in_sim()
    rgb_img = Image.fromarray(rgb_img.astype('uint8'), 'RGB')
    
    fig, ax = plt.subplots()
    ax.imshow(rgb_img)
    ax.axis('off')
    plt.show()
    
    x = depth_camera_coordinates[:, :, 0]
    y = depth_camera_coordinates[:, :, 1]
    z = depth_camera_coordinates[:, :, 2]
    
    masks, boxes, phrases, logits = model.predict(
        rgb_img, text_prompt)
    
    display_image_with_boxes(rgb_img, boxes,logits)
    display_image_with_masks(rgb_img, masks)
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
                print('confidence_score:', confidence_score)
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
                
                