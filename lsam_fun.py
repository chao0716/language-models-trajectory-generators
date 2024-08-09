import warnings
import numpy as np
import matplotlib.pyplot as plt
import requests
from PIL import Image
from io import BytesIO
from lang_sam import LangSAM

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
        
def detect_object(self, text_prompt, rgb_image):
    model = LangSAM()
    rgb_image = Image.fromarray(rgb_image.astype('uint8'), 'RGB')
    masks, boxes, phrases, logits = model.predict(rgb_image, text_prompt)
    if len(masks) == 0:
        print(f"No objects of the '{text_prompt}' prompt detected in the image.")
    else:
        # Convert masks to numpy arrays
        masks_np = [mask.squeeze().cpu().numpy() for mask in masks]
    
        # Display the original image and masks side by side
        display_image_with_masks(rgb_image, masks_np)
    
        # Display the image with bounding boxes and confidence scores
        display_image_with_boxes(rgb_image, boxes, logits)
    
        # Print the bounding boxes, phrases, and logits
        print_bounding_boxes(boxes)
        print_detected_phrases(phrases)
        print_logits(logits)

#         print("Position of " + segmentation_texts[i] + ":", list(np.around(bounding_cube_world_coordinates[4], 3)))

#         print("Dimensions:")
#         print("Width:", object_width)
#         print("Length:", object_length)
#         print("Height:", object_height)

#         if object_width < object_length:
#             print("Orientation along shorter side (width):", np.around(bounding_cubes_orientations[i][0], 3))
#             print("Orientation along longer side (length):", np.around(bounding_cubes_orientations[i][1], 3), "\n")
#         else:
#             print("Orientation along shorter side (length):", np.around(bounding_cubes_orientations[i][1], 3))
#             print("Orientation along longer side (width):", np.around(bounding_cubes_orientations[i][0], 3), "\n")
#%%
def main():
    model = LangSAM()
    image_pil = Image.open("image.png").convert("RGB")
    text_prompt = "block with blue color"
    masks, boxes, phrases, logits = model.predict(image_pil, text_prompt)
    
    if len(masks) == 0:
        print(f"No objects of the '{text_prompt}' prompt detected in the image.")
    else:
        # Convert masks to numpy arrays
        masks_np = [mask.squeeze().cpu().numpy() for mask in masks]
    
        # Display the original image and masks side by side
        display_image_with_masks(image_pil, masks_np)
    
        # Display the image with bounding boxes and confidence scores
        display_image_with_boxes(image_pil, boxes, logits)
    
        # Print the bounding boxes, phrases, and logits
        print_bounding_boxes(boxes)
        print_detected_phrases(phrases)
        print_logits(logits)
        
if __name__ == "__main__":
    main()