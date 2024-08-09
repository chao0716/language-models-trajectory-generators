import pybullet as p
import pybullet_data
from utils import render_camera
from lang_sam import LangSAM
from PIL import Image
import numpy as np
# Initialize PyBullet simulation
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.resetSimulation()

# Set gravity
p.setGravity(0, 0, -9.81)

# Load plane and set the environment
plane_id = p.loadURDF("plane.urdf")

# Create two blocks with different colors
block1_id = p.loadURDF("cube.urdf", basePosition=[0, 0, 0], globalScaling=0.05)
block2_id = p.loadURDF("cube.urdf", basePosition=[0.1, 0, 0], globalScaling=0.05)
# Set block colors
p.changeVisualShape(block1_id, -1, rgbaColor=[1, 0, 0, 1])  # Red
p.changeVisualShape(block2_id, -1, rgbaColor=[0, 0, 1, 1])  # Blue


rgb_img, depth_camera_coordinates = render_camera()
p.disconnect()

text_prompt = "block with red color"
LangSAM_model = LangSAM()
rgb_img = Image.fromarray(rgb_img.astype('uint8'), 'RGB')

def detect_object(LangSAM_model,text_prompt,rgb_img, depth_camera_coordinates):
    
    x = depth_camera_coordinates[:,:,0]
    y = depth_camera_coordinates[:,:,1]
    z = depth_camera_coordinates[:,:,2]
    
    masks, boxes, phrases, logits = LangSAM_model.predict(rgb_img, text_prompt)
    if len(masks) == 0:
        print(f"No objects of the '{text_prompt}' prompt detected in the image.")
    else:
        # Convert masks to numpy arrays
        masks_np = [mask.squeeze().cpu().numpy() for mask in masks]
    
        for i, (mask_np, box, logit) in enumerate(zip(masks_np, boxes, logits)):
            confidence_score = round(logit.item(), 2) # Convert logit to a scalar before rounding
            if confidence_score < 0.5:  
                pass
            else:
                x_min, y_min, x_max, y_max = box
                # Ensure the coordinates are integers
                x_min = int(x_min)
                y_min = int(y_min)
                x_max = int(x_max)
                y_max = int(y_max)
    
                # Calculate object dimensions in pixel units
                object_width_px = x_max - x_min
                object_length_px = y_max - y_min
                
                # Find the corresponding 3D coordinates of the bounding box
                x_min_world = x[y_min:y_max, x_min:x_max].min()
                x_max_world = x[y_min:y_max, x_min:x_max].max()
                y_min_world = y[y_min:y_max, x_min:x_max].min()
                y_max_world = y[y_min:y_max, x_min:x_max].max()
                z_min_world = z[y_min:y_max, x_min:x_max].min()
                z_max_world = z[y_min:y_max, x_min:x_max].max()
        
                # Calculate object dimensions in real-world units
                object_width_real = x_max_world - x_min_world
                object_length_real = y_max_world - y_min_world
                object_height_real = (z_max_world + z_min_world)/2
                
                # Calculate the center of the bounding box in pixel units
                center_pixel_x = int(x_min + object_width_px / 2)
                center_pixel_y = int(y_min + object_length_px / 2)
            
                # Extract the corresponding real-world coordinates from the camera_coordinates array
                center_real_x = x[center_pixel_y, center_pixel_x]
                center_real_y = y[center_pixel_y, center_pixel_x]
                center_real_z = z[center_pixel_y, center_pixel_x]
                
                print("Position of " + text_prompt+ str(i) + ":", list([center_real_x, center_real_y, center_real_z]))
        
                print("Dimensions:")
                print("Width:", object_width_real)
                print("Length:", object_length_real)
                print("Height:", center_real_z)
                
                # Calculating rotation in world frame
                bounding_cubes_orientation_width = np.arctan2(0, x_max - x_min)
                # Calculate length orientation 
                bounding_cubes_orientation_length = np.arctan2(y_max - y_min, 0)

                if object_width_real < object_length_real:
                    print("Orientation along shorter side (width):", np.around(bounding_cubes_orientation_width, 3))
                    print("Orientation along longer side (length):", np.around(bounding_cubes_orientation_length, 3), "\n")
                else:
                    print("Orientation along shorter side (length):", np.around(bounding_cubes_orientation_length, 3))
                    print("Orientation along longer side (width):", np.around(bounding_cubes_orientation_width, 3), "\n")

#%%
detect_object(LangSAM_model,text_prompt,rgb_img, depth_camera_coordinates)