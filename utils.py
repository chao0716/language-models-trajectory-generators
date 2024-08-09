import pybullet as p
import numpy as np

def render_camera():

    IMAGE_WIDTH = 640
    IMAGE_HEIGHT = 480
    camera_fov=55
    camera_position = [0, 0, 0.5]  # Camera position (above the blocks)
    target_position = [0, 0, 0]  # Camera looks at the center
    up_vector = [0, 1, 0]

    CAMERA_FAR = 1
    CAMERA_NEAR = 0.15
    HFOV_VFOV = IMAGE_WIDTH/IMAGE_HEIGHT

    view_matrix = p.computeViewMatrix(cameraEyePosition=camera_position,
                                      cameraTargetPosition=target_position,
                                      cameraUpVector=up_vector)
    
    projection_matrix = p.computeProjectionMatrixFOV(
        fov=camera_fov, aspect=HFOV_VFOV, nearVal=CAMERA_NEAR, farVal=CAMERA_FAR)
    
    _, _, rgb_image, depth_image, mask = p.getCameraImage(
        width=IMAGE_WIDTH, height=IMAGE_HEIGHT,
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL
    )

    # Convert depth image
    depth_image = np.array(depth_image).reshape(IMAGE_HEIGHT, IMAGE_WIDTH)

    # Convert depth buffer values to actual depth
    z_depth = CAMERA_FAR * CAMERA_NEAR / (CAMERA_FAR - (CAMERA_FAR - CAMERA_NEAR) * depth_image)

    
    rgb_image = np.array(rgb_image).reshape(IMAGE_HEIGHT, IMAGE_WIDTH, 4)[:, :, :3]  # Remove alpha channel
    
    # Get camera intrinsic parameters
    fx = IMAGE_WIDTH / (2.0 * np.tan(camera_fov * np.pi / 360.0))
    fy = fx
    cx = IMAGE_WIDTH / 2.0
    cy = IMAGE_HEIGHT / 2.0
    
    # Calculate the 3D points in the camera frame
    x = np.linspace(0, IMAGE_WIDTH - 1, IMAGE_WIDTH)
    y = np.linspace(0, IMAGE_HEIGHT - 1, IMAGE_HEIGHT)
    x, y = np.meshgrid(x, y)
    
    X = (x - cx) * z_depth / fx
    Y = (y - cy) * z_depth / fy
    Z = z_depth
    
    # Stack the XYZ coordinates
    camera_coordinates = np.stack((X, Y, Z), axis=-1)
    
    return rgb_image, camera_coordinates