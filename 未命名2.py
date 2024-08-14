import pyrealsense2 as rs
import numpy as np
import cv2

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Enable the color stream
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

try:
    # Wait for a coherent pair of frames: depth and color
    for i in range(100):
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

    if not color_frame:
        raise RuntimeError("No color frame captured")

    # Convert image to numpy array
    color_image = np.asanyarray(color_frame.get_data())

    # Save the image
    cv2.imwrite('color_image.png', color_image)

    # Display the image
    cv2.imshow('Color Image', color_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

finally:
    # Stop streaming
    pipeline.stop()
