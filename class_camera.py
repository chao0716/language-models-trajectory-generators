import pyrealsense2 as rs
import numpy as np
import json
import math
import matplotlib.pyplot as plt
import cv2
from PIL import Image

class Realsense:
    def __init__(self):
        
        self.__pipeline = rs.pipeline()
        config = rs.config()
        
        jsonDict = json.load(open("./d435i.json"))
        jsonString = str(jsonDict).replace("'", '\"')
        
        ctx = rs.context()
        dev = None
        
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

        print("[INFO] start streaming...")
        profile = self.__pipeline.start(config)
        self.__point_cloud = rs.pointcloud()

    def get_aligned_verts(self):

        for i in range(100):
            frames = self.__pipeline.wait_for_frames()
    
        # Create an align object
        # rs.align allows us to perform alignment of depth frames to others frames
        # The "align_to" is the stream type to which we plan to align depth frames.
        align_to = rs.stream.color
        align = rs.align(align_to)

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        # aligned_depth_frame is a 1920x1280 depth image
        aligned_depth_frame = aligned_frames.get_depth_frame()
        points_1080p = self.__point_cloud.calculate(aligned_depth_frame)
        aligned_verts = (np.asanyarray(points_1080p.get_vertices()).view(np.float32).reshape(720, 1280, 3))  # xyz

        color_frame = aligned_frames.get_color_frame()

        color_image = np.asanyarray(color_frame.get_data())

        return color_image, aligned_verts

    def stop_streaming(self):
        self.__pipeline.stop()

# %%
if __name__ == '__main__':
    d435 = Realsense()
    color_image, aligned_verts = d435.get_aligned_verts()
    d435.stop_streaming()
    
    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    plt.imshow(color_image)
    plt.show()
    plt.imshow(aligned_verts[:,:,2])
    plt.show()
