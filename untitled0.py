# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 17:57:10 2024

@author: chaoz
"""

import math

print(math.degrees(-1.533))

def move_above_bottle(bottle_position, height_above):
    """
    Generate a trajectory to move the end-effector to a position directly above the bottle.
    
    Parameters:
    bottle_position (list): The [x, y, z] position of the bottle.
    height_above (float): The height above the bottle to position the end-effector.
    
    Returns:
    list: Trajectory points to move above the bottle.
    """
    trajectory = []
    current_position = [0.0, 0, 0.2]
    target_position = [bottle_position[0], bottle_position[1], height_above]
    
    # Generate trajectory points
    steps = 2
    for i in range(steps):
        x = current_position[0] + (target_position[0] - current_position[0]) * (i / (steps - 1))
        y = current_position[1] + (target_position[1] - current_position[1]) * (i / (steps - 1))
        z = current_position[2] + (target_position[2] - current_position[2]) * (i / (steps - 1))
        trajectory.append([x, y, z, 0])
    
    return trajectory

# Move above the bottle
bottle_position = [-0.007, 0.001, 0.145]
height_above = 0.25  # Safe height above the bottle
trajectory_1 = move_above_bottle(bottle_position, height_above)
