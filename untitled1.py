import numpy as np
def generate_trajectory(start, end, steps=100):
    """
    Generates a linear trajectory from start to end with a given number of steps.
    
    Parameters:
    start (list): Starting pose [x, y, z, orientation]
    end (list): Ending pose [x, y, z, orientation]
    steps (int): Number of steps in the trajectory
    
    Returns:
    list: List of trajectory points
    """
    trajectory = []
    for t in np.linspace(0, 1, steps):
        point = [
            start[0] * (1 - t) + end[0] * t,
            start[1] * (1 - t) + end[1] * t,
            start[2] * (1 - t) + end[2] * t,
            start[3] * (1 - t) + end[3] * t
        ]
        trajectory.append(point)
    return trajectory

# Current end-effector position
current_position = [0.0, 0, 0.2, 0]

# Position directly above the red block
above_block_position = [-0.005, 0.002, 0.2, 1.571]

# Generate trajectory to move above the block
trajectory_1 = generate_trajectory(current_position, above_block_position)