import numpy as np

def generate_trajectory(start_pose, end_pose, num_points=100):
    """
    Generates a linear trajectory from start_pose to end_pose.
    
    Parameters:
    - start_pose: List of [x, y, z, yaw_orientation] for the starting position.
    - end_pose: List of [x, y, z, yaw_orientation] for the ending position.
    - num_points: Number of points in the trajectory.
    
    Returns:
    - trajectory: List of poses from start to end.
    """
    trajectory = []
    for t in np.linspace(0, 1, num_points):
        pose = [
            start_pose[0] * (1 - t) + end_pose[0] * t,
            start_pose[1] * (1 - t) + end_pose[1] * t,
            start_pose[2] * (1 - t) + end_pose[2] * t,
            start_pose[3] * (1 - t) + end_pose[3] * t
        ]
        trajectory.append(pose)
    return trajectory

# Current end-effector position
current_pose = [0.0, 0.0, 0.15, 0.0]

# Position above the blue block
target_pose = [0,0, 0.1, 0.0]

# Generate trajectory to move above the blue block
trajectory_1 = generate_trajectory(current_pose, target_pose)
