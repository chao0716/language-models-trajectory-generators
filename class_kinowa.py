import time

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.messages import Session_pb2, Base_pb2, BaseCyclic_pb2
from kortex_api.RouterClient import RouterClient
from kortex_api.TCPTransport import TCPTransport
from kortex_api.UDPTransport import UDPTransport
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.client_stubs.SessionClientRpc import SessionClient
import math
import transforms3d as t3d
import numpy as np
import matplotlib.pyplot as plt
import threading


class Kinowa:
    def __init__(self, ip="192.168.1.10", username="admin", password="admin"):
        super().__init__()
        self.__action_done = threading.Event()
        self.__ip = ip
        self.__username = username
        self.__password = password
        self.__connect_robot()
        
        # Gripper control
        self.base_command = BaseCyclic_pb2.Command()
        self.base_command.frame_id = 0
        self.base_command.interconnect.command_id.identifier = 0
        self.base_command.interconnect.gripper_command.command_id.identifier = 0
        self.motorcmd = self.base_command.interconnect.gripper_command.motor_cmd.add()
        base_feedback = self.base_cyclic.RefreshFeedback()
        self.motorcmd.position = base_feedback.interconnect.gripper_feedback.motor[
            0
        ].position
        self.motorcmd.velocity = 0
        self.motorcmd.force = 100

    def __connect_robot(self):
        try:
            self.udp_transport = UDPTransport()
            self.udp_transport.connect(self.__ip, 10001)
            self.router_real_time = RouterClient(self.udp_transport, lambda k: None)
            self.session_manager_rt = SessionClient(self.router_real_time)

            session_info = Session_pb2.CreateSessionInfo()
            session_info.username = self.__username
            session_info.password = self.__password
            session_info.session_inactivity_timeout = 60000  # (milliseconds)
            session_info.connection_inactivity_timeout = 2000  # (milliseconds)
            self.session_manager_rt.CreateSession(session_info)

            # Establish a connection with the robot
            self.transport = TCPTransport()
            self.transport.connect(self.__ip, 10000)
            self.router = RouterClient(self.transport, lambda k: None)
            self.session_manager = SessionClient(self.router)

            session_info = Session_pb2.CreateSessionInfo()
            session_info.username = self.__username
            session_info.password = self.__password
            session_info.session_inactivity_timeout = 60000  # (milliseconds)
            session_info.connection_inactivity_timeout = 2000  # (milliseconds)
            self.session_manager.CreateSession(session_info)

            self.base = BaseClient(self.router)
            self.base_cyclic = BaseCyclicClient(self.router_real_time)
            print("Connected to the robot successfully")
        except Exception as e:
            print(f"Failed to connect to the robot: {e}")

    def move(self, pose, blocking=False):
        rpy = t3d.euler.mat2euler(pose[:3, :3], "sxyz")
        xyz = pose[:3, 3]
        command = Base_pb2.ConstrainedPose()
        command.target_pose.x = xyz[0]
        command.target_pose.y = xyz[1]
        command.target_pose.z = xyz[2]
        command.target_pose.theta_x = math.degrees(rpy[0])
        command.target_pose.theta_y = math.degrees(rpy[1])
        command.target_pose.theta_z = math.degrees(rpy[2])
        # print('set_rpy', rpy)

        waypoint = Base_pb2.Action()
        waypoint.name = "move_to_pose"
        waypoint.application_data = ""
        waypoint.reach_pose.CopyFrom(command)

        notification_handle = None
        if blocking:
            notification_handle = self.base.OnNotificationActionTopic(self.__action_notification_callback, Base_pb2.NotificationOptions())
        self.base.ExecuteAction(waypoint)
        
        if blocking:
            self.__action_done.wait()
            self.base.Unsubscribe(notification_handle)
            self.__action_done.clear()

    def __action_notification_callback(self, notification):
        if notification.action_event in [Base_pb2.ACTION_END, Base_pb2.ACTION_ABORT]:
            self.__action_done.set()

    def get_pose(self):
        feedback = self.base.GetMeasuredCartesianPose()
        xyz = [feedback.x, feedback.y, feedback.z]
        rpy = [
            math.radians(feedback.theta_x),
            math.radians(feedback.theta_y),
            math.radians(feedback.theta_z),
        ]
        # print('get_rpy', rpy)
        pose = t3d.affines.compose(
            xyz, t3d.euler.euler2mat(rpy[0], rpy[1], rpy[2], "sxyz"), [1, 1, 1]
        )
        return pose
    
    def control_gripper(self, position):
        try:
            self.motorcmd.position = position
            self.base_cyclic.Refresh(self.base_command)
        
            time.sleep(1.0)
            print("Gripper command executed successfully.")
        except Exception as e:
            print(f"An error occurred while controlling the gripper: {e}")

    def close(self):
        try:
            self.session_manager.CloseSession()
            self.transport.disconnect()
            print("Disconnected from the robot")
        except Exception as e:
            print(f"Failed to disconnect from the robot: {e}")


class Mock:
    def __init__(self):
        self.__pose = np.eye(4)

    def __plot_pose(self, pose, axis_length=0.3, workspace_limit=1.0):
        # Validate the input
        if pose.shape != (4, 4):
            raise ValueError("The transformation matrix must be a 4x4 matrix.")

        # Extract rotation matrix and translation vector
        rotation_matrix = pose[:3, :3]
        translation_vector = pose[:3, 3]

        # Create figure and 3D axes
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # Plot the origin
        ax.scatter(
            translation_vector[0],
            translation_vector[1],
            translation_vector[2],
            color="k",
        )

        # Plot the x, y, and z axes
        x_axis = rotation_matrix[:, 0] * axis_length
        y_axis = rotation_matrix[:, 1] * axis_length
        z_axis = rotation_matrix[:, 2] * axis_length

        # Draw the axes
        ax.quiver(
            translation_vector[0],
            translation_vector[1],
            translation_vector[2],
            x_axis[0],
            x_axis[1],
            x_axis[2],
            color="r",
            label="X axis",
        )
        ax.quiver(
            translation_vector[0],
            translation_vector[1],
            translation_vector[2],
            y_axis[0],
            y_axis[1],
            y_axis[2],
            color="g",
            label="Y axis",
        )
        ax.quiver(
            translation_vector[0],
            translation_vector[1],
            translation_vector[2],
            z_axis[0],
            z_axis[1],
            z_axis[2],
            color="b",
            label="Z axis",
        )

        # Set labels and limits
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_xlim([-workspace_limit, workspace_limit])
        ax.set_ylim([-workspace_limit, workspace_limit])
        ax.set_zlim([0, 2 * workspace_limit])

        ax.legend()
        # save plot
        plt.savefig("pose.png")

    def move(self, pose, blocking=False):
        self.__pose = pose
        if blocking:
            time.sleep(1)
            self.__plot_pose(pose)
        else:
            self.__plot_pose(pose)

    def get_pose(self):
        return self.__pose

    def close(self):
        pass


class Robot:
    def __init__(self, robot):
        self.robot = robot

    def go_home(self, blocking=False):
        pose = t3d.affines.compose(
            [0.35, 0, 0.2],
            t3d.euler.euler2mat(
                math.radians(180), math.radians(0), math.radians(0), "sxyz"
            ),
            [1, 1, 1],
        )
        self.robot.move(pose, blocking)

    def move(self, pose, blocking=False):
        self.robot.move(pose, blocking)

    def get_pose(self):
        return self.robot.get_pose()

    def move_tool_xyz(self, x=0, y=0, z=0, blocking=False):
        pose = self.robot.get_pose()
        tool_pose_change = np.eye(4)
        tool_pose_change[0, 3] = x
        tool_pose_change[1, 3] = y
        tool_pose_change[2, 3] = z
        pose = pose @ tool_pose_change
        self.robot.move(pose, blocking=blocking)

    def move_tool_ypr(self, yaw=0, pitch=0, roll=0, blocking=False):
        pose = self.robot.get_pose()
        change = t3d.affines.compose(
            [0, 0, 0],
            t3d.euler.euler2mat(
                math.radians(roll), math.radians(pitch), math.radians(yaw), "szyx"
            ),
            [1, 1, 1],
        )
        pose = pose @ change
        self.robot.move(pose, blocking=blocking)

    def move_tool_rpy(self, roll=0, pitch=0, yaw=0, blocking=False):
        pose = self.robot.get_pose()
        change = t3d.affines.compose(
            [0, 0, 0],
            t3d.euler.euler2mat(
                math.radians(roll), math.radians(pitch), math.radians(yaw), "sxyz"
            ),
            [1, 1, 1],
        )
        pose = pose @ change
        self.robot.move(pose, blocking=blocking)

    def move_tool_xyzrpy(self, x=0, y=0, z=0, roll=0, pitch=0, yaw=0, blocking=False):
        pose = self.robot.get_pose()
        change = t3d.affines.compose(
            [x, y, z],
            t3d.euler.euler2mat(
                math.radians(roll), math.radians(pitch), math.radians(yaw), "sxyz"
            ),
            [1, 1, 1],
        )
        pose = pose @ change
        self.robot.move(pose, blocking=blocking)


def main():
    interface = Kinowa(ip="192.168.1.10")
    # interface = Mock()
    robot = Robot(interface)
    
    # interface.control_gripper(1.0)
    
    robot.go_home(blocking=True)
    
    # print('up')
    # robot.move_tool_xyzrpy(z=-0.1, blocking=True)
    # print('down')
   #  robot.move_tool_xyzrpy(z=0.1, blocking=True)

    # robot.move_tool_xyzrpy(0.9, 0.1, 0.9, 45)
    # robot.move_tool_xyzrpy(roll=45)
    # robot.move_tool_xyzrpy(pitch=45)
    # print(robot.get_pose())
    # print(pose)
    # robot.move(pose)
    # robot.go_home()
    # time.sleep(3.3)

    interface.close()


if __name__ == "__main__":
    main()
