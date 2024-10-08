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

def are_transforms_close(a, b=None, linear_tol=0.005, angular_tol=0.03):
    if b is None:
        b = np.eye(4)
    c = np.linalg.inv(a) @ b
    xyz = c[:3, 3]
    if np.any(np.abs(xyz) > linear_tol):
        return False
    rpy = t3d.euler.mat2euler(c[:3, :3])
    if np.any(np.abs(np.array(rpy)) > angular_tol):
        return False
    return True

class Kinowa:
    def __init__(self, ip="192.168.1.10", username="admin", password="admin"):
        super().__init__()
        self.__action_done = threading.Event()
        self.__ip = ip
        self.__username = username
        self.__password = password
        self.__connect_robot()

    def __connect_robot(self):
        try:

            # Establish a connection with the robot
            self.transport = TCPTransport()
            self.transport.connect(self.__ip, 10000)
            self.router = RouterClient(self.transport, lambda k: None)
            self.session_manager = SessionClient(self.router)

            session_info = Session_pb2.CreateSessionInfo()
            session_info.username = self.__username
            session_info.password = self.__password
            session_info.session_inactivity_timeout = 600000  # (milliseconds)
            session_info.connection_inactivity_timeout = 20000  # (milliseconds)
            self.session_manager.CreateSession(session_info)

            self.base = BaseClient(self.router)
            print("Connected to the robot successfully")
        except Exception as e:
            print(f"Failed to connect to the robot: {e}")

    def move(self, pose, blocking=True):
        while not are_transforms_close(self.get_pose(), pose):
            self._move(pose, blocking)
            # time.sleep(0.01)
            
    def _move(self, pose, blocking=True):
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
    
    def open_gripper_speed(self):
        # Create the GripperCommand we will send
        gripper_command = Base_pb2.GripperCommand()
        finger = gripper_command.gripper.finger.add()

        # Close the gripper with position increments
        gripper_command.mode = Base_pb2.GRIPPER_POSITION
        finger.finger_identifier = 1

        # Set speed to open gripper
        gripper_command.mode = Base_pb2.GRIPPER_SPEED
        finger.value = 0.1
        self.base.SendGripperCommand(gripper_command)
        gripper_request = Base_pb2.GripperRequest()

        # Wait for reported position to be opened
        gripper_request.mode = Base_pb2.GRIPPER_POSITION
        while True:
            gripper_measure = self.base.GetMeasuredGripperMovement(gripper_request)
            if len (gripper_measure.finger):
                # print("Current position is : {0}".format(gripper_measure.finger[0].value))
                if gripper_measure.finger[0].value < 0.01:
                    break
            else: # Else, no finger present in answer, end loop
                break

    def close_gripper_speed(self):

        # Create the GripperCommand we will send
        gripper_command = Base_pb2.GripperCommand()
        finger = gripper_command.gripper.finger.add()

        # Close the gripper with position increments
        gripper_command.mode = Base_pb2.GRIPPER_POSITION
        finger.finger_identifier = 1
        
        # Set speed to close gripper
        gripper_command.mode = Base_pb2.GRIPPER_SPEED
        finger.value = -0.1
        self.base.SendGripperCommand(gripper_command)
        gripper_request = Base_pb2.GripperRequest()
        

        # Wait for reported speed to be 0
        gripper_request.mode = Base_pb2.GRIPPER_SPEED
        while True:
            gripper_measure = self.base.GetMeasuredGripperMovement(gripper_request)
            if len (gripper_measure.finger):
                # print("Current speed is : {0}".format(gripper_measure.finger[0].value))
                if gripper_measure.finger[0].value == 0.0:
                    break
            else: # Else, no finger present in answer, end loop
                break

    def control_gripper(self, position, blocking = True):
        
        # Create the gripper command request
        gripper_command = Base_pb2.GripperCommand()
        gripper_command.mode = Base_pb2.GRIPPER_POSITION

        finger = gripper_command.gripper.finger.add()
        finger.finger_identifier = 1
        finger.value = position  # The API uses a scale of 0-100
        
        print (finger)
        return 

 
        notification_handle = None
        if blocking:
            notification_handle = self.base.OnNotificationActionTopic(self.__action_notification_callback, Base_pb2.NotificationOptions())
        
        # Send the gripper command and wait for the result
        self.base.SendGripperCommand(gripper_command)
        
        if blocking:
            self.__action_done.wait()
            self.base.Unsubscribe(notification_handle)
            self.__action_done.clear()
            
    def close(self):
        try:
            self.session_manager.CloseSession()
            self.transport.disconnect()
            print("Disconnected from the robot")
        except Exception as e:
            print(f"Failed to disconnect from the robot: {e}")
            
class Robot:
    def __init__(self, robot):
        self.robot = robot

    def go_home(self, blocking=True):
        pose = t3d.affines.compose(
            [0.525, 0.044, 0.2],
            # [0.525, 0.044, 0.0225],
            t3d.euler.euler2mat(
                math.radians(180), math.radians(0), math.radians(-90), "sxyz"
            ),
            [1, 1, 1],
        )
        
        self.robot.move(pose, blocking)
        
    def go_safe_place(self, blocking=True):
        pose = t3d.affines.compose(
            [0.25, 0.044, 0.2],
            t3d.euler.euler2mat(
                math.radians(180), math.radians(0), math.radians(-90), "sxyz"
            ),
            [1, 1, 1],
        )
        self.robot.move(pose, blocking)

    def move(self, pose, blocking=True):
        self.robot.move(pose, blocking)

    def get_pose(self):
        return self.robot.get_pose()

    def move_tool_xyz(self, x=0, y=0, z=0, blocking=True):
        pose = self.robot.get_pose()
        tool_pose_change = np.eye(4)
        tool_pose_change[0, 3] = x
        tool_pose_change[1, 3] = y
        tool_pose_change[2, 3] = z
        pose = pose @ tool_pose_change
        self.robot.move(pose, blocking=blocking)
        
    def move_world(self, x, y, z, yaw, blocking=True):
        x = x + 0.525
        y = y + 0.044
        z = z
        yaw = yaw
        pose = t3d.affines.compose(
            [x, y, z],
            t3d.euler.euler2mat(
                math.radians(180), math.radians(0), math.radians(-90+yaw), "sxyz"
            ),
            [1, 1, 1],
        )
        
        self.robot.move(pose, blocking)

    def move_tool_ypr(self, yaw=0, pitch=0, roll=0, blocking=True):
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

    def move_tool_rpy(self, roll=0, pitch=0, yaw=0, blocking=True):
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

    def move_tool_xyzrpy(self, x=0, y=0, z=0, roll=0, pitch=0, yaw=0, blocking=True):
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
        
    def gripper_open(self):
        self.robot.open_gripper_speed()

    def gripper_close(self):
        self.robot.close_gripper_speed()
        
#%%       
if __name__ == "__main__":

    interface = Kinowa(ip="192.168.1.10")
    robot = Robot(interface)
    
    robot.gripper_open()    
    #%%
    robot.gripper_close()
    #%%
    robot.gripper_open()
  
    robot.go_safe_place()
#%%
    robot.move_world(x = 0, y= 0, z = 0.2, yaw = 0)
#%%
#     print('up')
#     robot.move_tool_xyzrpy(z=-0.15)
#     print('down')
#     robot.move_tool_xyzrpy(z=0.15)
#     #%%
#     print('back')
#     robot.move_tool_xyzrpy(x=-0.05)
#     print('front')
#     robot.move_tool_xyzrpy(x=0.05)    
#     #%%
#     print('right')
#     robot.move_tool_xyzrpy(y=-0.05)
#     print('left')
#     robot.move_tool_xyzrpy(y=0.05)      
# #%%
#     robot.go_home()
#     robot.go_home()
# #%%
#     interface.close()


