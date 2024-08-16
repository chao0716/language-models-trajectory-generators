import time
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.messages import Session_pb2, Base_pb2
from kortex_api.RouterClient import RouterClient
from kortex_api.TCPTransport import TCPTransport
from kortex_api.autogen.client_stubs.SessionClientRpc import SessionClient
import math
import transforms3d as t3d
import numpy as np



class Robot():
    def __init__(self, ip='192.168.1.10', username='admin', password='admin'):
        super().__init__()
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
            session_info.session_inactivity_timeout = 60000   # (milliseconds)
            session_info.connection_inactivity_timeout = 2000 # (milliseconds)
            self.session_manager.CreateSession(session_info)
            
            self.base = BaseClient(self.router)
            print("Connected to the robot successfully")
        except Exception as e:
            print(f"Failed to connect to the robot: {e}")
    
    def move(self, pose):
        rpy = t3d.euler.mat2euler(pose[:3, :3], 'sxyz')
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
        
        self.base.ExecuteAction(waypoint)
        
    def get_pose(self):
        feedback = self.base.GetMeasuredCartesianPose()
        xyz = [feedback.x, feedback.y, feedback.z]
        rpy = [
            math.radians(feedback.theta_x),
            math.radians(feedback.theta_y),
            math.radians(feedback.theta_z)
        ]
        # print('get_rpy', rpy)
        pose = t3d.affines.compose(
            xyz,
            t3d.euler.euler2mat(rpy[0], rpy[1], rpy[2], 'sxyz'),
            [1, 1, 1]
        )
        return pose
    
    
    def go_home(self):
        pose = t3d.affines.compose(
            [0.3, 0.1, 0.2],
            t3d.euler.euler2mat(
                math.radians(180),
                math.radians(0),
                math.radians(0),
                'sxyz'
            ),
            [1, 1, 1]
        )
        self.move(pose) 

    def go_pos_not_effect_camera(self):
        pose = t3d.affines.compose(
            [0.3, 0.1, 0.2],
            t3d.euler.euler2mat(
                math.radians(180),
                math.radians(0),
                math.radians(0),
                'sxyz'
            ),
            [1, 1, 1]
        )
        self.move(pose)         
    
    def disconnect_robot(self):
        try:
            self.session_manager.CloseSession()
            self.transport.disconnect()
            print("Disconnected from the robot")
        except Exception as e:
            print(f"Failed to disconnect from the robot: {e}")
            
    def open_g():
        pass
    def close_g():
        pass
    
            
if __name__ == '__main__':
    robot = Robot(ip='192.168.1.10')
    
    pose = robot.get_pose()
    # print(pose)
    # pose[2, 3] += 0.005
    # print(pose)
    # robot.move(pose)
    robot.go_home()
    time.sleep(3.3)
    
    robot.disconnect_robot()
    

