import time

import cv2
import numpy as np

from gsmini import gs3drecon, gsdevice
import time

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2
from kortex_api.autogen.messages import BaseCyclic_pb2

class GripperControl:
    def __init__(self, router, router_real_time):

        ###########################################################################################
        # UDP and TCP sessions are used in this example.
        # TCP is used to perform the change of servoing mode
        # UDP is used for cyclic commands.
        #
        # 2 sessions have to be created: 1 for TCP and 1 for UDP
        ###########################################################################################

        self.router = router
        self.router_real_time = router_real_time

        # Create base client using TCP router
        self.base = BaseClient(self.router)

        # Create base cyclic client using UDP router.
        self.base_cyclic = BaseCyclicClient(self.router_real_time)

        # Create base cyclic command object.
        self.base_command = BaseCyclic_pb2.Command()
        self.base_command.frame_id = 0
        self.base_command.interconnect.command_id.identifier = 0
        self.base_command.interconnect.gripper_command.command_id.identifier = 0

        # Add motor command to interconnect's cyclic
        self.motorcmd = self.base_command.interconnect.gripper_command.motor_cmd.add()

        # Set gripper's initial position velocity and force
        base_feedback = self.base_cyclic.RefreshFeedback()
        self.motorcmd.position = base_feedback.interconnect.gripper_feedback.motor[0].position
        self.motorcmd.velocity = 0
        self.motorcmd.force = 100

        for actuator in base_feedback.actuators:
            self.actuator_command = self.base_command.actuators.add()
            self.actuator_command.position = actuator.position
            self.actuator_command.velocity = 0.0
            self.actuator_command.torque_joint = 0.0
            self.actuator_command.command_id = 0

        # Save servoing mode before changing it
        self.previous_servoing_mode = self.base.GetServoingMode()

        # Set base in low level servoing mode
        servoing_mode_info = Base_pb2.ServoingModeInformation()
        servoing_mode_info.servoing_mode = Base_pb2.LOW_LEVEL_SERVOING
        self.base.SetServoingMode(servoing_mode_info)

    def Cleanup(self):
        """
            Restore arm's servoing mode to the one that
            was effective before running the example.

        """
        # Restore servoing mode to the one that was in use before running the example
        self.base.SetServoingMode(self.previous_servoing_mode)

    def trackContact(self, current_contact):
        desired_contact = 1.1
        try:
            base_feedback = self.base_cyclic.Refresh(self.base_command)
            target_position = base_feedback.interconnect.gripper_feedback.motor[0].position + 1.0*(desired_contact-current_contact)
            if target_position > 30:
                target_position = 30
            elif target_position < 0:
                target_position = 0
            self.motorcmd.velocity = 1.0
            self.motorcmd.position = target_position

        except Exception as e:
            assert False, "An error occurred: %s" % e
        return
    
    def goToPosition(self, target_position):
        while True:
            try:
                base_feedback = self.base_cyclic.Refresh(self.base_command)

                # Calculate speed according to position error (target position VS current position)
                position_error = target_position - base_feedback.interconnect.gripper_feedback.motor[0].position

                # If positional error is small, stop gripper
                if abs(position_error) < 1.5:
                    position_error = 0
                    self.motorcmd.velocity = 0
                    self.base_cyclic.Refresh(self.base_command)
                    return
                else:
                    self.motorcmd.velocity = abs(position_error)
                    if self.motorcmd.velocity > 100.0:
                        self.motorcmd.velocity = 100.0
                    self.motorcmd.position = target_position

            except Exception as e:
                assert False, "An error occurred: %s" % e

    def goHome(self):
        self.goToPosition(0)
        return
    
    def initialGrasp(self, current_contact):
        desired_contact = 1.1
        try:
            base_feedback = self.base_cyclic.Refresh(self.base_command)

            if current_contact > desired_contact or abs(base_feedback.interconnect.gripper_feedback.motor[0].position-70)<3:
                self.motorcmd.velocity = 0
                self.motorcmd.position = base_feedback.interconnect.gripper_feedback.motor[0].position
                self.base_cyclic.Refresh(self.base_command)
                return True
            else:
                self.motorcmd.velocity = 10
                self.motorcmd.position = 70
                self.base_cyclic.Refresh(self.base_command)
                return False

        except Exception as e:
            assert False, "An error occurred: %s" % e
    

def main():
    # Set flags
    DEVICE = "cuda"
    MASK_MARKERS_FLAG = True
    CALCULATE_DEPTH_FLAG = True
    CALCULATE_SHEAR_FLAG = True
    VISUALIZE = False

    # the device ID can change after unplugging and changing the usb ports.
    # on linux run, v4l2-ctl --list-devices, in the terminal to get the device ID for camera
    # cam_id = gsdevice.get_camera_id("GelSight Mini")
    cam_id = 0
    dev = gsdevice.Camera(
        cam_id,
        calcDepth=CALCULATE_DEPTH_FLAG,
        calcShear=CALCULATE_SHEAR_FLAG,
        device=DEVICE,
        maskMarkersFlag=MASK_MARKERS_FLAG,
    )

    print("press q on image to exit")

    if CALCULATE_DEPTH_FLAG and VISUALIZE:
        # """ use this to plot just the 3d """
        vis3d = gs3drecon.Visualize3D(dev.imgw, dev.imgh, "", dev.mmpp)

    if CALCULATE_SHEAR_FLAG:
        color = np.random.randint(0, 255, (100, 3))
        init_markers = dev.get_initial_markers()

    # Import the utilities helper module
    import argparse
    import src.utils as utilities

    # Parse arguments
    parser = argparse.ArgumentParser()
    args = utilities.parseConnectionArguments(parser)

    # Create connection to the device and get the router
    with utilities.DeviceConnection.createTcpConnection(args) as router:

        with utilities.DeviceConnection.createUdpConnection(args) as router_real_time:
            gripper = GripperControl(router, router_real_time)
            gripper.goHome()
            while True:
                f1 = dev.get_image()
                dev.process_image(f1)
                if CALCULATE_SHEAR_FLAG:
                    markers = dev.get_markers()
                    if VISUALIZE:
                        for i, new in enumerate(markers):
                            a, b = new.ravel()
                            ix = int(init_markers[i, 0])
                            iy = int(init_markers[i, 1])
                            f1 = cv2.arrowedLine(
                                f1,
                                (ix, iy),
                                (int(a), int(b)),
                                (255, 255, 255),
                                thickness=1,
                                line_type=cv2.LINE_8,
                                tipLength=0.15,
                            )
                            f1 = cv2.circle(f1, (int(a), int(b)), 5, color[i].tolist(), -1)
                if CALCULATE_DEPTH_FLAG:
                    dm = dev.get_depth()
                    if VISUALIZE:
                        vis3d.update(dm)
                    current_contact = dm.copy()
                    inds = np.ones_like(current_contact)
                    inds[np.where(current_contact < 1.0)] = 0
                    current_contact[np.where(current_contact < 1.0)] = 0
                    current_contact = np.sum(current_contact) / max(np.sum(inds),1)
                    # print("Current contact: ", current_contact)
                    # gripper.trackContact(current_contact)

                cv2.imshow("Image", f1)
                key = cv2.waitKey(1)
                if key & 0xFF == ord("q"):
                    gripper.goHome()
                    break
                elif key & 0xFF == ord("h"):
                    gripper.goHome()
                elif key & 0xFF == ord("s"):
                    while not gripper.initialGrasp(current_contact):
                        f1 = dev.get_image()
                        dev.process_image(f1)
                        dm = dev.get_depth()
                        if VISUALIZE:
                            vis3d.update(dm)
                        current_contact = dm.copy()
                        inds = np.ones_like(current_contact)
                        inds[np.where(current_contact < 1.0)] = 0
                        current_contact[np.where(current_contact < 1.0)] = 0
                        current_contact = np.sum(current_contact) / max(np.sum(inds),1)

            gripper.Cleanup()
    dev.disconnect()


if __name__ == "__main__":
    main()
