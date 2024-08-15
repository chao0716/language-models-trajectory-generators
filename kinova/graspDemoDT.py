import cv2
import numpy as np

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2
from kortex_api.autogen.messages import BaseCyclic_pb2

import time
import src.utils as utilities
from src.getDensity import getFilteredDensity
from src.tracker import Tracker


class Robot:
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
        self.motorcmd.position = base_feedback.interconnect.gripper_feedback.motor[
            0
        ].position
        self.motorcmd.velocity = 0
        self.motorcmd.force = 100

        for actuator_feedback in base_feedback.actuators:
            self.actuator_command = self.base_command.actuators.add()
            self.actuator_command.position = actuator_feedback.position
            self.actuator_command.velocity = 0.0
            self.actuator_command.torque_joint = 0.0
            self.actuator_command.command_id = 0
        # Save servoing mode before changing it
        self.previous_servoing_mode = self.base.GetServoingMode()
        self.setHighLevelServoing()

    def setLowLevelServoing(self):
        # Set base in low level servoing mode
        servoing_mode_info = Base_pb2.ServoingModeInformation()
        servoing_mode_info.servoing_mode = Base_pb2.LOW_LEVEL_SERVOING
        self.base.SetServoingMode(servoing_mode_info)

    def getFeedback(self):
        feedback = self.base_cyclic.RefreshFeedback()
        return np.array(
            (
                feedback.base.tool_pose_x,
                feedback.base.tool_pose_y,
                feedback.base.tool_pose_z,
                feedback.base.tool_pose_theta_x,
                feedback.base.tool_pose_theta_y,
                feedback.base.tool_pose_theta_z,
                feedback.interconnect.gripper_feedback.motor[0].position,
            )
        )

    def setHighLevelServoing(self):
        base_servo_mode = Base_pb2.ServoingModeInformation()
        base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
        self.base.SetServoingMode(base_servo_mode)

    def cartesianVelocity(self, x, y, z):
        command = Base_pb2.TwistCommand()

        command.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_BASE
        command.duration = 0

        twist = command.twist
        twist.linear_x = 0
        twist.linear_y = 0
        twist.linear_z = z
        twist.angular_x = 0
        twist.angular_y = 0
        twist.angular_z = 0

        self.base.SendTwistCommand(command)

    def stopMotion(self):
        self.base.Stop()

    def Cleanup(self):
        """
        Restore arm's servoing mode to the one that
        was effective before running the example.

        """
        # Restore servoing mode to the one that was in use before running the example
        self.base.SetServoingMode(self.previous_servoing_mode)

    def gripperPosition(self, target_position, velocity=1):
        self.setLowLevelServoing()
        base_feedback = self.base_cyclic.RefreshFeedback()
        for i, actuator_feedback in enumerate(base_feedback.actuators):
            self.base_command.actuators[i].position = actuator_feedback.position
        try:
            while True:
                # Calculate speed according to position error (target position VS current position)
                position_error = (
                    target_position
                    - base_feedback.interconnect.gripper_feedback.motor[0].position
                )

                # If positional error is small, stop gripper
                if abs(position_error) < 1.5:
                    position_error = 0
                    self.motorcmd.velocity = 0
                    self.base_cyclic.Refresh(self.base_command)
                    self.setHighLevelServoing()
                    return
                else:
                    self.motorcmd.velocity = velocity
                    self.motorcmd.position = target_position
                base_feedback = self.base_cyclic.Refresh(self.base_command)
            
        except Exception as e:
            assert False, "An error occurred: %s" % e

    def gripperHome(self):
        self.gripperPosition(0, 100)
        return

    def initialGrasp(self, current_contact):
        desired_contact = 4.0
        base_feedback = self.base_cyclic.RefreshFeedback()
        for i, actuator_feedback in enumerate(base_feedback.actuators):
            self.base_command.actuators[i].position = actuator_feedback.position
        try:
            if (
                current_contact > desired_contact
                or base_feedback.interconnect.gripper_feedback.motor[0].position > 95
                or abs(current_contact - desired_contact) < 0.5
            ):
                self.motorcmd.velocity = 0
                self.motorcmd.position = (
                    base_feedback.interconnect.gripper_feedback.motor[0].position
                )
                self.base_cyclic.Refresh(self.base_command)
                return True
            else:
                self.motorcmd.velocity = 1
                self.motorcmd.position = 95
                self.base_cyclic.Refresh(self.base_command)
                return False

        except Exception as e:
            assert False, "An error occurred: %s" % e


def main():
    VISUALIZE_RAW = False
    VISUALIZE_ARROW = True

    cam_id = 0
    cam = utilities.Camera(cam_id)
    tracker = Tracker(adaptive=True, cuda=False)  # cuda=True if using opencv cuda

    print("press q on image to exit")

    import argparse

    # Parse arguments
    parser = argparse.ArgumentParser()
    args = utilities.parseConnectionArguments(parser)

    # Create connection to the device and get the router
    with utilities.DeviceConnection.createTcpConnection(args) as router:

        with utilities.DeviceConnection.createUdpConnection(args) as router_real_time:
            robot = Robot(router, router_real_time)
            robot.gripperHome()
            while True:
                f1 = cam.getFrame()
                f1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
                flow = tracker.track(f1)
                density = getFilteredDensity(flow, use_cuda=True)
                density = (density + 1.0) / 1.0
                density = (density * 255.0).astype("uint8")
                threshold, _ = utilities.getContactBoundary(density)
                density[np.where(density < threshold)] = 0
                current_contact = density.astype(np.float32)
                density = cv2.applyColorMap(density, cv2.COLORMAP_HOT)

                inds = np.ones_like(current_contact)
                inds[np.where(current_contact < threshold)] = 0.0
                current_contact = np.sum(current_contact) / max(np.sum(inds), 1)
                # print("Current contact: ", current_contact)

                cv2.imshow("Normal Contact", density)
                if VISUALIZE_ARROW:
                    arrows = utilities.put_optical_flow_arrows_on_image(
                        np.zeros_like(density), flow[15:-15, 15:-15, :]
                    )
                    cv2.imshow("Shear Contact", arrows)
                if VISUALIZE_RAW:
                    cv2.imshow("Image", f1)
                key = cv2.waitKey(1)
                if key & 0xFF == ord("q"):
                    robot.gripperHome()
                    break
                elif key & 0xFF == ord("h"):
                    robot.gripperHome()
                elif key & 0xFF == ord("s"):
                    robot.setLowLevelServoing()
                    while not robot.initialGrasp(current_contact):
                        f1 = cam.getFrame()
                        f1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
                        flow = tracker.track(f1)
                        density = getFilteredDensity(flow, use_cuda=True)
                        density = (density + 1.0) / 1.0
                        density = (density * 255.0).astype("uint8")
                        threshold, _ = utilities.getContactBoundary(density)
                        density[np.where(density < threshold)] = 0
                        current_contact = density.astype(np.float32)
                        density = cv2.applyColorMap(density, cv2.COLORMAP_HOT)

                        inds = np.ones_like(current_contact)
                        inds[np.where(current_contact < threshold)] = 0.0
                        current_contact = np.sum(current_contact) / max(np.sum(inds), 1)
                    robot.setHighLevelServoing()
                    time.sleep(1.0)
                    grasp_frame = cv2.cvtColor(cam.getFrame(), cv2.COLOR_BGR2GRAY)
                    follow_tracker = Tracker(adaptive=True, cuda=False)
                    follow_flow = follow_tracker.track(grasp_frame)
                    flow_vis = np.zeros_like(density)
                    cv2.imshow("Follow Contact", flow_vis)
                    start_position = robot.getFeedback()[:3]
                    release_ths = 1.0
                    while cv2.waitKey(1) & 0xFF != ord("s"):
                        grasp_frame = cv2.cvtColor(cam.getFrame(), cv2.COLOR_BGR2GRAY)

                        # visualize the normal
                        flow = tracker.track(grasp_frame)
                        density = getFilteredDensity(flow, use_cuda=True)
                        density = (density + 1.0) / 1.0
                        density = (density * 255.0).astype("uint8")
                        threshold, _ = utilities.getContactBoundary(density)
                        density[np.where(density < threshold)] = 0
                        density = cv2.applyColorMap(density, cv2.COLORMAP_HOT)
                        cv2.imshow("Normal Contact", density)

                        # visualize the follow vector
                        follow_flow = follow_tracker.track(grasp_frame)
                        flow_vis = np.zeros_like(density)
                        follow_flow = follow_flow.reshape(-1, 2).mean(axis=0)
                        print(follow_flow[1])  # 2 as ths
                        cv2.arrowedLine(
                            flow_vis,
                            pt1=(flow_vis.shape[1] // 2, flow_vis.shape[0] // 2),
                            pt2=(
                                flow_vis.shape[1] // 2 + int(10 * follow_flow[0]),
                                flow_vis.shape[0] // 2 + int(-10 * follow_flow[1]),
                            ),
                            color=(0, 128, 128),
                            thickness=1,
                            tipLength=0.3,
                        )
                        cv2.imshow("Follow Contact", flow_vis)
                        current_position = robot.getFeedback()[:3]
                        follow_flow_mag = np.linalg.norm(follow_flow)
                        print(follow_flow_mag)
                        if (
                            follow_flow[1] > 0.0 and follow_flow_mag > release_ths*0.7
                        ):
                            break
                        elif (
                            follow_flow[1] < -0.0 and follow_flow_mag > 0.8
                        ):
                            release_ths += follow_flow_mag
                            robot.gripperPosition(robot.getFeedback()[6] + 3)
                            time.sleep(1.5)
                            follow_tracker = Tracker(adaptive=True, cuda=False)
                            follow_flow = follow_tracker.track(cv2.cvtColor(cam.getFrame(), cv2.COLOR_BGR2GRAY))
                    robot.stopMotion()
                    robot.gripperHome()
                    cv2.destroyWindow("Follow Contact")

            robot.Cleanup()
    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
