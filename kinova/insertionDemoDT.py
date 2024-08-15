import threading
from math import acos

import cv2
import numpy as np
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2, BaseCyclic_pb2
from numpy import linalg as LA
from scipy.spatial.transform import Rotation as R

import src.utils as utilities
from src.getDensity import getFilteredDensity
from src.tracker import Tracker
from scipy import ndimage

# Maximum allowed waiting time during actions (in seconds)
TIMEOUT_DURATION = 20


def check_for_end_or_abort(e):
    """Return a closure checking for END or ABORT notifications

    Arguments:
    e -- event to signal when the action is completed
        (will be set when an END or ABORT occurs)
    """

    def check(notification, e=e):
        if (
            notification.action_event == Base_pb2.ACTION_END
            or notification.action_event == Base_pb2.ACTION_ABORT
        ):
            print("EVENT : " + Base_pb2.ActionEvent.Name(notification.action_event))
            e.set()

    return check


def get_pose(mask):
    def PCA(pts):
        pts = pts.reshape(-1, 2).astype(np.float64)
        mv = np.mean(pts, 0).reshape(2, 1)
        pts -= mv.T
        w, v = LA.eig(np.dot(pts.T, pts))
        w_max = np.max(w)
        w_min = np.min(w)

        col = np.where(w == w_max)[0]
        if len(col) > 1:
            col = col[-1]
        V_max = v[:, col]

        col_min = np.where(w == w_min)[0]
        if len(col_min) > 1:
            col_min = col_min[-1]
        V_min = v[:, col_min]

        return V_max, V_min, w_max, w_min

    mask = mask.astype(np.uint8)
    # cv2.imshow("mask", mask * 255)
    # Display the resulting frame
    coors = np.where(mask == 1)
    X = coors[1].reshape(-1, 1)
    y = coors[0].reshape(-1, 1)
    if len(X) < 50:
        return (
            0.0,
            0.0,
            0.0,
        )
    pts = np.concatenate([X, y], axis=1)
    v_max, _, _, _ = PCA(pts)
    if v_max[1] < 0:
        v_max *= -1
    m = np.mean(pts, 0).reshape(-1)

    theta = acos(v_max[0].item() / (v_max[0].item() ** 2 + v_max[1].item() ** 2) ** 0.5)
    return (
        m[0] / mask.shape[1],
        m[1] / mask.shape[0],
        theta / np.pi * 180.0,
    )


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
            )
        )

    def cartesian_delta_action(self, delta, speed=0.05):
        action = Base_pb2.Action()
        action.name = "Cartesian delta action movement"
        action.application_data = ""

        feedback = self.base_cyclic.RefreshFeedback()

        action.reach_pose.constraint.speed.translation = speed  # (m/s)

        cartesian_pose = action.reach_pose.target_pose
        cartesian_pose.x = feedback.base.tool_pose_x + delta[0]  # (meters)
        cartesian_pose.y = feedback.base.tool_pose_y + delta[1]  # (meters)
        cartesian_pose.z = feedback.base.tool_pose_z + delta[2]  # (meters)
        cartesian_pose.theta_x = feedback.base.tool_pose_theta_x  # (degrees)
        cartesian_pose.theta_y = feedback.base.tool_pose_theta_y  # (degrees)
        cartesian_pose.theta_z = feedback.base.tool_pose_theta_z  # (degrees)
        if len(delta) == 6:
            cartesian_pose.theta_x += delta[3]  # (degrees)
            cartesian_pose.theta_y += delta[4]  # (degrees)
            cartesian_pose.theta_z += delta[5]  # (degrees)

        e = threading.Event()
        notification_handle = self.base.OnNotificationActionTopic(
            check_for_end_or_abort(e), Base_pb2.NotificationOptions()
        )

        self.base.ExecuteAction(action)

        finished = e.wait(TIMEOUT_DURATION)
        self.base.Unsubscribe(notification_handle)

        if finished:
            pass
            # print("Cartesian delta movement completed")
        else:
            print("Timeout on action notification wait")
        return

    def cartesian_goal_action(self, goal, speed=0.01):
        if len(goal) != 6:
            print("Goal must be a 6 element list")
            return False
        action = Base_pb2.Action()
        action.name = "Cartesian goal action movement"
        action.application_data = ""

        action.reach_pose.constraint.speed.translation = speed  # (m/s)
        cartesian_pose = action.reach_pose.target_pose
        cartesian_pose.x = goal[0]  # (meters)
        cartesian_pose.y = goal[1]  # (meters)
        cartesian_pose.z = goal[2]  # (meters)
        cartesian_pose.theta_x = goal[3]  # (degrees)
        cartesian_pose.theta_y = goal[4]  # (degrees)
        cartesian_pose.theta_z = goal[5]  # (degrees)

        e = threading.Event()
        notification_handle = self.base.OnNotificationActionTopic(
            check_for_end_or_abort(e), Base_pb2.NotificationOptions()
        )

        self.base.ExecuteAction(action)

        finished = e.wait(TIMEOUT_DURATION)
        self.base.Unsubscribe(notification_handle)

        if finished:
            pass
            # print("Cartesian goal movement completed")
        else:
            print("Timeout on action notification wait")
        return

    def deltaAction2cartesianGoal(self, delta):
        if len(delta) != 3:
            print("Delta must be a 3 element list")
            return False

        feedback = self.base_cyclic.RefreshFeedback()

        rotation = R.from_euler(
            "xyz",
            [
                feedback.base.tool_pose_theta_x,
                feedback.base.tool_pose_theta_y,
                feedback.base.tool_pose_theta_z,
            ],
            degrees=True,
        )
        delta_in_base = rotation.apply(np.array((delta[:3])).flatten())
        return np.array(
            [
                feedback.base.tool_pose_x + delta_in_base[0],
                feedback.base.tool_pose_y + delta_in_base[1],
                feedback.base.tool_pose_z + delta_in_base[2],
                feedback.base.tool_pose_theta_x,
                feedback.base.tool_pose_theta_y,
                feedback.base.tool_pose_theta_z,
            ]
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

    def gripperPosition(self, target_position):
        gripper_command = Base_pb2.GripperCommand()
        finger = gripper_command.gripper.finger.add()
        gripper_command.mode = Base_pb2.GRIPPER_POSITION
        finger.finger_identifier = 1
        finger.value = target_position
        self.base.SendGripperCommand(gripper_command)

    def gripperHome(self):
        self.gripperPosition(0)
        return

    def initialGrasp(self, current_contact):
        desired_contact = 4.0
        base_feedback = self.base_cyclic.RefreshFeedback()
        for i, actuator_feedback in enumerate(base_feedback.actuators):
            self.base_command.actuators[i].position = actuator_feedback.position
        try:
            if (
                current_contact > desired_contact
                or base_feedback.interconnect.gripper_feedback.motor[0].position > 98
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
                self.motorcmd.position = 98
                self.base_cyclic.Refresh(self.base_command)
                return False

        except Exception as e:
            assert False, "An error occurred: %s" % e


def main():
    VISUALIZE_RAW = False
    VISUALIZE_ARROW = True

    cam_id = 0
    mmpp = 0.0275
    sensor_center = np.array([225, 190])  # x, y
    recorded_robot_position = np.array(
        [
            0.72520846,
            0.28782326,
            0.1,
            90.0,
            -179.9,
            90.0,
        ]
    )
    recorded_d = 190.5306105688388
    recorded_line_theta = 89.95524165714389
    rotation_center = sensor_center.copy()
    rotation_center[0] += 30.0 / mmpp
    roi = np.array([[113, 63], [580, 18], [588, 466], [114, 396]], dtype="float32")
    cam = utilities.Camera(cam_id, roi=roi)
    tracker = Tracker(adaptive=True, cuda=False)  # cuda=True if using opencv cuda

    print("Press 's' to grasp, 'q' to exit, 'h' to home the gripper, 'g' to insert")

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
                # print(robot.getFeedback())
                f1 = cam.getFrame()
                f1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
                flow = tracker.track(f1)
                density = getFilteredDensity(flow, use_cuda=True)
                density = (density + 1.0) / 1.0
                density = (density * 255.0).astype("uint8")
                threshold, _ = utilities.getContactBoundary(density)
                density[np.where(density < threshold)] = 0
                center_of_mass = ndimage.center_of_mass(density)
                mask = np.ones_like(density)
                mask[np.where(density < threshold)] = 0
                x, y, theta = get_pose(mask)
                if density.max() > 15:
                    print(
                        "x: ",
                        x,
                        " y: ",
                        y,
                        " theta: ",
                        theta,
                        "center: ",
                        (center_of_mass[1], center_of_mass[0]),
                    )
                current_contact = density.astype(np.float32)
                density = cv2.applyColorMap(density, cv2.COLORMAP_HOT)

                inds = np.ones_like(current_contact)
                inds[np.where(current_contact < threshold)] = 0.0
                # print("Current contact: ", current_contact)

                cv2.imshow("Normal Contact", density)

                if VISUALIZE_ARROW:
                    arrows = utilities.put_optical_flow_arrows_on_image(
                        np.zeros_like(density), flow[15:-15, 15:-15, :], nhhd=True
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
                elif key & 0xFF == ord("l"):  # learn the current hole location
                    # adjust orientation
                    robot.gripperHome()
                    current_feedback = robot.getFeedback()
                    robot.cartesian_goal_action(
                        [
                            current_feedback[0],
                            current_feedback[1],
                            current_feedback[2],
                            90.0,
                            -179.9,
                            90.0,
                        ],
                        speed=0.05,
                    )  # set norminal orientation
                    # grasp
                    robot.setLowLevelServoing()
                    while not robot.initialGrasp(
                        np.sum(current_contact) / max(np.sum(inds), 1)
                    ):

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

                    robot.setHighLevelServoing()
                    while cv2.waitKey(1) & 0xFF != ord("q"):
                        print("Insert into the hole, then press q to continue")
                        f1 = cam.getFrame()
                        f1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
                        flow = tracker.track(f1)
                    recorded_robot_position = robot.getFeedback()[:3]
                    recorded_robot_position[2] = 0.1  # check value
                    print("Recorded robot position: ", recorded_robot_position)
                    grasp_frame = cv2.cvtColor(cam.getFrame(), cv2.COLOR_BGR2GRAY)

                    density = getFilteredDensity(
                        tracker.track(grasp_frame), use_cuda=True
                    )
                    density = (density + 1.0) / 1.0
                    density = (density * 255.0).astype("uint8")
                    threshold, _ = utilities.getContactBoundary(density)
                    threshold += 5
                    density[np.where(density < threshold)] = 0
                    mask = np.ones_like(density)
                    mask[np.where(density < threshold)] = 0
                    line_x, line_y, recorded_line_theta = get_pose(mask)
                    slope = np.tan(np.pi / 180.0 * recorded_line_theta)
                    recorded_d = abs(
                        np.dot(
                            np.array([1, slope]),
                            np.array(
                                [
                                    line_x - rotation_center[0],
                                    line_y - rotation_center[1],
                                ]
                            ),
                        )
                        / np.linalg.norm(np.array([1, slope]))
                    )
                    print("Recorded d ", recorded_d, " theta: ", recorded_line_theta)
                    robot.gripperHome()
                elif key & 0xFF == ord("s"):  # grasp
                    robot.setLowLevelServoing()
                    while not robot.initialGrasp(
                        np.sum(current_contact) / max(np.sum(inds), 1)
                    ):

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
                    print('max contact:',np.sum(current_contact) / max(np.sum(inds), 1))
                    robot.setHighLevelServoing()
                elif key & 0xFF == ord("g"):  # insert
                    grasp_frame = cv2.cvtColor(cam.getFrame(), cv2.COLOR_BGR2GRAY)

                    density = getFilteredDensity(
                        tracker.track(grasp_frame), use_cuda=True
                    )
                    density = (density + 1.0) / 1.0
                    density = (density * 255.0).astype("uint8")
                    threshold, _ = utilities.getContactBoundary(density)
                    threshold += 5
                    density[np.where(density < threshold)] = 0
                    mask = np.ones_like(density)
                    mask[np.where(density < threshold)] = 0
                    x, y, theta = get_pose(mask)
                    current_feedback = robot.getFeedback()
                    robot.cartesian_goal_action(
                        [
                            current_feedback[0],
                            current_feedback[1],
                            current_feedback[2],
                            theta + recorded_line_theta - 90.0,
                            -179.9,
                            90.0,
                        ],
                        speed=0.05,
                    )  # check first two angles
                    print("Finished rotating")
                    print("x: ", x, " y: ", y, " theta: ", theta)

                    # # finished rotate, now horizontal move
                    slope = np.tan(np.pi / 180.0 * (theta))
                    current_d = abs(
                        np.dot(
                            np.array([1, slope]),
                            np.array([x - rotation_center[0], y - rotation_center[1]]),
                        )
                        / np.linalg.norm(np.array([1, slope]))
                    )
                    current_feedback = robot.getFeedback()
                    robot.cartesian_goal_action(
                        [
                            recorded_robot_position[0],
                            recorded_robot_position[1],
                            recorded_robot_position[2],
                            current_feedback[3],
                            current_feedback[4],
                            current_feedback[5],
                        ],
                        speed=0.05,
                    )
                    robot.cartesian_delta_action(
                        [
                            0.0,
                            0.0,
                            np.sin(np.pi / 180.0 * recorded_line_theta)
                            * (recorded_d - current_d)
                            * mmpp
                            / 1000.0,
                        ],
                        speed=0.01,
                    )
                    # start moving downward
                    follow_tracker = Tracker(adaptive=True, cuda=False)
                    grasp_frame = cv2.cvtColor(cam.getFrame(), cv2.COLOR_BGR2GRAY)
                    follow_flow = follow_tracker.track(grasp_frame)
                    flow_vis = np.zeros_like(density)
                    cv2.imshow("Follow Contact", flow_vis)
                    start_position = robot.getFeedback()[:3]
                    shear_stop_ths = 1.5  # 2 as ths
                    vel_down = -0.01
                    print("Start moving down")
                    while True:
                        key = cv2.waitKey(1)
                        if key & 0xFF == ord("q"):
                            break
                        grasp_frame = cv2.cvtColor(cam.getFrame(), cv2.COLOR_BGR2GRAY)

                        follow_flow = follow_tracker.track(grasp_frame)
                        flow_vis = np.zeros_like(density)
                        follow_flow = follow_flow.reshape(-1, 2).mean(axis=0)
                        follow_flow_mag = np.linalg.norm(follow_flow)
                        # print(follow_flow_mag)
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
                        if (
                            follow_flow_mag < shear_stop_ths
                            and current_position[2] - start_position[2] > -0.1
                        ):
                            robot.cartesianVelocity(
                                0.0,
                                0.0,
                                vel_down + follow_flow_mag / shear_stop_ths * vel_down,
                            )
                        else:
                            robot.cartesianVelocity(0.0, 0.0, 0.0)
                            print(
                                "Stopping motion ",
                                "follow_flow_mag: ",
                                follow_flow_mag,
                                " distance travelled ",
                                current_position[2] - start_position[2],
                            )
                            break
                    robot.stopMotion()
                    robot.gripperHome()

            robot.Cleanup()
    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
