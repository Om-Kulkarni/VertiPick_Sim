#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from pynput import keyboard

import pybullet
import pybullet_utils.bullet_client as bc
import pybullet_data
import numpy as np
from scipy.spatial.transform import Rotation as R
import threading
import time

# --- Constants ---
# You must change this to the actual path of your Panda URDF file
# For example: '/path/to/your/ros2_ws/src/franka_ros2/franka_description/robots/panda_arm.urdf'
URDF_PATH = "franka_urdf/panda.urdf" 
EE_LINK_NAME = "panda_hand_tcp_joint"  # The name of the end-effector link in your URDF

# Movement speeds
POS_INCREMENT = 0.02  # meters
ROT_INCREMENT = 0.05  # radians

# Gripper positions
GRIPPER_OPEN = 0.04
GRIPPER_CLOSED = 0.00


class PandaTeleopNode(Node):
    """
    A ROS 2 node for teleoperating a Franka Panda arm's end-effector via the keyboard.

    This node listens to keyboard presses to control the end-effector's position
    and orientation. It uses PyBullet for inverse kinematics (IK) to calculate the
    required joint angles and publishes them to the '/joint_command' topic as a
    sensor_msgs/msg/JointState message.
    """

    def __init__(self):
        """Initializes the node, publishers, subscribers, and the IK engine."""
        super().__init__('panda_teleop_node')
        self.get_logger().info("Panda Keyboard Teleop Node started. 🦾")

        # --- ROS 2 Communications ---
        self.joint_command_publisher = self.create_publisher(JointState, '/joint_command', 10)
        self.joint_state_subscriber = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_states_callback,
            10)

        # --- State Variables ---
        self.arm_joint_names = [f'panda_joint{i}' for i in range(1, 8)]
        self.gripper_joint_names = ['panda_finger_joint1', 'panda_finger_joint2']
        
        self.current_joint_positions = None
        self.target_ee_pos = np.zeros(3)
        self.target_ee_orn_quat = np.array([0., 0., 0., 1.]) # (x, y, z, w)
        self.target_gripper_pos = GRIPPER_OPEN
        self.initial_state_received = False

        # --- Inverse Kinematics (PyBullet) Setup ---
        self._setup_pybullet()

        # --- Keyboard Listener ---
        # Run the listener in a separate thread to not block rclpy.spin()
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener_thread = threading.Thread(target=self.listener.start)
        self.listener_thread.daemon = True
        self.listener_thread.start()

        self.print_instructions()

    def _setup_pybullet(self):
        """Initializes a PyBullet client and loads the robot URDF for IK."""
        self.get_logger().info(f"Loading URDF for IK: {URDF_PATH}")
        try:
            self.p = bc.BulletClient(connection_mode=pybullet.DIRECT)
            self.p.setAdditionalSearchPath(pybullet_data.getDataPath())
            self.robot_id = self.p.loadURDF(URDF_PATH, useFixedBase=True)
            self.num_joints = self.p.getNumJoints(self.robot_id)
            
            # Find the end-effector link index
            self.ee_link_index = None
            for i in range(self.num_joints):
                info = self.p.getJointInfo(self.robot_id, i)
                if info[12].decode('UTF-8') == EE_LINK_NAME:
                    self.ee_link_index = i
                    break
            
            if self.ee_link_index is None:
                raise Exception(f"End-effector link '{EE_LINK_NAME}' not found in URDF.")

            self.get_logger().info(f"PyBullet initialized successfully. End-effector link '{EE_LINK_NAME}' has index {self.ee_link_index}.")

        except Exception as e:
            self.get_logger().error(f"Failed to initialize PyBullet or load URDF: {e}")
            self.get_logger().error("Please ensure the URDF_PATH is correct and the file is accessible.")
            rclpy.shutdown()

    def print_instructions(self):
        """Prints the control instructions to the console."""
        print("\n" + "="*50)
        print(" Panda Teleop Control Panel")
        print("="*50)
        print(" W/S: Move Forward/Backward (+/- X)")
        print(" A/D: Strafe Left/Right (+/- Y)")
        print(" Space/C: Move Up/Down (+/- Z)")
        print(" Q/E: Rotate Left/Right (Yaw)")
        print("\n Z/X: Close/Open Gripper")
        print("\n Press 'Esc' to exit.")
        print("="*50)
        print("Waiting for initial joint states...")


    def joint_states_callback(self, msg: JointState):
        """
        Callback to update the current state of the robot's joints.
        On the first message, it computes the initial end-effector pose.
        """
        # Store a dictionary of current joint positions for easy lookup
        current_joint_positions_dict = {name: pos for name, pos in zip(msg.name, msg.position)}
        
        # Ensure we have all the necessary joint states
        try:
            self.current_joint_positions = [current_joint_positions_dict[name] for name in self.arm_joint_names]
        except KeyError:
            # This can happen if the /joint_states topic doesn't publish all arm joints initially
            return

        if not self.initial_state_received and self.current_joint_positions:
            self.get_logger().info("Initial joint states received. Ready to teleoperate.")
            
            # Reset PyBullet state to the real robot's state
            for i in range(len(self.current_joint_positions)):
                self.p.resetJointState(self.robot_id, i, self.current_joint_positions[i])

            # Calculate initial end-effector pose using Forward Kinematics (FK)
            ee_state = self.p.getLinkState(self.robot_id, self.ee_link_index)
            initial_pos = ee_state[0]
            initial_orn_quat = ee_state[1] # (x,y,z,w)
            
            # Set the target pose to the current pose
            self.target_ee_pos = np.array(initial_pos)
            self.target_ee_orn_quat = np.array(initial_orn_quat)

            self.initial_state_received = True

    def on_press(self, key):
        """Callback for keyboard presses, updates target pose and triggers IK."""
        if not self.initial_state_received:
            return  # Don't accept commands until we know where the arm is

        try:
            # --- Position Control ---
            if key.char == 'w':
                self.target_ee_pos[0] += POS_INCREMENT
            elif key.char == 's':
                self.target_ee_pos[0] -= POS_INCREMENT
            elif key.char == 'a':
                self.target_ee_pos[1] += POS_INCREMENT
            elif key.char == 'd':
                self.target_ee_pos[1] -= POS_INCREMENT
            elif key.char == 'c':
                self.target_ee_pos[2] -= POS_INCREMENT

            # --- Orientation Control (Yaw) ---
            elif key.char == 'q':
                self.update_yaw(ROT_INCREMENT)
            elif key.char == 'e':
                self.update_yaw(-ROT_INCREMENT)

            # --- Gripper Control ---
            elif key.char == 'z':
                self.target_gripper_pos = GRIPPER_CLOSED
                self.get_logger().info("Gripper: Closing")
            elif key.char == 'x':
                self.target_gripper_pos = GRIPPER_OPEN
                self.get_logger().info("Gripper: Opening")

        except AttributeError:
            # Handle special keys like spacebar
            if key == keyboard.Key.space:
                self.target_ee_pos[2] += POS_INCREMENT
            elif key == keyboard.Key.esc:
                self.get_logger().info("Escape key pressed. Shutting down.")
                self.listener.stop()
                rclpy.shutdown()
                return # Exit callback
        
        # For any movement key, calculate and publish the new joint command
        self.calculate_and_publish()

    def update_yaw(self, angle_change):
        """Updates the yaw of the target orientation."""
        # Create a rotation object for the yaw change around the Z-axis
        yaw_rotation = R.from_euler('z', angle_change)
        
        # Get current orientation as a Scipy Rotation object
        current_rotation = R.from_quat(self.target_ee_orn_quat)
        
        # Apply the yaw change
        new_rotation = yaw_rotation * current_rotation
        
        # Update the target orientation
        self.target_ee_orn_quat = new_rotation.as_quat()

    def calculate_and_publish(self):
        """
        Calculates IK for the target pose and publishes the command.
        """
        # Calculate Inverse Kinematics
        joint_poses = self.p.calculateInverseKinematics(
            self.robot_id,
            self.ee_link_index,
            self.target_ee_pos,
            self.target_ee_orn_quat,
            # We can add joint limits, rest poses etc. here if needed
        )
        
        # The result includes all joints, so we only take the arm joints
        arm_joint_poses = joint_poses[:len(self.arm_joint_names)]

        # --- Construct and Publish JointState Message ---
        joint_state_msg = JointState()
        joint_state_msg.header.stamp = self.get_clock().now().to_msg()
        
        # Combine arm and gripper commands
        joint_state_msg.name = self.arm_joint_names + self.gripper_joint_names
        joint_state_msg.position = list(arm_joint_poses) + [self.target_gripper_pos] * 2
        
        self.joint_command_publisher.publish(joint_state_msg)
        # self.get_logger().info(f"Publishing command for pos: {self.target_ee_pos.round(3)}")


def main(args=None):
    rclpy.init(args=args)
    teleop_node = PandaTeleopNode()
    
    try:
        rclpy.spin(teleop_node)
    except KeyboardInterrupt:
        pass
    finally:
        # Cleanup
        teleop_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()