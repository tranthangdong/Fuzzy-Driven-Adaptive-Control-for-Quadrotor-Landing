# -----------------------------------------------------------------------------------------------
#  offboard_control.py
#  Description: This file contains utility functions for offboard control in PX4.
#  Author:  Dong LT. Tran
#  Email:   tranthangdong@duytan.edu.vn
#  Date:    2024-06-15
# -----------------------------------------------------------------------------------------------
#!/usr/bin/env python3
import os
import sys
# ----------------------------------------------------------------------------------------------
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
# ----------------------------------------------------------------------------------------------
import rclpy
import math
import cv2
import numpy as np
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy, QoSHistoryPolicy
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleStatus, VehicleAttitude, VehicleLocalPosition

# Gazebo harmonic library directly connected
# -----------------------------------------------------------------------------------------------
from gz.transport13 import Node as GzNode
from gz.msgs10.image_pb2 import Image as GzImage
from gz.msgs10.pose_v_pb2 import Pose_V as GzPoseV

from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor

def public_attitude_callback(self, msg):
        """Callback receives Quaternion from PX4 and converts to Euler angle (Roll, Pitch)"""
        qw = msg.q[0]
        qx = msg.q[1]
        qy = msg.q[2]
        qz = msg.q[3]

        # Algorithm for converting Quaternion to Euler angle (Roll/Pitch)
        # 1. Calculate the Roll angle (Angle of inclination of the wing to the X-axis).
        sinr_cosp = 2.0 * (qw * qx + qy * qz)
        cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
        self.drone_roll = math.atan2(sinr_cosp, cosr_cosp)

        # 2. Calculate the Pitch Angle (Angle of the front end to the Y-axis).
        sinp = 2.0 * (qw * qy - qz * qx)
        if abs(sinp) >= 1:
            self.drone_pitch = math.copysign(math.pi / 2.0, sinp)
        else:
            self.drone_pitch = math.asin(sinp)

def publish_offboard_control_mode(self):
        msg = OffboardControlMode()
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_control_mode_pub.publish(msg)

def publish_trajectory_setpoint(self, x, y, z):
        msg = TrajectorySetpoint()
        msg.position = [float(x), float(y), float(z)]
        msg.yaw = 0.0
        msg.velocity = [float('nan'), float('nan'), float('nan')]
        msg.acceleration = [float('nan'), float('nan'), float('nan')]
        msg.jerk = [float('nan'), float('nan'), float('nan')]
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_setpoint_pub.publish(msg)

def publish_vehicle_command(self, command, param1=0.0, param2=0.0):
        msg = VehicleCommand()
        msg.param1 = param1
        msg.param2 = param2
        msg.command = command
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.vehicle_command_pub.publish(msg)

def gazebo_world_pose_callback(self, msg):
            """Callback reveals the absolute position (Ground Truth) of the Quadrotor from Gazebo World"""
            try:
                # Browse through the list of models currently available in the Gazebo world.
                for pose in msg.pose:
                    if pose.name == 'x500_mono_cam_down_0':
                        self.world_gt_x = pose.position.x
                        self.world_gt_y = pose.position.y
                        self.world_gt_z = pose.position.z
                        break
            except Exception as e:
                pass
    
def local_position_callback(self, msg):
        """Callback reveals the local position (NED Frame) estimated from PX4's EKF2"""
        self.drone_x = msg.x
        self.drone_y = msg.y
        self.drone_z = msg.z  # Note: In the NED frame, z should be negative when flying upwards