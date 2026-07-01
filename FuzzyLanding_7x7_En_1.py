# -----------------------------------------------------------------------------------------------
#  FuzzyLanding_7x7.py
#  Description: Enhanced precision landing algorithm using 7x7 fuzzy controller 
#               (2 inputs: Altitude and Lateral Error, 1 output: Gain to adjust speed)
#               WITH WIND COMPENSATION AND ADAPTIVE CONTROL
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
from datetime import datetime
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy, QoSHistoryPolicy
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleStatus, VehicleAttitude, VehicleLocalPosition
from utils.offboard_control import public_attitude_callback, publish_offboard_control_mode, publish_trajectory_setpoint, publish_vehicle_command, gazebo_world_pose_callback, local_position_callback

# Gazebo harmonic library directly connected
# -----------------------------------------------------------------------------------------------
from gz.transport13 import Node as GzNode
from gz.msgs10.image_pb2 import Image as GzImage
from gz.msgs10.pose_v_pb2 import Pose_V as GzPoseV

from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor

class OffboardPrecisionLanding(Node):
    def __init__(self):
        super().__init__('drone_landing_node')

        # Initialize logging for flight data
        # ---------------------------------------------------------------------------------------
        self.log_data = []
        self.log_file_path = f"flight_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.log_headers = ['timestamp', 'drone_x', 'drone_y', 'drone_z', 
                        'target_x', 'target_y', 'target_z', 'aruco_detected',
                        'offset_x', 'offset_y', 'altitude', 'error', 
                        'p_gain', 'flight_state', 'wind_strength', 'oscillation']

        # Assign Functions to Methods of a Class
        # ---------------------------------------------------------------------------------------
        self.public_attitude_callback = public_attitude_callback.__get__(self)
        self.publish_offboard_control_mode = publish_offboard_control_mode.__get__(self)
        self.publish_trajectory_setpoint = publish_trajectory_setpoint.__get__(self)
        self.publish_vehicle_command = publish_vehicle_command.__get__(self)
        self.gazebo_world_pose_callback = gazebo_world_pose_callback.__get__(self)
        self.local_position_callback = local_position_callback.__get__(self)

        # Initialize aircraft and camera state values
        # ---------------------------------------------------------------------------------------
        self.vehicle_status = VehicleStatus() 
        self.drone_x = 0.0
        self.drone_y = 0.0
        self.drone_z = 0.0
        
        # Initialize the actual tilt angle state value (in radians)
        # ---------------------------------------------------------------------------------------
        self.drone_roll = 0.0
        self.drone_pitch = 0.0

        # Image processing buffer converter and closed-loop control
        # ---------------------------------------------------------------------------------------
        self.aruco_detected = False
        self.aruco_offset_x = 0.0
        self.aruco_offset_y = 0.0

        self.raw_offset_x = 0.0
        self.raw_offset_y = 0.0

        self.aruco_offset_x_corrected = 0.0
        self.aruco_offset_y_corrected = 0.0

        # Configurate parameters to estimate pose aruco
        self.aruco_marker_length = 0.5 

        # Simulated Camera Matrix
        self.camera_matrix = np.array([[500.0,   0.0, 320.0],
                                       [  0.0, 500.0, 240.0],
                                       [  0.0,   0.0,   1.0]], dtype=np.float32)
        
        self.dist_coeffs = np.zeros((5, 1), dtype=np.float32)

        # The ArUco recognition configuration
        # ---------------------------------------------------------------------------------------
        try:
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
            self.aruco_params = cv2.aruco.DetectorParameters()
            self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
            self.opencv_version_new = True
        except AttributeError:
            self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
            self.aruco_params = cv2.aruco.DetectorParameters_create()
            self.opencv_version_new = False

        # Separate Callback Group for ROS 2 Flight Control Flow
        # ---------------------------------------------------------------------------------------
        self.control_group = MutuallyExclusiveCallbackGroup()

        # QoS Profile configuration
        # ---------------------------------------------------------------------------------------
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Publishers
        # ---------------------------------------------------------------------------------------
        self.offboard_control_mode_pub = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)
        self.trajectory_setpoint_pub = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_profile)
        self.vehicle_command_pub = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', qos_profile)

        # Subscribers
        # ---------------------------------------------------------------------------------------
        self.vehicle_status_sub = self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status', self.status_callback, qos_profile, callback_group=self.control_group
        )
        
        self.vehicle_attitude_sub = self.create_subscription(
            VehicleAttitude, '/fmu/out/vehicle_attitude', self.public_attitude_callback, qos_profile, callback_group=self.control_group
        )

        self.vehicle_local_position_sub = self.create_subscription(
            VehicleLocalPosition, '/fmu/out/vehicle_local_position', self.local_position_callback, qos_profile, callback_group=self.control_group
        )
        
        # Connect Camera via GZ-TRANSPORT
        # ---------------------------------------------------------------------------------------
        self.gz_node = GzNode()
        self.gz_topic = '/world/mworlds/model/x500_mono_cam_down_0/link/camera_link/sensor/camera/image'
        self.gz_node.subscribe(GzImage, self.gz_topic, self.gazebo_camera_callback)

        # Register for ground truth pose from gazebo world
        # ---------------------------------------------------------------------------------------
        self.gz_pose_topic = '/world/mworlds/pose/info' 
        self.gz_node.subscribe(GzPoseV, self.gz_pose_topic, self.gazebo_world_pose_callback)

        # Main control loop (10Hz)
        # ---------------------------------------------------------------------------------------
        self.dt = 0.1  
        self.timer = self.create_timer(self.dt, self.timer_callback, callback_group=self.control_group)
        
        # FSM status for precise landing control
        # ---------------------------------------------------------------------------------------
        self.offboard_setpoint_counter = 0
        self.flight_state = "TAKEOFF"
        self.state_timer = 0.0

        # Dynamic Target Position (NED system)
        # ---------------------------------------------------------------------------------------
        self.target_x = 0.0
        self.target_y = 0.0
        self.target_z = -5.0

        # Absolute Ground Truth Position from Gazebo World (ENU)
        # ---------------------------------------------------------------------------------------
        self.world_gt_x = 0.0 
        self.world_gt_y = 0.0  
        self.world_gt_z = 0.0  

        # ============================================================
        # WIND COMPENSATION AND ADAPTIVE CONTROL VARIABLES
        # ============================================================
        self.error_history = []  # Lưu lịch sử sai số
        self.error_velocity = 0.0  # Vận tốc thay đổi sai số
        self.prev_error = 0.0
        self.accumulated_correction = 0.0
        self.wind_resistance_factor = 1.0
        self.consecutive_error_increase = 0
        self.oscillation_detected = False
        self.oscillation_counter = 0
        self.prev_steps = []  # Lưu các bước điều chỉnh trước đó
        self.wind_strength = 0.0
        self.wind_direction_x = 0.0
        self.wind_direction_y = 0.0
        
        # Ngưỡng phát hiện
        self.WIND_THRESHOLD = 0.03
        self.OSCILLATION_THRESHOLD = 3

        # Initialize OpenCV Display Window
        self.get_logger().info("Precision Landing: ROS2 + Gz-Transport ArUco Tracking READY!")
        self.get_logger().info("Wind Compensation and Adaptive Control ENABLED!")

    def gazebo_camera_callback(self, msg):
        """
        Camera data processing function: Receives images from Gazebo Core, 
        identifies ArUco, and calculates the deviation for closed-loop control.
        """
        try:
            # Decode image
            image_data = np.frombuffer(msg.data, dtype=np.uint8)
            
            if msg.pixel_format_type == 3:  # RGB_INT8
                cv_image = image_data.reshape((msg.height, msg.width, 3))
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
            elif msg.pixel_format_type == 1:  # L_INT8 (Mono)
                cv_image = image_data.reshape((msg.height, msg.width))
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)
            else:
                channels = len(msg.data) // (msg.height * msg.width)
                cv_image = image_data.reshape((msg.height, msg.width, channels))
                if channels == 3:
                    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
            
            height, width, _ = cv_image.shape
            cam_center_x = width // 2
            cam_center_y = height // 2

            # ArUco Marker Recognition
            if self.opencv_version_new:
                corners, ids, rejected = self.aruco_detector.detectMarkers(cv_image)
            else:
                corners, ids, rejected = cv2.aruco.detectMarkers(cv_image, self.aruco_dict, parameters=self.aruco_params)
            
            if ids is not None:
                self.aruco_detected = True
                cv2.aruco.drawDetectedMarkers(cv_image, corners, ids)
                
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners, self.aruco_marker_length, self.camera_matrix, self.dist_coeffs
                )
                
                rvec = rvecs[0]
                tvec = tvecs[0]

                cv2.drawFrameAxes(cv_image, self.camera_matrix, self.dist_coeffs, rvec, tvec, 0.3)

                pose_x = tvec[0][0]
                pose_y = tvec[0][1]
                pose_z = tvec[0][2]

                rot_x = math.degrees(rvec[0][0])
                rot_y = math.degrees(rvec[0][1])
                rot_z = math.degrees(rvec[0][2])

                pose_text = f"Posecam -> Tag: X:{pose_x:.2f}m Y:{pose_y:.2f}m Z:{pose_z:.2f}m"
                cv2.putText(cv_image, pose_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                rvec_text = f"Rvec (Deg) -> RotX:{rot_x:.1f} RotY:{rot_y:.1f} RotZ(Yaw):{rot_z:.1f}"
                cv2.putText(cv_image, rvec_text, (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

                c = corners[0][0]
                aruco_center_x = int((c[0][0] + c[1][0] + c[2][0] + c[3][0]) / 4)
                aruco_center_y = int((c[0][1] + c[1][1] + c[2][1] + c[3][1]) / 4)
                
                cv2.circle(cv_image, (aruco_center_x, aruco_center_y), 5, (0, 255, 0), -1)
                
                pixel_error_x = aruco_center_x - cam_center_x
                pixel_error_y = cam_center_y - aruco_center_y

                roll_rad = self.drone_roll    
                pitch_rad = self.drone_pitch  
                current_altitude = abs(self.drone_z) if abs(self.drone_z) > 0.5 else abs(self.target_z)
                fov_rad = math.radians(60.0) 
                meters_per_pixel = (2.0 * current_altitude * math.tan(fov_rad / 2.0)) / width

                self.raw_offset_x = pixel_error_x * meters_per_pixel
                self.raw_offset_y = pixel_error_y * meters_per_pixel

                bias_y = current_altitude * math.tan(pitch_rad)
                bias_x = current_altitude * math.tan(roll_rad)

                self.aruco_offset_x_corrected = self.raw_offset_x - bias_x
                self.aruco_offset_y_corrected = self.raw_offset_y - bias_y
                
                cv2.line(cv_image, (cam_center_x, cam_center_y), (aruco_center_x, aruco_center_y), (255, 0, 0), 2)
            else:
                self.aruco_detected = False

            cv2.circle(cv_image, (cam_center_x, cam_center_y), 6, (0, 0, 255), 2)

            # Display wind compensation status
            wind_text = f"Wind: {self.wind_strength:.2f} | Osc: {self.oscillation_detected}"
            cv2.putText(cv_image, wind_text, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            status_text = f"State: {self.flight_state} | Detected: {self.aruco_detected}"
            cv2.putText(cv_image, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            self.latest_preview_image = cv_image
            
        except Exception as e:
            self.get_logger().error(f"Image processing and ArUco tracking errors: {e}")

    def status_callback(self, msg):
        self.vehicle_status = msg

    def fuzzy_membership_altitude(self, h):
        """
        Calculate membership values for altitude (7 fuzzy sets)
        Altitude ranges: 0-10m with thresholds at 0.5, 1.5, 3.0, 5.0, 7.0, 8.5, 10.0
        """
        # Extreme Low (EL): 0-0.5m, peak at 0m
        if h <= 0.5:
            mu_el = 1.0 - (h / 0.5)
        else:
            mu_el = 0.0
        
        # Very Low (VL): 0.5-1.5m, peak at 1.0m
        if 0.5 <= h <= 1.5:
            mu_vl = 1.0 - abs(h - 1.0) / 0.5
        else:
            mu_vl = 0.0
        
        # Low (L): 1.0-3.0m, peak at 2.0m
        if 1.0 <= h <= 3.0:
            mu_l = 1.0 - abs(h - 2.0) / 1.0
        else:
            mu_l = 0.0
        
        # Medium (M): 2.0-5.0m, peak at 3.5m
        if 2.0 <= h <= 5.0:
            mu_m = 1.0 - abs(h - 3.5) / 1.5
        else:
            mu_m = 0.0
        
        # High (H): 4.0-7.0m, peak at 5.5m
        if 4.0 <= h <= 7.0:
            mu_h = 1.0 - abs(h - 5.5) / 1.5
        else:
            mu_h = 0.0
        
        # Very High (VH): 6.0-8.5m, peak at 7.25m
        if 6.0 <= h <= 8.5:
            mu_vh = 1.0 - abs(h - 7.25) / 1.25
        else:
            mu_vh = 0.0
        
        # Extreme High (EH): 8.0-10m, peak at 10m
        if h >= 8.0:
            mu_eh = min(1.0, (h - 8.0) / 2.0)
        else:
            mu_eh = 0.0
        
        return mu_el, mu_vl, mu_l, mu_m, mu_h, mu_vh, mu_eh

    def fuzzy_membership_error(self, e):
        """
        Calculate membership values for horizontal error (7 fuzzy sets)
        Error ranges: 0-1.2m with thresholds at 0.02, 0.05, 0.10, 0.20, 0.40, 0.70, 1.00
        """
        # Zero (ZE): 0-0.02m, peak at 0m
        if e <= 0.02:
            mu_ze = 1.0 - (e / 0.02)
        else:
            mu_ze = 0.0
        
        # Very Small (VS): 0.02-0.05m, peak at 0.035m
        if 0.02 <= e <= 0.05:
            mu_vs = 1.0 - abs(e - 0.035) / 0.015
        else:
            mu_vs = 0.0
        
        # Small (SM): 0.035-0.10m, peak at 0.065m
        if 0.035 <= e <= 0.10:
            mu_sm = 1.0 - abs(e - 0.065) / 0.035
        else:
            mu_sm = 0.0
        
        # Normal (NO): 0.065-0.20m, peak at 0.13m
        if 0.065 <= e <= 0.20:
            mu_no = 1.0 - abs(e - 0.13) / 0.07
        else:
            mu_no = 0.0
        
        # Big (BI): 0.13-0.40m, peak at 0.25m
        if 0.13 <= e <= 0.40:
            mu_bi = 1.0 - abs(e - 0.25) / 0.15
        else:
            mu_bi = 0.0
        
        # Very Big (VB): 0.25-0.70m, peak at 0.45m
        if 0.25 <= e <= 0.70:
            mu_vb = 1.0 - abs(e - 0.45) / 0.25
        else:
            mu_vb = 0.0
        
        # Extreme Big (EB): 0.60-1.00m, peak at 1.00m
        if e >= 0.60:
            mu_eb = min(1.0, (e - 0.60) / 0.60)
        else:
            mu_eb = 0.0
        
        return mu_ze, mu_vs, mu_sm, mu_no, mu_bi, mu_vb, mu_eb

    def fuzzy_controller_7x7(self, h, e):
        """
        7x7 Fuzzy Controller with 49 rules
        Input: altitude (h), horizontal error (e)
        Output: gain value (0.001 to 0.10)
        """
        # Get membership values
        alt_mu = self.fuzzy_membership_altitude(h)
        err_mu = self.fuzzy_membership_error(e)
        
        # Define output gains (7x7 matrix)
        # Rows: Altitude (EL, VL, L, M, H, VH, EH)
        # Columns: Error (ZE, VS, SM, NO, BI, VB, EB)
        gains = [
            # ZE    VS    SM    NO    BI    VB    EB
            [0.005, 0.005, 0.006, 0.010, 0.015, 0.010, 0.005],  # EL (Extreme Low)
            [0.005, 0.010, 0.020, 0.030, 0.035, 0.030, 0.020],  # VL (Very Low)
            [0.010, 0.020, 0.030, 0.050, 0.065, 0.050, 0.030],  # L (Low)
            [0.020, 0.030, 0.050, 0.080, 0.085, 0.080, 0.050],  # M (Medium)
            [0.030, 0.050, 0.080, 0.090, 0.090, 0.090, 0.080],  # H (High)
            [0.050, 0.080, 0.100, 0.100, 0.100, 0.070, 0.050],  # VH (Very High)
            [0.050, 0.080, 0.100, 0.100, 0.100, 0.070, 0.050]   # EH (Extreme High)
        ]
        
        # Apply fuzzy rules (Mamdani inference with min operator)
        weights = []
        for i in range(7):
            for j in range(7):
                weight = min(alt_mu[i], err_mu[j])
                weights.append(weight)
        
        # Defuzzification (Center of Gravity)
        sum_weight = sum(weights)
        if sum_weight > 0:
            numerator = 0
            idx = 0
            for i in range(7):
                for j in range(7):
                    numerator += weights[idx] * gains[i][j]
                    idx += 1
            gain = numerator / sum_weight
        else:
            gain = 0.02  # Default safe value
        
        # Additional logic for stability
        # If error is very large, keep gain moderate to avoid overshoot
        if e > 0.6 and h < 3.0:
            gain = min(gain, 0.02)
        
        return gain

    def detect_wind_and_oscillation(self, e, step_x, step_y):
        """
        Detect wind strength and oscillation based on error and step changes
        """
        # Update error history
        self.error_history.append(e)
        if len(self.error_history) > 10:
            self.error_history.pop(0)
        
        # Calculate error velocity
        if len(self.error_history) > 1:
            self.error_velocity = (self.error_history[-1] - self.error_history[-2]) / self.dt
        
        # Detect wind (continuous error increase in one direction)
        if len(self.error_history) > 5:
            recent_errors = self.error_history[-5:]
            # Check if error is consistently increasing
            if all(recent_errors[i] < recent_errors[i+1] for i in range(4)):
                self.consecutive_error_increase += 1
            else:
                self.consecutive_error_increase = max(0, self.consecutive_error_increase - 1)
            
            # Estimate wind strength based on error velocity
            if abs(self.error_velocity) > 0.01:
                self.wind_strength = min(1.0, abs(self.error_velocity) * 2.0)
            else:
                self.wind_strength = max(0, self.wind_strength - 0.01)
        
        # Detect oscillation (step direction changes)
        self.prev_steps.append((step_x, step_y))
        if len(self.prev_steps) > 5:
            self.prev_steps.pop(0)
        
        if len(self.prev_steps) > 3:
            # Check if steps are oscillating
            direction_changes = 0
            for i in range(1, len(self.prev_steps)):
                prev_sign_x = np.sign(self.prev_steps[i-1][0])
                curr_sign_x = np.sign(self.prev_steps[i][0])
                prev_sign_y = np.sign(self.prev_steps[i-1][1])
                curr_sign_y = np.sign(self.prev_steps[i][1])
                
                if prev_sign_x != curr_sign_x and prev_sign_x != 0 and curr_sign_x != 0:
                    direction_changes += 1
                if prev_sign_y != curr_sign_y and prev_sign_y != 0 and curr_sign_y != 0:
                    direction_changes += 1
            
            if direction_changes >= 3:
                self.oscillation_counter += 1
            else:
                self.oscillation_counter = max(0, self.oscillation_counter - 1)
            
            self.oscillation_detected = self.oscillation_counter > self.OSCILLATION_THRESHOLD

    def timer_callback(self):
        """
        Main loop function for flight control (10Hz)
        """
        if self.offboard_setpoint_counter < 10:
            self.publish_offboard_control_mode()
            self.publish_trajectory_setpoint(0.0, 0.0, 0.0)
            self.offboard_setpoint_counter += 1
            return

        if self.offboard_setpoint_counter == 10:
            self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
            self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
            self.offboard_setpoint_counter += 1

        self.publish_offboard_control_mode()
        self.state_timer += self.dt

        # Flight State Machine
        if self.flight_state == "TAKEOFF":
            self.target_x = 0.0
            self.target_y = 0.0
            self.target_z = -25.0

            if self.state_timer > 10 or abs(self.drone_z + 9.0) < 1.0: 
                self.flight_state = "WP1"
                self.state_timer = 0.0
                self.get_logger().info(f"[TAKEOFF MODE] Reached altitude {self.state_timer:.2f}m")     
        
        elif self.flight_state == "WP1":
            self.target_x = 107.32
            self.target_y = -135.42
            self.target_z = -25.0
            
            if self.state_timer > 30.0:  
                self.flight_state = "DESTINATION_REACHED"
                self.state_timer = 0.0
                self.get_logger().info(f"[OFFBOARD MODE] Drone has reached target location WP1.")
        
        elif self.flight_state == "DESTINATION_REACHED":
            self.target_z = -9.0

            if self.state_timer > 9.0 or abs(self.world_gt_z - 9.0) < 0.5: 
                self.flight_state = "SEARCH_TRACK"
                self.state_timer = 0.0
                self.get_logger().info("[OFFBOARD MODE] Switching to landing target search mode.")
        
        elif self.flight_state == "SEARCH_TRACK":
            if self.aruco_detected:
                
                # ==================================================================
                # 1. ADVANCED 7x7 FUZZY CONTROLLER
                # ==================================================================
                # Take the actual elevation h (NED z negative) and the actual horizontal error e (meters).
                h = abs(self.drone_z) if abs(self.drone_z) > 0.3 else abs(self.target_z)
                e = math.sqrt(self.aruco_offset_x_corrected**2 + self.aruco_offset_y_corrected**2)

                # Apply 7x7 Fuzzy Controller
                p_gain = self.fuzzy_controller_7x7(h, e)

                # ==================================================================
                # CALCULATE JUMP STEPS AND UPDATE TRAJECTORY SETPOINT OF PX4
                # ==================================================================
                step_x = self.aruco_offset_y_corrected * p_gain
                step_y = self.aruco_offset_x_corrected * p_gain

                # Detect wind and oscillation before applying adaptations
                self.detect_wind_and_oscillation(e, step_x, step_y)

                # ==================================================================
                # WIND COMPENSATION
                # ==================================================================
                if self.wind_strength > self.WIND_THRESHOLD:
                    # Add compensation in opposite direction of wind
                    wind_compensation = min(0.05, self.wind_strength * 0.1)
                    if self.error_velocity > 0:
                        step_x += np.sign(self.aruco_offset_x_corrected) * wind_compensation
                        step_y += np.sign(self.aruco_offset_y_corrected) * wind_compensation
                    
                    self.get_logger().info(
                        f"🌬️ Wind compensation: {wind_compensation:.3f} m/s", 
                        throttle_duration_sec=0.5
                    )

                # ==================================================================
                # OSCILLATION DAMPENING
                # ==================================================================
                if self.oscillation_detected:
                    # Reduce gain to dampen oscillations
                    p_gain = p_gain * 0.7
                    self.get_logger().warning(
                        f"🔄 Oscillation detected! Reducing gain to {p_gain:.4f}", 
                        throttle_duration_sec=0.5
                    )

                # ==================================================================
                # ADAPTIVE MAX STEP BASED ON CONDITIONS
                # ==================================================================
                # Base max step based on altitude
                base_max_step = 0.05
                if h > 8.0:
                    base_max_step = 0.15
                elif h > 5.0:
                    base_max_step = 0.10
                elif h > 3.0:
                    base_max_step = 0.06
                elif h > 1.5:
                    base_max_step = 0.04
                else:
                    base_max_step = 0.03
                
                # Adjust for wind
                if self.wind_strength > 0.05:
                    base_max_step = base_max_step * 1.2
                
                # Adjust for oscillation
                if self.oscillation_detected:
                    base_max_step = base_max_step * 0.5
                
                # Apply hardware saturation
                max_safe_step = min(0.25, base_max_step)
                
                step_x = max(-max_safe_step, min(max_safe_step, step_x))
                step_y = max(-max_safe_step, min(max_safe_step, step_y))

                # Perform cumulative setpoint addition.
                self.target_x = self.target_x + step_x
                self.target_y = self.target_y + step_y

                # ==================================================================
                # LOGGING AND STATUS
                # ==================================================================
                self.get_logger().info(
                    f"[Fuzzy 7x7] H:{h:.2f}m | Err:{e:.3f}m | Gain:{p_gain:.4f} | "
                    f"Wind:{self.wind_strength:.2f} | Osc:{self.oscillation_detected} | "
                    f"Step:{max_safe_step:.3f}m", 
                    throttle_duration_sec=0.5
                )

                # ==================================================================
                # SAFE LANDING CONDITIONS
                # ==================================================================
                # Land when: error small, altitude low, no oscillation
                if e < 0.03 and h < 0.3 and not self.oscillation_detected:
                    self.flight_state = "PRECISION_LAND"
                    self.get_logger().info("✅ Landing condition met!")
                
                # If strong wind at low altitude, hold position
                elif self.consecutive_error_increase > 8 and h < 1.0:
                    self.get_logger().warning("🌬️ Strong wind detected at low altitude! Holding position...")
                    # Maintain altitude, only adjust position
                    self.target_z = -1.0

            else:
                # ArUco lost: hover and search
                self.target_z = -8.0
                self.target_x += 0.01  # Slight rotation to find marker
                self.get_logger().warning("🔍 ArUco lost! Searching...")

        elif self.flight_state == "PRECISION_LAND":
            # Save flight log
            import csv
            with open(self.log_file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.log_headers)
                writer.writerows(self.log_data)
            self.get_logger().info(f"Log saved to {self.log_file_path}")

            # Execute landing
            self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
            cv2.destroyAllWindows()
            self.timer.cancel()
            self.get_logger().info("Precision Landing completed successfully!")
            os._exit(0)

        # Publish trajectory setpoint
        self.publish_trajectory_setpoint(self.target_x, self.target_y, self.target_z)

def main(args=None):
    rclpy.init(args=args)
    node = OffboardPrecisionLanding()
    
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)

    cv2.namedWindow("Drone FPV Camera View (Gz-Transport)", cv2.WINDOW_AUTOSIZE)
    
    try:
        while rclpy.ok():
            executor.spin_once(timeout_sec=0.01)
            
            if hasattr(node, 'latest_preview_image') and node.latest_preview_image is not None:
                cv2.imshow("Drone FPV Camera View (Gz-Transport)", node.latest_preview_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        pass

    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()