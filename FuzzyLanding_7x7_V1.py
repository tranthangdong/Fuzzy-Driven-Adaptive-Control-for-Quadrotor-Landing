# -----------------------------------------------------------------------------------------------
#  FuzzyLanding_7x7.py
#  Description: Enhanced precision landing algorithm using 7x7 fuzzy controller 
#               (2 inputs: Altitude and Lateral Error, 1 output: Gain to adjust speed)
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
                        'p_gain', 'flight_state']

        # Assign Functions to Methods of a Class
        # ---------------------------------------------------------------------------------------
        self.public_attitude_callback = public_attitude_callback.__get__(self)
        self.publish_offboard_control_mode = publish_offboard_control_mode.__get__(self)
        self.publish_trajectory_setpoint = publish_trajectory_setpoint.__get__(self)
        self.publish_vehicle_command = publish_vehicle_command.__get__(self)
        self.gazebo_world_pose_callback = gazebo_world_pose_callback.__get__(self)
        self.local_position_callback = local_position_callback.__get__(self)

        # Initialize aircraft and camera state values - Khởi tạo giá trị trạng thái máy bay và camera
        # ---------------------------------------------------------------------------------------
        self.vehicle_status = VehicleStatus() 
        self.drone_x = 0.0
        self.drone_y = 0.0
        self.drone_z = 0.0
        
        # Initialize the actual tilt angle state value (in radians) - Khởi tạo giá trị trạng thái góc nghiên thực tế 
        # ---------------------------------------------------------------------------------------
        self.drone_roll = 0.0
        self.drone_pitch = 0.0

        # Image processing buffer converter and closed-loop control - Bộ biến đệm xử lý ảnh và điều khiển vòng kín
        # ---------------------------------------------------------------------------------------
        self.aruco_detected = False
        self.aruco_offset_x = 0.0  # Mét (Sai lệch hướng East của drone)
        self.aruco_offset_y = 0.0  # Mét (Sai lệch hướng North của drone)

        self.raw_offset_x = 0.0
        self.raw_offset_y = 0.0

        self.aruco_offset_x_corrected = 0.0
        self.aruco_offset_y_corrected = 0.0

        # Configurate parameters to estimate pose aruco
        # Actual dimensions of the ArUco code (Example: 0.5 meters = 50cm)
        self.aruco_marker_length = 0.5 

        # Simulated Camera Matrix (Replace with the actual gazebo camera matrix if available for absolute accuracy)
        # Assume the camera has a resolution of approximately 640x480 or equivalent with a FOV of 60 degrees.
        self.camera_matrix = np.array([[500.0,   0.0, 320.0],
                                       [  0.0, 500.0, 240.0],
                                       [  0.0,   0.0,   1.0]], dtype=np.float32)
        
        # Distortion coefficient (default is 0 for an ideal camera in Gazebo)
        self.dist_coeffs = np.zeros((5, 1), dtype=np.float32)
        # ------------------------------------------------

        # The ArUco recognition configuration is compatible with OpenCV versions 4.7
        # ---------------------------------------------------------------------------------------
        try:
            # OpenCV 4.7+
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
            self.aruco_params = cv2.aruco.DetectorParameters()
            self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
            self.opencv_version_new = True
        except AttributeError:
            # If it fails, the configuration will be automatically downgraded to be compatible with OpenCV 4.6 syntax.
            self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
            self.aruco_params = cv2.aruco.DetectorParameters_create()
            self.opencv_version_new = False

        # Separate Callback Group for ROS 2 Flight Control Flow - Tách biệt Nhóm Callback cho luồng điều khiển bay ROS 2
        # ---------------------------------------------------------------------------------------
        self.control_group = MutuallyExclusiveCallbackGroup()

        # 1. The QoS Profile configuration is fully compatible with PX4 - Cấu hình QoS Profile tương thích hoàn toàn với PX4
        # ---------------------------------------------------------------------------------------
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        # 2. Publishers điều khiển bay (ROS 2)
        # ---------------------------------------------------------------------------------------
        self.offboard_control_mode_pub = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)
        self.trajectory_setpoint_pub = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_profile)
        self.vehicle_command_pub = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', qos_profile)

        # 3. Subscribers track aircraft status (ROS 2) - Subscribers theo dõi trạng thái máy bay (ROS 2)
        # ---------------------------------------------------------------------------------------
        self.vehicle_status_sub = self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status', self.status_callback, qos_profile, callback_group=self.control_group
        )
        
        # 4. Subscribers track aircraft attitude (ROS 2) - Subscribers theo dõi góc nghiên thực tế
        # ---------------------------------------------------------------------------------------
        self.vehicle_attitude_sub = self.create_subscription(
            VehicleAttitude, '/fmu/out/vehicle_attitude', self.public_attitude_callback, qos_profile, callback_group=self.control_group
        )

        # 5. Subscribers track aircraft local position (ROS 2) - Subscribers theo dõi vị trí cục bộ từ EKF2 PX4
        # ---------------------------------------------------------------------------------------
        self.vehicle_local_position_sub = self.create_subscription(
            VehicleLocalPosition, '/fmu/out/vehicle_local_position', self.local_position_callback, qos_profile, callback_group=self.control_group
        )
        
        # 6. Connect Camera via GZ-TRANSPORT
        # ---------------------------------------------------------------------------------------
        self.gz_node = GzNode()
        self.gz_topic = '/world/mworlds/model/x500_mono_cam_down_0/link/camera_link/sensor/camera/image'
        
        self.gz_node.subscribe(GzImage, self.gz_topic, self.gazebo_camera_callback)

        # 7. Register for a prime ground truth spot from gazebo world
        # ---------------------------------------------------------------------------------------
        self.gz_pose_topic = '/world/mworlds/pose/info' 
        self.gz_node.subscribe(GzPoseV, self.gz_pose_topic, self.gazebo_world_pose_callback) # 

        # 8. Main control loop (10Hz) - Vòng lặp điều khiển chính (10Hz)
        self.dt = 0.1  
        self.timer = self.create_timer(self.dt, self.timer_callback, callback_group=self.control_group)
        
        # 9. FSM status for precise landing control - Trạng thái của FSM điều khiển tự động hạ cánh chính xác
        self.offboard_setpoint_counter = 0
        self.flight_state = "TAKEOFF"
        self.state_timer = 0.0

        # 10. Dynamic Target Position (According to the NED system of PX4) - Vị trí Target động (Theo hệ NED của PX4)
        self.target_x = 0.0
        self.target_y = 0.0
        self.target_z = -5.0

        # 11. Absolute Ground Truth Position from Gazebo World (ENU System: East-North-Up)
        self.world_gt_x = 0.0 
        self.world_gt_y = 0.0  
        self.world_gt_z = 0.0  

        # 12. Initialize OpenCV Display Window
        # cv2.namedWindow("Drone FPV Camera View (Gz-Transport)", cv2.WINDOW_AUTOSIZE)
        # cv2.startWindowThread()
        self.get_logger().info("Precision Landing: ROS2 + Gz-Transport ArUco Tracking READY!")

    def gazebo_camera_callback(self, msg):
        # Camera data processing function: Receives images from Gazebo Core, identifies ArUco, and calculates the deviation to control the closed-loop landing accurately.
        # ---------------------------------------------------------------------------------------
        # Input:
        # - msg.data: Binary byte array of raw image from the camera
        # - mgs.width, msg.height: Image dimensions
        # - msg.pixel_format_type: Pixel format type (3 = RGB_INT8)

        # Output:
        # - self.aruco_detected: ArUco detection flag
        # - self.aruco_offset_x, self.aruco_offset_y: Deviation from image center to ArUco center (meters)
        # - self.aruco_offset_x_corrected, self.aruco_offset_y_corrected: The deviation has been compensated for the body angle (meters).
        # - Display image with ArUco marker bounding box and status information on OpenCV interface

        try:
            # 1. Decode a binary byte array into an image matrix - Giải mã mảng byte nhị phân thành ma trận ảnh
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

            # 2. ArUco Marker Recognition Algorithm - Thuật toán nhận diện ArUco Marker
            if self.opencv_version_new:
                corners, ids, rejected = self.aruco_detector.detectMarkers(cv_image)
            else:
                corners, ids, rejected = cv2.aruco.detectMarkers(cv_image, self.aruco_dict, parameters=self.aruco_params)
            
            if ids is not None:
                self.aruco_detected = True
                cv2.aruco.drawDetectedMarkers(cv_image, corners, ids)
                
                # Use a single Pose estimation function for each marker
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners, self.aruco_marker_length, self.camera_matrix, self.dist_coeffs
                )
                
                # Retrieve the rvec (rotation vector) and tvec (translation vector) of the first marker found.
                rvec = rvecs[0]
                tvec = tvecs[0]

                # Draw a 3D coordinate system (X: Red, Y: Green, Z: Blue) on the marker (Axis length = 0.3 meters)
                cv2.drawFrameAxes(cv_image, self.camera_matrix, self.dist_coeffs, rvec, tvec, 0.3)

                # 1. Extract the translation distance from Camera to ArUco (Camera coordinate system: X-right, Y-down, Z-forward)
                pose_x = tvec[0][0]
                pose_y = tvec[0][1]
                pose_z = tvec[0][2] # This is the straight-line distance from camera to tag

                # 2. Extract and convert the rotation angles from the rvec vector to degrees (Degrees)
                # rvec[0][0]: Góc xoay quanh trục X (Roll của tag so với cam)
                # rvec[0][1]: Góc xoay quanh trục Y (Pitch của tag so với cam)
                # rvec[0][2]: Góc xoay quanh trục Z (Yaw của tag so với cam - góc bạn cần dùng để xoay drone)
                rot_x = math.degrees(rvec[0][0])
                rot_y = math.degrees(rvec[0][1])
                rot_z = math.degrees(rvec[0][2])

                # 3. Display the estimated POSE information on the screen.
                pose_text = f"Posecam -> Tag: X:{pose_x:.2f}m Y:{pose_y:.2f}m Z:{pose_z:.2f}m"
                cv2.putText(cv_image, pose_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                # ----------------------------------------------------

                # 4. Display the rvec rotation information (Degrees) on the screen at Y=85 (Cyan color)
                rvec_text = f"Rvec (Deg) -> RotX:{rot_x:.1f} RotY:{rot_y:.1f} RotZ(Yaw):{rot_z:.1f}"
                cv2.putText(cv_image, rvec_text, (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

                # 5. Calculate the geometric center of the ArUco marker (Average of 4 corners)
                c = corners[0][0]
                aruco_center_x = int((c[0][0] + c[1][0] + c[2][0] + c[3][0]) / 4)
                aruco_center_y = int((c[0][1] + c[1][1] + c[2][1] + c[3][1]) / 4)
                
                # 6. Draw the center point of the ArUco marker (Green color)
                cv2.circle(cv_image, (aruco_center_x, aruco_center_y), 5, (0, 255, 0), -1)
                
                # 7. Calculate the pixel error from the center of the image to the center of the ArUco marker
                pixel_error_x = aruco_center_x - cam_center_x
                pixel_error_y = cam_center_y - aruco_center_y

                # 8. Calculate the conversion from pixels to meters based on altitude
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
                
                # 9. Draw a line connecting the drone's center to the ArUco marker's center (Blue color)
                cv2.line(cv_image, (cam_center_x, cam_center_y), (aruco_center_x, aruco_center_y), (255, 0, 0), 2)
            else:
                self.aruco_detected = False

            # 10. Draw the center point of the drone (Red color)
            cv2.circle(cv_image, (cam_center_x, cam_center_y), 6, (0, 0, 255), 2)

            # 11. Display the processing status data on the graphical interface
            status_text = f"State: {self.flight_state} | Detected: {self.aruco_detected}"
            cv2.putText(cv_image, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            self.latest_preview_image = cv_image  # Save the image to Node's cache variable.
            
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
        
        # Small (S): 0.035-0.10m, peak at 0.065m
        if 0.035 <= e <= 0.10:
            mu_s = 1.0 - abs(e - 0.065) / 0.035
        else:
            mu_s = 0.0
        
        # Medium (M): 0.065-0.20m, peak at 0.13m
        if 0.065 <= e <= 0.20:
            mu_m = 1.0 - abs(e - 0.13) / 0.07
        else:
            mu_m = 0.0
        
        # Large (L): 0.13-0.40m, peak at 0.25m
        if 0.13 <= e <= 0.40:
            mu_l = 1.0 - abs(e - 0.25) / 0.15
        else:
            mu_l = 0.0
        
        # Very Large (VL): 0.25-0.70m, peak at 0.45m
        if 0.25 <= e <= 0.70:
            mu_vl = 1.0 - abs(e - 0.45) / 0.25
        else:
            mu_vl = 0.0
        
        # Extreme Large (EL): 0.60-1.00m, peak at 1.00m
        if e >= 0.60:
            mu_el = min(1.0, (e - 0.60) / 0.60)
        else:
            mu_el = 0.0
        
        return mu_ze, mu_vs, mu_s, mu_m, mu_l, mu_vl, mu_el

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
        # Columns: Error (ZE, VS, S, M, L, VL, EL)
        gains = [
            # ZE    VS    S     M     L     VL    EL
            [0.005, 0.005, 0.006, 0.010, 0.015, 0.010, 0.005],  # EL (Extreme Low)
            [0.005, 0.006, 0.010, 0.020, 0.020, 0.020, 0.010],  # VL (Very Low)
            [0.005, 0.010, 0.020, 0.030, 0.035, 0.030, 0.020],  # L (Low)
            [0.010, 0.020, 0.030, 0.050, 0.065, 0.050, 0.030],  # M (Medium)
            [0.020, 0.030, 0.050, 0.080, 0.085, 0.080, 0.050],  # H (High)
            [0.030, 0.050, 0.080, 0.090, 0.090, 0.090, 0.080],  # VH (Very High)
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
        # 1. If error is very small and altitude is low, reduce gain to prevent oscillation
        #if e < 0.02 and h < 1.0:
        #    gain = min(gain, 0.01)
        
        # 2. If error is very large, keep gain moderate to avoid overshoot
        if e > 0.6 and h < 3.0:
            gain = min(gain, 0.02)
        
        # 3. Apply saturation to prevent extreme values
        #gain = max(0.001, min(0.12, gain))
        
        return gain

    def timer_callback(self):
        # Main loop function for flight control (10Hz): Finite state machine handles the precise landing problem.
        # ---------------------------------------------------------------------------------------
        # Input:
        # - self.vehicle_status: Flight status updated from PX4
        # - self.aruco_detected, self.aruco_offset_x_corrected,
        #   self.aruco_offset_y_corrected: ArUco detection results and corrected offset values
        # Output: 
        # - self.target_x, self.target_y, self.target_z: Target coordinates calculated for precise landing control
        # - Stage 1: Takeoff and move to a position near the ArUco marker (Waypoint)
        # - Stage 2: Search and track the ArUco marker
        # - Stage 3: When close to the ArUco marker, issue a direct landing command to the FCU
        # - Stage 4: Display status information and errors on the OpenCV graphical interface
        # Note: This loop will continuously update the target coordinates based on the corrected errors to control the drone's movement towards the ArUco marker in a smooth and stable manner.

        if self.offboard_setpoint_counter < 10:
            self.publish_offboard_control_mode()
            self.publish_trajectory_setpoint(0.0, 0.0, 0.0)
            self.offboard_setpoint_counter += 1
            return

        if self.offboard_setpoint_counter == 10:
            self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)  # Aircraft Arm
            self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)      # Switch toOffboard Mode
            self.offboard_setpoint_counter += 1

        self.publish_offboard_control_mode()
        self.state_timer += self.dt

        # Computer visual hybrid state machine (fsm) control
        if self.flight_state == "TAKEOFF":
            # Take off vertically to a height of 5 meters (X=0, Y=0, Z=-5)
            self.target_x = 0.0
            self.target_y = 0.0
            self.target_z = -25

            # Give the drone 7 seconds to reach altitude or check if it's close to the target altitude.
            if self.state_timer > 10 or abs(self.drone_z + 9.0) < 1.0: 
                self.flight_state = "WP1"
                self.state_timer = 0.0
                self.get_logger().info(f"[TAKEOFF MODE] Reached altitude {self.state_timer:.2f}m")     
        
        elif self.flight_state == "WP1":
            # Move to the coordinates of point WP1 (Z=-8)
            self.target_x = 107.32        #108.32
            self.target_y = -135.42         #-136.42
            self.target_z = -25.0
            
            # Give the drone 15 seconds to complete the travel distance
            if self.state_timer > 30.0:  
                self.flight_state = "DESTINATION_REACHED"
                self.state_timer = 0.0
                self.get_logger().info(f"[OFFBOARD MODE] {self.target_z:.2f} : Drone has reached target location WP1. Begin descent.")
        
        elif self.flight_state == "DESTINATION_REACHED":
            self.target_z = -9.0

            # Give the drone 5 seconds to stabilize altitude and start searching for ArUco
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

                # Adaptive saturation based on altitude and error
                # Higher altitude allows larger steps, lower altitude requires smaller steps for precision
                # Saturation safety jump limiter to prevent mechanical body jerk (20cm)
                max_step = 0.2
                if h > 4.0:
                    max_step = 0.05
                else:
                    max_step = 0.02
                    
                #if h > 8.0:
                #    max_step = 0.15  # High altitude: can move faster
                #elif h > 5.0:
                #    max_step = 0.10  # Medium-high altitude
                #elif h > 3.0:
                #    max_step = 0.06  # Medium altitude
                #elif h > 1.5:
                #    max_step = 0.04  # Low altitude: slow down
                #else:
                #    max_step = 0.025  # Very low altitude: very precise movements
                
                # If error is very large, allow slightly larger steps but still controlled
                # if e > 0.5 and h > 5.0:
                #    max_step = min(max_step * 1.5, 0.20)
                
                # If error is very small, limit max step to avoid oscillation
                # if e < 0.05:
                #    max_step = min(max_step, 0.01)
                
                #step_x = max(-max_step, min(max_step, step_x))
                #step_y = max(-max_step, min(max_step, step_y))

                # Perform cumulative setpoint addition.
                self.target_x = self.target_x + step_x
                self.target_y = self.target_y + step_y
                
                # Print parameters to retrieve data for graphing in Paper.
                self.get_logger().info(
                    f"[Fuzzy 7x7] H:{h:.2f}m | Err:{e:.3f}m | Gain:{p_gain:.4f} | Step:{max_step:.3f}m", 
                    throttle_duration_sec=0.5
                )

                # ==================================================================
                # 2. ADJUSTING SAFE LANDING CONDITIONS (STRICT LOGIC HAS BEEN CORRECTED)
                # ==================================================================
                CRITICAL_ALTITUDE = -0.20     
                STRICT_ERROR_LIMIT = 0.05    
                LOOSE_ERROR_LIMIT = 0.20     

                if self.target_z < CRITICAL_ALTITUDE:
                    # At high altitudes: Only allow lowering the height if the error is within your allowed Large range (25cm = 0.25m).
                    if e < 0.25:
                        self.target_z += 0.05
                    else:
                        pass 
                else:
                    # At low altitudes: Force the error to be within the strict limit (5cm) before activating PRECISION_LAND
                    if e < STRICT_ERROR_LIMIT:
                        self.flight_state = "PRECISION_LAND"
                        self.get_logger().info("Landing mode activated.")
                    else:
                        pass 
                # Log flight data for analysis
                self.log_data.append([
                    self.state_timer,
                    self.drone_x, self.drone_y, self.drone_z,
                    self.target_x, self.target_y, self.target_z,
                    self.aruco_detected,
                    self.aruco_offset_x_corrected, self.aruco_offset_y_corrected,
                    h, e, p_gain,
                    self.flight_state
                ])

            else:
                # If ArUco not detected, hover and slowly rotate to find it
                # This helps recover from lost tracking
                pass
            
            
        elif self.flight_state == "PRECISION_LAND":
            import csv
            with open(self.log_file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.log_headers)
                writer.writerows(self.log_data)
            self.get_logger().info(f"Log saved to {self.log_file_path}")


            self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
            cv2.destroyAllWindows()
            self.timer.cancel()
            self.get_logger().info("Precision Landing completed successfully!")
            os._exit(0)

        # Push the Target coordinates from the closed-loop calculation down to the PX4 controller.
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