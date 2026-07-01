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
from utils.offboard_control import public_attitude_callback, publish_offboard_control_mode, publish_trajectory_setpoint, publish_vehicle_command, gazebo_world_pose_callback, local_position_callback

# --- GAZEBO HARMONIC LIBRARY DIRECTLY CONNECTED
from gz.transport13 import Node as GzNode
from gz.msgs10.image_pb2 import Image as GzImage
from gz.msgs10.pose_v_pb2 import Pose_V as GzPoseV
# -----------------------------------------------------------------------------------------------

from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor

class OffboardPrecisionLanding(Node):
    def __init__(self):
        super().__init__('drone_landing_node')
        # === GÁN CÁC FUNCTION THÀNH METHOD CỦA CLASS ===
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
        
        # Initialize the actual tilt angle state value (in radians) - Khởi tạo giá trị trạng thái góc nghiên thực tế 
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

        # --- CẤU HÌNH THÔNG SỐ ĐỂ ƯỚC LƯỢNG POSE ARUCO ---
        # Kích thước thực tế của mã ArUco ngoài thực tế (Ví dụ: 0.4 mét = 40cm)
        self.aruco_marker_length = 0.5 

        # Ma trận Camera giả lập (Thay bằng ma trận thực tế của camera gazebo nếu có để chính xác tuyệt đối)
        # Giả định camera độ phân giải khoảng 640x480 hoặc tương đương với FOV 60 độ
        self.camera_matrix = np.array([[500.0,   0.0, 320.0],
                                       [  0.0, 500.0, 240.0],
                                       [  0.0,   0.0,   1.0]], dtype=np.float32)
        # Hệ số biến dạng (mặc định bằng 0 đối với camera lý tưởng trong Gazebo)
        self.dist_coeffs = np.zeros((5, 1), dtype=np.float32)
        # ------------------------------------------------

        # The ArUco recognition configuration is compatible with OpenCV versions - Cấu hình bộ nhận diện ArUco tương thích với các phiên bản OpenCV 
        # ---------------------------------------------------------------------------------------
        try:
            # OpenCV 4.7+
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
            self.aruco_params = cv2.aruco.DetectorParameters()
            self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
            self.opencv_version_new = True
        except AttributeError:
            # Nếu lỗi, tự động hạ cấp cấu hình tương thích cú pháp OpenCV 4.6
            self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
            self.aruco_params = cv2.aruco.DetectorParameters_create()
            self.opencv_version_new = False

        # Separate Callback Group for ROS 2 Flight Control Flow - Tách biệt Nhóm Callback cho luồng điều khiển bay ROS 2
        # ---------------------------------------------------------------------------------------
        self.control_group = MutuallyExclusiveCallbackGroup()

        # 1. Cấu hình QoS Profile tương thích hoàn toàn với PX4
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        # 2. Publishers điều khiển bay (ROS 2)
        self.offboard_control_mode_pub = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)
        self.trajectory_setpoint_pub = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_profile)
        self.vehicle_command_pub = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', qos_profile)

        # 3. Subscribers theo dõi trạng thái máy bay (ROS 2)
        self.vehicle_status_sub = self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status', self.status_callback, qos_profile, callback_group=self.control_group
        )
        
        # 4. Subscribers theo dõi góc nghiên thực tế
        self.vehicle_attitude_sub = self.create_subscription(
            VehicleAttitude, '/fmu/out/vehicle_attitude', self.public_attitude_callback, qos_profile, callback_group=self.control_group
        )

        # 4b. Subscribers theo dõi vị trí cục bộ từ EKF2 PX4
        self.vehicle_local_position_sub = self.create_subscription(
            VehicleLocalPosition, '/fmu/out/vehicle_local_position', self.local_position_callback, qos_profile, callback_group=self.control_group
        )
        
        # 5. KẾT NỐI CAMERA QUA GZ-TRANSPORT
        self.gz_node = GzNode()
        self.gz_topic = '/world/mworlds/model/x500_mono_cam_down_0/link/camera_link/sensor/camera/image'
        #self.gz_topic = '/world/mworlds/model/x500_depth_down_0/link/camera_link/sensor/IMX214/image'
        
        self.gz_node.subscribe(GzImage, self.gz_topic, self.gazebo_camera_callback)

        # 6. ĐĂNG KÝ LẤY VỊ TRÍ TUYỆT ĐỐI GROUND TRUTH TỪ GAZEBO WORLD
        self.gz_pose_topic = '/world/mworlds/pose/info' 
        self.gz_node.subscribe(GzPoseV, self.gz_pose_topic, self.gazebo_world_pose_callback) # 

        # 7. Vòng lặp điều khiển chính (10Hz)
        self.dt = 0.1  
        self.timer = self.create_timer(self.dt, self.timer_callback, callback_group=self.control_group)
        
        # Trạng thái của FSM điều khiển tự động hạ cánh chính xác
        self.offboard_setpoint_counter = 0
        self.flight_state = "TAKEOFF"
        self.state_timer = 0.0

        # Vị trí Target động (Theo hệ NED của PX4)
        self.target_x = 0.0
        self.target_y = 0.0
        self.target_z = -5.0  # Cất cánh lên độ cao ban đầu 5 mét

        # Vị trí tuyệt đối Ground Truth từ Gazebo World (Hệ ENU: Đông-Bắc-Trên)
        self.world_gt_x = 0.0 
        self.world_gt_y = 0.0  
        self.world_gt_z = 0.0  

        # Khởi tạo cửa sổ hiển thị giao diện OpenCV
        #cv2.namedWindow("Drone FPV Camera View (Gz-Transport)", cv2.WINDOW_AUTOSIZE)
        #cv2.startWindowThread()
        self.get_logger().info("🚀 Precision Landing: ROS2 + Gz-Transport ArUco Tracking READY!")

    def gazebo_camera_callback(self, msg):
        # Hàm xử lý dữ liệu camera: Nhận ảnh từ Gazebo Core, nhận diện ArUco và tính toán sai lệch để điều khiển vòng kín hạ cánh chính xác
        # ---------------------------------------------------------------------------------------
        # Input:
        # - msg.data: Mảng byte nhị phân của ảnh thô từ camera
        # - mgs.width, msg.height: Kích thước ảnh
        # - msg.pixel_format_type: Loại định dạng pixel (3 = RGB_INT8

        # Output:
        # - self.aruco_detected: Cờ nhận diện ArUco
        # - self.aruco_offset_x, self.aruco_offset_y: Sai lệch tính toán từ tâm ảnh đến tâm ArUco (mét)
        # - self.aruco_offset_x_corrected, self.aruco_offset_y_corrected: Sai lệch đã được bù trừ góc nghiêng thân (mét)
        # - Hiển thị ảnh với khung bám ArUco và thông tin trạng thái trên giao diện OpenCV

        try:
            # 1. Giải mã mảng byte nhị phân thành ma trận ảnh
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

            # 2. Thuật toán nhận diện ArUco Marker
            if self.opencv_version_new:
                corners, ids, rejected = self.aruco_detector.detectMarkers(cv_image)
            else:
                corners, ids, rejected = cv2.aruco.detectMarkers(cv_image, self.aruco_dict, parameters=self.aruco_params)
            
            if ids is not None:
                self.aruco_detected = True
                # Vẽ khung bám màu xanh xung quanh mã ArUco tìm thấy
                cv2.aruco.drawDetectedMarkers(cv_image, corners, ids)
                
                # --- TIẾN HÀNH ƯỚC LƯỢNG POSE (VỊ TRÍ & GÓC XOAY) ---
                # Sử dụng hàm ước lượng Pose đơn lẻ cho từng marker
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners, self.aruco_marker_length, self.camera_matrix, self.dist_coeffs
                )
                
                # Lấy rvec (rotation vector) và tvec (translation vector) của marker đầu tiên tìm thấy
                rvec = rvecs[0]
                tvec = tvecs[0]

                # Vẽ hệ trục tọa độ 3D (X: Đỏ, Y: Xanh lá, Z: Xanh dương) lên marker (Độ dài trục vẽ = 0.3 mét)
                cv2.drawFrameAxes(cv_image, self.camera_matrix, self.dist_coeffs, rvec, tvec, 0.3)

                # 1. Trích xuất khoảng cách tịnh tiến từ Camera tới ArUco (Hệ tọa độ Camera: X-phải, Y-xuống, Z-thẳng trước mặt)
                pose_x = tvec[0][0]
                pose_y = tvec[0][1]
                pose_z = tvec[0][2] # Đây chính là khoảng cách thẳng từ camera tới tag

                # 2. Trích xuất và chuyển đổi góc quay từ Vector rvec sang đơn vị ĐỘ (Degrees)
                # rvec[0][0]: Góc xoay quanh trục X (Roll của tag so với cam)
                # rvec[0][1]: Góc xoay quanh trục Y (Pitch của tag so với cam)
                # rvec[0][2]: Góc xoay quanh trục Z (Yaw của tag so với cam - góc bạn cần dùng để xoay drone)
                rot_x = math.degrees(rvec[0][0])
                rot_y = math.degrees(rvec[0][1])
                rot_z = math.degrees(rvec[0][2])


                # 3. Hiển thị thông tin POSE ước lượng được lên màn hình
                pose_text = f"Posecam -> Tag: X:{pose_x:.2f}m Y:{pose_y:.2f}m Z:{pose_z:.2f}m"
                cv2.putText(cv_image, pose_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                # ----------------------------------------------------

                # 4. Hiển thị thông tin góc quay rvec (Độ) lên màn hình ở tọa độ Y=85 (Màu xanh cyan)
                rvec_text = f"Rvec (Deg) -> RotX:{rot_x:.1f} RotY:{rot_y:.1f} RotZ(Yaw):{rot_z:.1f}"
                cv2.putText(cv_image, rvec_text, (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

                # Tính toán tâm hình học của ArUco (Trung bình cộng 4 góc)
                c = corners[0][0]
                aruco_center_x = int((c[0][0] + c[1][0] + c[2][0] + c[3][0]) / 4)
                aruco_center_y = int((c[0][1] + c[1][1] + c[2][1] + c[3][1]) / 4)
                
                # Vẽ điểm tâm ArUco màu xanh lá
                cv2.circle(cv_image, (aruco_center_x, aruco_center_y), 5, (0, 255, 0), -1)
                
                # Tính toán sai lệch pixel từ tâm ảnh đến tâm ArUco
                pixel_error_x = aruco_center_x - cam_center_x
                pixel_error_y = cam_center_y - aruco_center_y

                # Tính toán quy đổi pixel ra mét dựa trên độ cao    
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
                
                # Vẽ đường nối định vị từ tâm drone đến tâm ArUco
                cv2.line(cv_image, (cam_center_x, cam_center_y), (aruco_center_x, aruco_center_y), (255, 0, 0), 2)
            else:
                self.aruco_detected = False

            # Vẽ hồng tâm trung tâm của Drone (Màu đỏ)
            cv2.circle(cv_image, (cam_center_x, cam_center_y), 6, (0, 0, 255), 2)
            
            # Hiển thị dữ liệu trạng thái xử lý lên màn hình đồ họa
            status_text = f"State: {self.flight_state} | Detected: {self.aruco_detected}"
            cv2.putText(cv_image, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            self.latest_preview_image = cv_image # Lưu ảnh vào biến đệm của Node
            
        except Exception as e:
            self.get_logger().error(f"Lỗi xử lý ảnh và bám ArUco: {e}")

    def status_callback(self, msg):
        self.vehicle_status = msg

    def timer_callback(self):
        # Hàm vòng lặp chính điều khiển bay (10Hz): Máy trạng thái hữu hạn xử lý bài toán hạ cánh chính xác
        # ---------------------------------------------------------------------------------------
        # Input:
        # - self.vehicle_status: Trạng thái máy bay cập nhật từ PX4
        # - self.aruco_detected, self.aruco_offset_x_corrected,
        #   self.aruco_offset_y_corrected: Kết quả nhận diện ArUco và sai lệch đã bù trừ góc nghiêng thân
        # Output: 
        # - self.target_x, self.target_y, self.target_z: Tọa độ mục tiêu động được tính toán để điều khiển vòng kín hạ cánh chính xác
        # - Giai đoạn 1: Cất cánh và di chuyển đến vị trí gần ArUco (Waypoint)
        # - Giai đoạn 2: Tìm kiếm và bám đuổi Ar
        # - Giai đoạn 3: Khi đã tiếp cận cực cận ArUco, ra lệnh khóa hạ cánh trực tiếp cho FCU
        # - Giai đoạn 4: Hiển thị thông tin trạng thái và sai lệch trên giao diện đồ họa OpenCV
        # Lưu ý: Vòng lặp này sẽ liên tục cập nhật tọa độ mục tiêu dựa trên sai lệch đã bù trừ để điều khiển drone di chuyển về phía tâm ArUco một cách mượt mà và ổn định nhất có thể.
        
        if self.offboard_setpoint_counter < 10:
            self.publish_offboard_control_mode()
            self.publish_trajectory_setpoint(0.0, 0.0, 0.0)
            self.offboard_setpoint_counter += 1
            return

        if self.offboard_setpoint_counter == 10:
            self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)  #Arm máy bay
            self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)      #Chuyển sang Offboard Mode
            self.offboard_setpoint_counter += 1

        self.publish_offboard_control_mode()
        self.state_timer += self.dt

        self.get_logger().info(
            f"📊 [PX4 EKF2]: X={self.target_x:.2f}m, Y={self.target_y:.2f}m, Cao độ={-self.target_z:.2f}m, Độ lệch gốc X = {self.aruco_offset_x_corrected:.2f}m,Độ lệch gốc Y = {self.aruco_offset_y_corrected:.2f}"
            f"🌍 [Gazebo World GT]: X={self.world_gt_x:.2f}m, Y={self.world_gt_y:.2f}m, Cao độ={self.world_gt_z:.2f}m",
            throttle_duration_sec=1.0
        )

        # self.state_timer = 0.0
        # CẬP NHẬT ĐỒNG BỘ ĐỘ CAO (Bóc tách ước lượng từ trạng thái PX4)
        # Trong thực tế, bạn có thể subscribe thêm /fmu/out/vehicle_local_position để lấy vị trí mượt hơn
        # Ở đây ta đồng bộ hóa tương đối theo các mốc đích để kiểm soát vòng kín
        
        # MÁY TRẠNG THÁI (FSM) ĐIỀU KHIỂN LAI THỊ GIÁC MÁY TÍNH
        if self.flight_state == "TAKEOFF":
            # Giai đoạn 1.1: Cất cánh thẳng đứng lên độ cao 5 mét (X=0, Y=0, Z=-5)
            self.target_x = 0.0
            self.target_y = 0.0
            self.target_z = -25

            if self.state_timer > 10 or abs(self.drone_z + 10.0) < 1.0: # Cho drone 7 giây để đạt độ cao hoặc kiểm tra nếu đã gần đạt độ cao mục tiêu
                self.flight_state = "WP1"
                self.state_timer = 0.0
                self.get_logger().info(f"[TAKEOFF MODE] Đạt độ cao {self.state_timer:.2f}m")     
        
        elif self.flight_state == "WP1":
            # Giai đoạn 1.2: Di chuyển tới tọa độ điểm WP1 (Z=-8)
            # self.target_x = 205.15
            # self.target_y = 327.33
            self.target_x = 108.32
            self.target_y = -136.42
            self.target_z = -25.0
            
            if self.state_timer > 30.0:  # Cho drone 15 giây để hoàn thành quãng đường di chuyển
                self.flight_state = "DESTINATION_REACHED"
                self.state_timer = 0.0
                self.get_logger().info(f"[OFFBOARD MODE] {self.target_z:.2f} : Đã tới vị trí mục tiêu WP1. Bắt đầu hạ độ cao")
        
        elif self.flight_state == "DESTINATION_REACHED":
            # Giai đoạn 1.2: Hạ đôj cao
            #self.target_x = 205.15
            #self.target_y = 327.33
            # self.target_x = 108.32
            # self.target_y = -136.42
            self.target_z = -10.0

            if self.state_timer > 10.0 or abs(self.world_gt_z - 10.0) < 0.5: # Cho drone 5 giây để ổn định độ cao và bắt đầu tìm kiếm ArUco
                self.flight_state = "SEARCH_TRACK"
                self.state_timer = 0.0
                self.get_logger().info("[OFFBOARD MODE] Chuyển sang chế độ tìm điểm hạ cánh.")

        elif self.flight_state == "SEARCH_TRACK":
            if self.aruco_detected:
                
                # ==================================================================
                # 1. BỘ ĐIỀU KHIỂN MỜ NÂNG CAO (N-INPUT N-OUTPUT FUZZY CONTROLLER)
                # ==================================================================
                # Lấy cao độ thực tế h (NED z âm) và sai số ngang thực tế e (mét)
                h = abs(self.drone_z) if abs(self.drone_z) > 0.3 else abs(self.target_z)
                e = math.sqrt(self.aruco_offset_x_corrected**2 + self.aruco_offset_y_corrected**2)

                # --- MỜ HÓA ĐẦU VÀO 1: CAO ĐỘ (ALTITUDE - Ngưỡng: 3m, 5m, 8m) ---
                # Tập mờ LOW (Đạt 1.0 khi h <= 3m, triệt tiêu tại 5m)
                if h <= 3.0:
                    mu_alt_low = 1.0
                elif 3.0 < h < 5.0:
                    mu_alt_low = (5.0 - h) / (5.0 - 3.0)
                else:
                    mu_alt_low = 0.0

                # Tập mờ MEDIUM (Đạt đỉnh 1.0 tại đúng 5m, triệt tiêu ở 3m và 8m)
                if h <= 3.0 or h >= 8.0:
                    mu_alt_med = 0.0
                elif 3.0 < h <= 5.0:
                    mu_alt_med = (h - 3.0) / (5.0 - 3.0)
                else: # 5.0 < h < 8.0
                    mu_alt_med = (8.0 - h) / (8.0 - 5.0)

                # Tập mờ HIGH (Bắt đầu tăng từ 5m, đạt 1.0 từ 8m trở lên)
                if h <= 5.0:
                    mu_alt_high = 0.0
                elif 5.0 < h < 8.0:
                    mu_alt_high = (h - 5.0) / (8.0 - 5.0)
                else:
                    mu_alt_high = 1.0

                # --- MỜ HÓA ĐẦU VÀO 2: SAI SỐ NGANG (ERROR - Ngưỡng: 15cm, 20cm, 25cm) ---
                # Đổi đơn vị cấu hình từ cm sang mét: 15cm = 0.15m, 20cm = 0.20m, 25cm = 0.25m
                # Tập mờ SMALL (Đạt 1.0 khi e <= 0.15m, triệt tiêu tại 0.20m)
                if e <= 0.15:
                    mu_err_small = 1.0
                elif 0.15 < e < 0.20:
                    mu_err_small = (0.20 - e) / (0.20 - 0.15)
                else:
                    mu_err_small = 0.0

                # Tập mờ NORMAL (Đạt đỉnh 1.0 tại đúng 0.20m, triệt tiêu ở 0.15m và 0.25m)
                if e <= 0.15 or e >= 0.25:
                    mu_err_norm = 0.0
                elif 0.15 < e <= 0.20:
                    mu_err_norm = (e - 0.15) / (0.20 - 0.15)
                else: # 0.20 < e < 0.25
                    mu_err_norm = (0.25 - e) / (0.25 - 0.20)

                # Tập mờ LARGE (Bắt đầu tăng từ 0.20m, đạt 1.0 từ 0.25m trở lên)
                if e <= 0.20:
                    mu_err_large = 0.0
                elif 0.20 < e < 0.25:
                    mu_err_large = (e - 0.20) / (0.25 - 0.20)
                else:
                    mu_err_large = 1.0


                # --- ĐỊNH NGHĨA 6 MỨC ĐẦU RA ĐƠN TRỊ (OUTPUT SINGLETONS) ---
                g_very_small = 0.01
                g_small      = 0.03
                g_normal     = 0.04  # Định nghĩa mức bình ổn mượt thay vì 0.0 để tránh đứng im cứng
                g_med        = 0.06
                g_large      = 0.08
                g_very_large = 0.10

                # --- HỆ THỐNG 9 LUẬT MỜ VÀ TÍNH TRỌNG SỐ KÍCH HOẠT (AND -> min) ---
                w1 = min(mu_alt_high, mu_err_small)  # Luật 1: Cao + Sai số nhỏ -> Gain Large
                w2 = min(mu_alt_high, mu_err_norm)   # Luật 2: Cao + Sai số vừa -> Gain Very Large
                w3 = min(mu_alt_high, mu_err_large)  # Luật 3: Cao + Sai số lớn -> Gain Very Large

                # Tầng giữa (Medium Altitude)
                w4 = min(mu_alt_med, mu_err_small)   # Luật 4: Vừa + Sai số nhỏ -> Gain Med
                w5 = min(mu_alt_med, mu_err_norm)    # Luật 5: Vừa + Sai số vừa -> Gain Med
                w6 = min(mu_alt_med, mu_err_large)   # Luật 6: Vừa + Sai số lớn -> Gain Large

                # Tầng sát đất (Low Altitude)
                w7 = min(mu_alt_low, mu_err_small)   # Luật 7: Thấp + Sai số nhỏ -> Gain Very Small (Chống chao đảo mặt đất)
                w8 = min(mu_alt_low, mu_err_norm)    # Luật 8: Thấp + Sai số vừa -> Gain Small
                w9 = min(mu_alt_low, mu_err_large)   # Luật 9: Thấp + Sai số lớn -> Gain Med (Tăng lực kéo khi bị gió thổi lệch)

                # --- GIẢI MỜ TRỌNG TÂM (Defuzzification) ---
                sum_w = w1 + w2 + w3 + w4 + w5 + w6 + w7 + w8 + w9
                
                if sum_w > 0.0:
                    p_gain = (w1 * g_large + w2 * g_very_large + w3 * g_very_large +
                              w4 * g_med   + w5 * g_med        + w6 * g_large +
                              w7 * g_very_small + w8 * g_small + w9 * g_med) / sum_w
                else:
                    p_gain = g_small # Giá trị an toàn dự phòng

                # ==================================================================
                # TÍNH TOÁN BƯỚC NHẢY VÀ CẬP NHẬT TRAJECTORY SETPOINT CỦA PX4
                # ==================================================================
                step_x = self.aruco_offset_y_corrected * p_gain
                step_y = self.aruco_offset_x_corrected * p_gain

                # Bộ giới hạn bước nhảy an toàn Saturation tránh giật thân cơ học (20cm)
                max_step = 0.2
                if h > 4.0:
                    max_step = 0.08  # Giảm từ 0.2 xuống 0.08 để bóp chết dao động lắc qua lắc lại
                else:
                    max_step = 0.04
                step_x = max(-max_step, min(max_step, step_x))
                step_y = max(-max_step, min(max_step, step_y))

                # Thực hiện cộng dồn setpoint
                self.target_x = self.target_x + step_x
                self.target_y = self.target_y + step_y
                
                # In thông số phục vụ lấy số liệu xuất đồ thị cho Paper
                self.get_logger().info(
                    f"[Fuzzy 3x3] H:{h:.2f}m | Err:{e:.3f}m | Đăng_Ký_Gain:{p_gain:.4f}", 
                    throttle_duration_sec=0.5
                )

                # ==================================================================
                # 2. ĐIỀU CHỈNH ĐIỀU KIỆN HẠ CÁNH AN TOÀN (LOGIC CHẶT CHẼ ĐÃ SỬA
                # ==================================================================
                CRITICAL_ALTITUDE = -0.2     
                STRICT_ERROR_LIMIT = 0.05    
                LOOSE_ERROR_LIMIT = 0.20     

                if self.target_z < CRITICAL_ALTITUDE:
                    # Ở trên cao: Chỉ cho phép hạ độ cao nếu sai số nằm trong vùng Large cho phép của bạn (25cm = 0.25m)
                    if e < 0.25:
                        self.target_z += 0.05  
                        # self.drone_z = self.target_z
                    else:
                        pass # Đóng băng độ cao nếu lệch quá 25cm để bộ luật mờ tăng Gain kéo thân về
                else:
                    # Đang cận đất: Ép sai số về mức cực nhỏ 5cm theo ý bạn mới kích hoạt PRECISION_LAND
                    if e < STRICT_ERROR_LIMIT:
                        self.flight_state = "PRECISION_LAND"
                        self.get_logger().info("🎯 [Fuzzy 3x3] Đã đồng tâm tuyệt đối! Khóa hạ cánh tiếp đất.")
                    else:
                        pass # Giữ im độ cao rà sát đất để ép bám tâm

                #if horizontal_error < LOOSE_ERROR_LIMIT:
                #    if self.target_z < CRITICAL_ALTITUDE:
                #        self.target_z += 0.15  
                #        self.drone_z = self.target_z
                #    else:
                #        self.flight_state = "PRECISION_LAND"
                #        self.get_logger().info("🎯 [Fuzzy Mode] Tâm đạt chuẩn tuyệt đối! Khóa tiếp đất.")
                #else:
                #    pass # Lệch tâm lớn thì đứng im Hover để Fuzzy Gain kéo về tâm
            else:
                # if self.target_z > -15.0:
                #    self.target_z -= 0.05
                pass

        elif self.flight_state == "PRECISION_LAND":
            # Giai đoạn 3: Ra lệnh trực tiếp cho FCU thực hiện khóa hạ cánh an toàn dứt điểm tại tâm ArUco
            self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
            cv2.destroyAllWindows()
            self.timer.cancel()
            self.get_logger().info("🏁 Đã thực hiện Precision Landing thành công lên tâm mã ArUco!")
            os._exit(0) # Thoát hoàn toàn chương trình sau khi hạ cánh thành công để tránh gửi lệnh tiếp tục sau khi đã hạ cánh xong

        # Đẩy tọa độ Target tính toán vòng kín xuống bộ điều khiển của PX4
        self.publish_trajectory_setpoint(self.target_x, self.target_y, self.target_z)


def main(args=None):
    rclpy.init(args=args)
    node = OffboardPrecisionLanding()
    
    #executor = MultiThreadedExecutor()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)

    # Khởi tạo cửa sổ hiển thị đồ họa OpenCV DUY NHẤT tại Luồng Chính (Main Thread)
    cv2.namedWindow("Drone FPV Camera View (Gz-Transport)", cv2.WINDOW_AUTOSIZE)
    
    try:
        while rclpy.ok():
            # Cho phép executor xử lý các callback ROS2 và Gz-Transport trong 10ms
            executor.spin_once(timeout_sec=0.01)
            
            # Kiểm tra nếu luồng phụ đã xử lý xong và có ảnh mới
            if hasattr(node, 'latest_preview_image') and node.latest_preview_image is not None:
                # Hiển thị ảnh an toàn từ Luồng Chính
                cv2.imshow("Drone FPV Camera View (Gz-Transport)", node.latest_preview_image)
                
                # Hàm waitKey bắt buộc phải chạy ở Main Thread để giữ cửa sổ không bị crash
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