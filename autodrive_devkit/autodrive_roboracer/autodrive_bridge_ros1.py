#!/usr/bin/env python3

################################################################################

# Copyright (c) 2025, Tinker Twins
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

################################################################################

# ROS 1 module imports
import argparse  # Command-line argument parsing
import rospy  # ROS 1 client library for Python
import tf2_ros  # ROS bindings for tf2 library to handle transforms
from time import monotonic
from std_msgs.msg import Int32, Float32, Header  # Int32, Float32 and Header message classes
from geometry_msgs.msg import Point, TransformStamped  # Point and TransformStamped message classes
from sensor_msgs.msg import JointState, Imu, LaserScan, Image  # JointState, Imu, LaserScan and Image message classes
from geometry_msgs.msg import PoseWithCovarianceStamped  # PoseWithCovarianceStamped message class

from nav_msgs.msg import Odometry  # Odometry message
try:
    from tf_transformations import euler_from_quaternion, quaternion_from_euler  # ROS2/Noetic style
except ImportError:
    from tf.transformations import euler_from_quaternion
    from tf.transformations import quaternion_from_euler  # ROS1 melodic/kinetic fallback
from threading import Thread  # Thread-based parallelism

# Python module imports
from cv_bridge import CvBridge  # ROS bridge for opencv library to handle images
from gevent import pywsgi  # Pure-Python gevent-friendly WSGI server
from gevent import sleep as gevent_sleep  # Cooperative sleep for Socket.IO/gevent loop pacing
from geventwebsocket.handler import WebSocketHandler  # Handler for WebSocket messages and lifecycle events
import socketio  # Socket.IO realtime client and server
import math  # Mathematical functions
import numpy as np  # Scientific computing
import base64  # Base64 binary-to-text encoding/decoding scheme
from io import BytesIO  # Manipulate bytes data in memory
from PIL import Image  # Python Imaging Library's (PIL's) Image module
import gzip  # Inbuilt module to compress and decompress data and files
import autodrive_roboracer.config as config  # AutoDRIVE Ecosystem ROS 2 configuration for RoboRacer vehicle

from ackermann_msgs.msg import AckermannDriveStamped
import tf_conversions

################################################################################

# AutoDRIVE class
class AutoDRIVE:
    def __init__(self):
        # Vehicle data
        self.id = 1
        self.throttle = 0
        self.steering = 0
        self.speed = 0
        self.encoder_angles = np.zeros(2, dtype=float)
        self.position = np.zeros(3, dtype=float)
        self.orientation_quaternion = np.zeros(4, dtype=float)
        self.orientation_euler_angles = np.zeros(3, dtype=float)
        self.angular_velocity = np.zeros(3, dtype=float)
        self.linear_velocity = np.zeros(3, dtype=float)
        self.linear_acceleration = np.zeros(3, dtype=float)
        self.lidar_scan_rate = 40
        self.lidar_range_array = np.zeros(1080, dtype=float)
        self.lidar_intensity_array = np.asarray([])
        self.front_camera_image = np.zeros((192, 108, 3), dtype=np.uint8)
        # Race data
        self.lap_count = 0
        self.lap_time = 0
        self.last_lap_time = 0
        self.best_lap_time = 0
        self.collision_count = 0
        # Vehicle commands
        self.throttle_command = 0.0  # [-1, 1]
        self.steering_command = 0.0  # [-1, 1]
        # Simulation commands
        self.reset_command = False  # True or False

        self.previous_collision_count = -1

    def init_ackermann_to_autodrive_params(self):
        # for ackermann to autodrive
        self.max_speed = 22.8 #rospy.get_param("~max_speed", 22.8)
        self.max_steering_angle = 0.5236 #rospy.get_param("~max_steering_angle", 0.5236)

    '''
    def init_wheel_odometry(self):
        # for wheel odometry
        self.wheel_radius = 0.0590 # rospy.get_param("~wheel_radius", 0.0590)  # meters
        self.wheel_base = 0.3240 # rospy.get_param("~wheel_base", 0.3240)  # meters
        self.counts_per_rev = 6.5 # rospy.get_param("~counts_per_rev", 6.5)
        self.publish_tf = False # rospy.get_param("~publish_tf", False)
        self.max_acceleration = 4.0 # rospy.get_param("~max_acceleration", 4.0)

        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.est_speed = 0.0
        self.angular_speed = 0.0

        self.current_left_position = 0.0
        self.current_right_position = 0.0
        self.prev_left_position = 0.0
        self.prev_right_position = 0.0

        self.last_steering_angle = 0.0
        #self.first_steering_msg = False

        self.current_time = rospy.Time.now()
        self.prev_time = rospy.Time.now()
    '''





################################################################################

# Global declarations
global autodrive_bridge, cv_bridge, publishers, transform_broadcaster, static_broadcaster, real_mode
autodrive = AutoDRIVE()
real_mode = False
initpose_sent = False
initpose_reason = "connect"
initpose_map_tx = 3.1532168
initpose_map_ty = -0.7809472
initpose_yaw_offset = math.pi / 2.0
initpose_x_covariance = 0.01
initpose_y_covariance = 0.01
initpose_yaw_covariance = 0.0076
reinit_on_collision = True
bridge_rate_hz = 40.0
bridge_period_sec = 1.0 / bridge_rate_hz
next_bridge_emit_time = monotonic()

#########################################################
# ROS 1 MESSAGE GENERATING FUNCTIONS
#########################################################

def create_int_msg(i, val):
    i.data = int(val)
    return i


def create_float_msg(f, val):
    f.data = float(val)
    return f


def create_joint_state_msg(js, joint_angle, joint_name, frame_id):
    js.header = Header()
    js.header.stamp = rospy.Time.now()
    js.header.frame_id = frame_id
    js.name = [joint_name]
    js.position = [joint_angle]
    js.velocity = []
    js.effort = []
    return js


def create_point_msg(p, position):
    p.x = position[0]
    p.y = position[1]
    p.z = position[2]
    return p


def create_imu_msg(imu, orientation_quaternion, angular_velocity, linear_acceleration, timestamp = None):
    imu.header = Header()
    if not timestamp:
        timestamp = rospy.Time.now()
    imu.header.stamp = timestamp
    imu.header.frame_id = 'imu_autodrive'
    imu.orientation.x = orientation_quaternion[0]
    imu.orientation.y = orientation_quaternion[1]
    imu.orientation.z = orientation_quaternion[2]
    imu.orientation.w = orientation_quaternion[3]
    imu.orientation_covariance = [0.0025, 0.0, 0.0, 0.0, 0.0025, 0.0, 0.0, 0.0, 0.0025]
    imu.angular_velocity.x = angular_velocity[0]
    imu.angular_velocity.y = angular_velocity[1]
    imu.angular_velocity.z = angular_velocity[2]
    imu.angular_velocity_covariance = [0.0025, 0.0, 0.0, 0.0, 0.0025, 0.0, 0.0, 0.0, 0.0025]
    imu.linear_acceleration.x = linear_acceleration[0]
    imu.linear_acceleration.y = linear_acceleration[1]
    imu.linear_acceleration.z = linear_acceleration[2]
    imu.linear_acceleration_covariance = [0.0025, 0.0, 0.0, 0.0, 0.0025, 0.0, 0.0, 0.0, 0.0025]
    return imu


def create_laserscan_msg(ls, lidar_scan_rate, lidar_range_array, lidar_intensity_array, timestamp = None):
    ls.header = Header()
    if not timestamp:
        timestamp = rospy.Time.now()
    ls.header.stamp = timestamp
    ls.header.frame_id = 'laser'
    ls.angle_min = -2.35619  # Minimum angle of laser scan (0 degrees)
    ls.angle_max = 2.35619  # Maximum angle of laser scan (270 degrees)
    ls.angle_increment = 0.004363323  # Angular resolution of laser scan (0.25 degree)
    ls.time_increment = (1 / lidar_scan_rate) / 1080  # Time required to scan 1 degree
    ls.scan_time = 1 / lidar_scan_rate  # Time required to complete a scan
    ls.range_min = 0.06  # Minimum sensor range (in meters)
    ls.range_max = 10.0  # Maximum sensor range (in meters)
    ls.ranges = lidar_range_array
    ls.intensities = lidar_intensity_array
    return ls


def create_image_msg(img, frame_id, timestamp = None):
    img = cv_bridge.cv2_to_imgmsg(img, encoding="rgb8")
    img.header = Header()
    if not timestamp:
        timestamp = rospy.Time.now()
    img.header.stamp = timestamp
    img.header.frame_id = frame_id
    return img


def create_tf_msg(child_frame_id, parent_frame_id, position_tf, orientation_tf, timestamp=None):
    tf = TransformStamped()
    if not timestamp:
        timestamp = rospy.Time.now()
    tf.header.stamp = timestamp
    tf.header.frame_id = parent_frame_id
    tf.child_frame_id = child_frame_id
    tf.transform.translation.x = position_tf[0]  # Pos X
    tf.transform.translation.y = position_tf[1]  # Pos Y
    tf.transform.translation.z = position_tf[2]  # Pos Z
    tf.transform.rotation.x = orientation_tf[0]  # Quat X
    tf.transform.rotation.y = orientation_tf[1]  # Quat Y
    tf.transform.rotation.z = orientation_tf[2]  # Quat Z
    tf.transform.rotation.w = orientation_tf[3]  # Quat W
    return tf


def broadcast_transforms(tf_broadcaster, autodrive, timestamp):
    tf_list = []
    parent_frame = "base_link" if real_mode else "roboracer_1"
    if not real_mode:
        tf_list.append(create_tf_msg("roboracer_1", "world", autodrive.position, autodrive.orientation_quaternion, timestamp))  # Vehicle frame defined at center of rear axle
    tf_list.append(create_tf_msg("left_encoder", parent_frame, np.asarray([0.0, 0.12, 0.0]), quaternion_from_euler(0.0, 120 * autodrive.encoder_angles[0] % 6.283, 0.0), timestamp))
    tf_list.append(create_tf_msg("right_encoder", parent_frame, np.asarray([0.0, -0.12, 0.0]), quaternion_from_euler(0.0, 120 * autodrive.encoder_angles[1] % 6.283, 0.0), timestamp))
    tf_list.append(create_tf_msg("ips", parent_frame, np.asarray([0.08, 0.0, 0.055]), np.asarray([0.0, 0.0, 0.0, 1.0]), timestamp))
    if not real_mode:
        tf_list.append(create_tf_msg("imu", parent_frame, np.asarray([0.08, 0.0, 0.055]), np.asarray([0.0, 0.0, 0.0, 1.0]), timestamp))
        tf_list.append(create_tf_msg("lidar", parent_frame, np.asarray([0.2733, 0.0, 0.096]), np.asarray([0.0, 0.0, 0.0, 1.0]), timestamp))
    tf_list.append(create_tf_msg("front_camera", parent_frame, np.asarray([-0.015, 0.0, 0.15]), np.asarray([0, 0.0871557, 0, 0.9961947]), timestamp))
    tf_list.append(create_tf_msg("front_left_wheel", parent_frame, np.asarray([0.33, 0.118, 0.0]), quaternion_from_euler(0.0, 0.0, np.arctan((2 * 0.141537 * np.tan(autodrive.steering)) / (2 * 0.141537 - 2 * 0.0765 * np.tan(autodrive.steering)))), timestamp))
    tf_list.append(create_tf_msg("front_right_wheel", parent_frame, np.asarray([0.33, -0.118, 0.0]), quaternion_from_euler(0.0, 0.0, np.arctan((2 * 0.141537 * np.tan(autodrive.steering)) / (2 * 0.141537 + 2 * 0.0765 * np.tan(autodrive.steering)))), timestamp))
    tf_list.append(create_tf_msg("rear_left_wheel", parent_frame, np.asarray([0.0, 0.118, 0.0]), quaternion_from_euler(0.0, autodrive.encoder_angles[0] % 6.283, 0.0), timestamp))
    tf_list.append(create_tf_msg("rear_right_wheel", parent_frame, np.asarray([0.0, -0.118, 0.0]), quaternion_from_euler(0.0, autodrive.encoder_angles[1] % 6.283, 0.0), timestamp))
    for tf in tf_list:
        tf_broadcaster.sendTransform(tf)


def send_static_transforms():
    """
    Static TFs are owned by the stack launch files and Cartographer in real mode.
    Keeping this as a no-op prevents the bridge from adding a second map/odom/base_link chain.
    """
    return


def pace_bridge_emit():
    global next_bridge_emit_time

    if bridge_rate_hz <= 0.0:
        return

    now = monotonic()
    delay = next_bridge_emit_time - now
    if delay > 0.0:
        gevent_sleep(delay)
        now = monotonic()

    next_bridge_emit_time += bridge_period_sec
    if next_bridge_emit_time < now:
        next_bridge_emit_time = now + bridge_period_sec


#########################################################
# ROS 1 MESSAGE DEFINITIONS
#########################################################

msg_int32 = Int32()
msg_float32 = Float32()
msg_jointstate = JointState()
msg_point = Point()
msg_pose_with_covariance_stamped = PoseWithCovarianceStamped()
msg_imu = Imu()
msg_imu_real = Imu()
msg_laserscan = LaserScan()
msg_laserscan_real = LaserScan()
msg_transform = TransformStamped()
msg_odom = Odometry()


def parse_v1_orientation_euler_angles(data):
    for key in ("V1 Orientation Euler Angle", "V1 Orientation Euler Angles"):
        if key not in data:
            continue

        raw = data[key]
        if isinstance(raw, (list, tuple, np.ndarray)):
            values = np.asarray(raw, dtype=float)
        else:
            text = str(raw).replace(",", " ").strip("[]()")
            values = np.fromstring(text, dtype=float, sep=' ')

        if values.size >= 3:
            return values[:3]

    return np.asarray([], dtype=float)


def yaw_from_v1_orientation(orientation_quaternion, orientation_euler_angles=None):
    quat = np.asarray(orientation_quaternion, dtype=float)
    if quat.size >= 4 and np.all(np.isfinite(quat[:4])) and np.linalg.norm(quat[:4]) > 0.0:
        _, _, yaw = euler_from_quaternion((
            float(quat[0]),
            float(quat[1]),
            float(quat[2]),
            float(quat[3])
        ))
        return yaw

    if orientation_euler_angles is not None and len(orientation_euler_angles) >= 3:
        if np.all(np.isfinite(orientation_euler_angles[:3])):
            yaw = float(orientation_euler_angles[2])
            return math.radians(yaw) if abs(yaw) > 2.0 * math.pi else yaw

    rospy.logwarn("Invalid V1 orientation; using zero yaw for /initialpose")
    return 0.0


def v1_pose_to_initialpose_map(position, orientation_quaternion, orientation_euler_angles=None):
    xw = float(position[0])
    yw = float(position[1])

    map_position = (
        -yw + initpose_map_tx,
        xw + initpose_map_ty,
        0.0
    )

    yaw_w = yaw_from_v1_orientation(orientation_quaternion, orientation_euler_angles)
    yaw_m = yaw_w + initpose_yaw_offset
    map_orientation = quaternion_from_euler(0.0, 0.0, yaw_m)

    return map_position, map_orientation


def make_initialpose_msg(position, orientation_quaternion, orientation_euler_angles=None, timestamp=None):
    if timestamp is None:
        timestamp = rospy.Time.now()

    map_position, map_orientation_quaternion = v1_pose_to_initialpose_map(
        position, orientation_quaternion, orientation_euler_angles
    )

    msg = PoseWithCovarianceStamped()
    msg.header.stamp = timestamp
    msg.header.frame_id = "map"
    msg.pose.pose.position.x = map_position[0]
    msg.pose.pose.position.y = map_position[1]
    msg.pose.pose.position.z = map_position[2]
    msg.pose.pose.orientation.x = map_orientation_quaternion[0]
    msg.pose.pose.orientation.y = map_orientation_quaternion[1]
    msg.pose.pose.orientation.z = map_orientation_quaternion[2]
    msg.pose.pose.orientation.w = map_orientation_quaternion[3]

    cov = [0.0] * 36
    cov[0] = initpose_x_covariance
    cov[7] = initpose_y_covariance
    cov[35] = initpose_yaw_covariance
    msg.pose.covariance = cov
    return msg


def publish_initialpose(position, orientation_quaternion, orientation_euler_angles, reason, timestamp=None):
    global initpose_sent

    if 'pub_initpose' not in publishers:
        return False

    msg = make_initialpose_msg(
        position, orientation_quaternion, orientation_euler_angles, timestamp
    )
    publishers['pub_initpose'].publish(msg)
    initpose_sent = True
    rospy.loginfo(
        "Published /initialpose from AutoDRIVE V1 pose (%s): raw=(%.3f, %.3f, %.3f), map=(%.3f, %.3f, %.3f)",
        reason,
        position[0], position[1], position[2],
        msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z
    )
    return True

#########################################################
# ROS 1 PUBLISHER FUNCTIONS
#########################################################

# VEHICLE DATA PUBLISHER FUNCTIONS


def publish_actuator_feedbacks(throttle, steering):
    publishers['pub_throttle'].publish(create_float_msg(msg_float32, throttle))
    publishers['pub_steering'].publish(create_float_msg(msg_float32, steering))


def publish_speed_data(speed):
    publishers['pub_speed'].publish(create_float_msg(msg_float32, speed))


def publish_encoder_data(encoder_angles):
    publishers['pub_left_encoder'].publish(create_joint_state_msg(msg_jointstate, encoder_angles[0], "left_encoder", "left_encoder"))
    publishers['pub_right_encoder'].publish(create_joint_state_msg(msg_jointstate, encoder_angles[1], "right_encoder", "right_encoder"))


def publish_ips_data(position):
    publishers['pub_ips'].publish(create_point_msg(msg_point, position))


def publish_imu_data(orientation_quaternion, angular_velocity, linear_acceleration, timestamp):
    publishers['pub_imu'].publish(create_imu_msg(msg_imu, orientation_quaternion, angular_velocity, linear_acceleration, timestamp))


def publish_raw_imu_data(orientation_quaternion, angular_velocity, linear_acceleration, timestamp):
    imu_msg = create_imu_msg(msg_imu_real, orientation_quaternion, angular_velocity, linear_acceleration, timestamp)
    imu_msg.header.frame_id = "imu"
    publishers['pub_raw_imu'].publish(imu_msg)


def publish_lidar_scan(lidar_scan_rate, lidar_range_array, lidar_intensity_array, timestamp):
    publishers['pub_lidar'].publish(create_laserscan_msg(msg_laserscan, lidar_scan_rate, lidar_range_array.tolist(), lidar_intensity_array.tolist(), timestamp))


def publish_scan(lidar_scan_rate, lidar_range_array, lidar_intensity_array, timestamp):
    # Mirror of publish_lidar_scan for /scan topic
    scan_msg = create_laserscan_msg(msg_laserscan_real, lidar_scan_rate, lidar_range_array.tolist(), lidar_intensity_array.tolist(), timestamp)
    scan_msg.header.frame_id = "laser"
    publishers['pub_scan'].publish(scan_msg)

def publish_current_pose():
    global autodrive, transform_broadcaster
    odom_msg = Odometry()
    odom_msg.header.stamp = autodrive.current_time
    odom_msg.header.frame_id = "odom"
    odom_msg.child_frame_id = "base_link"

    odom_msg.pose.pose.position.x = autodrive.x
    odom_msg.pose.pose.position.y = autodrive.y
    odom_msg.pose.pose.position.z = 0.0

    q = tf_conversions.transformations.quaternion_from_euler(0.0, 0.0, autodrive.theta)
    odom_msg.pose.pose.orientation.x = q[0]
    odom_msg.pose.pose.orientation.y = q[1]
    odom_msg.pose.pose.orientation.z = q[2]
    odom_msg.pose.pose.orientation.w = q[3]

    odom_msg.twist.twist.linear.x = autodrive.est_speed
    odom_msg.twist.twist.angular.z = autodrive.angular_speed

    publishers['pub_odom'].publish(odom_msg)

    '''
    if autodrive.publish_tf:
        odom_tf = TransformStamped()
        odom_tf.header.stamp = autodrive.current_time
        odom_tf.header.frame_id = "odom"
        odom_tf.child_frame_id = "base_link"
        odom_tf.transform.translation.x = autodrive.x
        odom_tf.transform.translation.y = autodrive.y
        odom_tf.transform.translation.z = 0.0
        odom_tf.transform.rotation.x = q[0]
        odom_tf.transform.rotation.y = q[1]
        odom_tf.transform.rotation.z = q[2]
        odom_tf.transform.rotation.w = q[3]
        transform_broadcaster.sendTransform(odom_tf)
    '''

def publish_camera_images(front_camera_image, timestamp):
    publishers['pub_front_camera'].publish(create_image_msg(front_camera_image, "front_camera", timestamp))


def publish_lap_count_data(lap_count):
    publishers['pub_lap_count'].publish(create_int_msg(msg_int32, lap_count))


def publish_lap_time_data(lap_time):
    publishers['pub_lap_time'].publish(create_float_msg(msg_float32, lap_time))


def publish_last_lap_time_data(last_lap_time):
    publishers['pub_last_lap_time'].publish(create_float_msg(msg_float32, last_lap_time))


def publish_best_lap_time_data(best_lap_time):
    publishers['pub_best_lap_time'].publish(create_float_msg(msg_float32, best_lap_time))


def publish_collision_count_data(collision_count):
    publishers['pub_collision_count'].publish(create_int_msg(msg_int32, collision_count))



#########################################################
# ROS 1 SUBSCRIBER CALLBACKS
#########################################################

# VEHICLE DATA SUBSCRIBER CALLBACKS


def callback_throttle_command(throttle_command_msg):
    global autodrive
    autodrive.throttle_command = float(np.round(throttle_command_msg.data, 3))


def callback_steering_command(steering_command_msg):
    global autodrive
    autodrive.steering_command = float(np.round(steering_command_msg.data, 3))


def callback_reset_command(reset_command_msg):
    global autodrive
    autodrive.reset_command = reset_command_msg.data

def drive_callback(msg):
    global autodrive
    speed = msg.drive.speed
    steering_angle = msg.drive.steering_angle

    if abs(speed) > autodrive.max_speed:
        rospy.logwarn("Speed command exceeds maximum speed. Clipping to %.4f", autodrive.max_speed)
        speed = math.copysign(autodrive.max_speed, speed)

    if abs(steering_angle) > autodrive.max_steering_angle:
        rospy.logwarn(
            "Steering command exceeds maximum steering angle. Clipping to %.4f",
            autodrive.max_steering_angle,
        )
        steering_angle = math.copysign(autodrive.max_steering_angle, steering_angle)

    throttle_msg = Float32()
    throttle_msg.data = speed / autodrive.max_speed
    callback_throttle_command(throttle_msg)

    steering_msg = Float32()
    steering_msg.data = steering_angle / autodrive.max_steering_angle
    callback_steering_command(steering_msg)

#########################################################
# WEBSOCKET SERVER INFRASTRUCTURE
#########################################################

# Initialize the server
sio = socketio.Server(async_mode='gevent')


# Registering "connect" event handler for the server
@sio.on('connect')
def connect(sid, environ):
    global initpose_sent, initpose_reason
    initpose_sent = False
    initpose_reason = "connect"
    rospy.loginfo("Connected!")

@sio.on('disconnect')
def disconnect(sid):
    global initpose_sent, initpose_reason
    initpose_sent = False
    initpose_reason = "reconnect"
    rospy.loginfo("Disconnected!")


# Registering "Bridge" event handler for the server
@sio.on('Bridge')
def bridge(sid, data):
    # Global declarations
    global autodrive, autodrive_bridge, cv_bridge, publishers, transform_broadcaster
    global real_mode, initpose_sent, initpose_reason
    #import pprint
    #pprint.pprint(data)

    # Wait for data to become available
    if data:
        ########################################################################
        # INCOMMING DATA
        ########################################################################
        timestamp = None
        timestamp = rospy.Time.now()
        # Actuator feedbacks
        autodrive.throttle = float(data["V1 Throttle"])
        autodrive.steering = float(data["V1 Steering"])
        # Speed
        #autodrive.speed = float(data["V1 Speed"])
        # Wheel encoders
        autodrive.encoder_angles = np.fromstring(data["V1 Encoder Angles"], dtype=float, sep=' ')
        # IPS
        autodrive.position = np.fromstring(data["V1 Position"], dtype=float, sep=' ')
        # IMU
        autodrive.orientation_quaternion = np.fromstring(data["V1 Orientation Quaternion"], dtype=float, sep=' ')
        autodrive.orientation_euler_angles = parse_v1_orientation_euler_angles(data)
        autodrive.angular_velocity = np.fromstring(data["V1 Angular Velocity"], dtype=float, sep=' ')
        autodrive.linear_velocity = np.fromstring(data["V1 Linear Velocity"], dtype=float, sep=' ')

        autodrive.linear_acceleration = np.fromstring(data["V1 Linear Acceleration"], dtype=float, sep=' ')
        # LIDAR
        autodrive.lidar_scan_rate = float(data["V1 LIDAR Scan Rate"])
        autodrive.lidar_range_array = np.fromstring(gzip.decompress(base64.b64decode(data["V1 LIDAR Range Array"])).decode('utf-8'), sep='\n')
        # Cameras
        autodrive.front_camera_image = np.asarray(Image.open(BytesIO(base64.b64decode(data["V1 Front Camera Image"]))))
        # Lap data
        autodrive.lap_count = int(float(data["V1 Lap Count"]))
        autodrive.lap_time = float(data["V1 Lap Time"])
        autodrive.last_lap_time = float(data["V1 Last Lap Time"])
        autodrive.best_lap_time = float(data["V1 Best Lap Time"])
        autodrive.collision_count = int(float(data["V1 Collisions"]))
        collision_detected = (
            autodrive.previous_collision_count >= 0 and
            autodrive.collision_count > autodrive.previous_collision_count
        )
        autodrive.previous_collision_count = autodrive.collision_count

        # Actuator feedbacks
        publish_actuator_feedbacks(autodrive.throttle, autodrive.steering)
        # Speed
        #publish_speed_data(autodrive.speed)
        # Wheel encoders
        publish_encoder_data(autodrive.encoder_angles)
        # IPS
        publish_ips_data(autodrive.position)
        if not initpose_sent:
            publish_initialpose(
                autodrive.position, autodrive.orientation_quaternion,
                autodrive.orientation_euler_angles,
                initpose_reason, timestamp
            )
        if collision_detected:
            autodrive.throttle_command = 0.0
            autodrive.steering_command = 0.0
            if reinit_on_collision:
                initpose_sent = False
                initpose_reason = "collision_count"
        # IMU
        publish_imu_data(autodrive.orientation_quaternion, autodrive.angular_velocity, autodrive.linear_acceleration, timestamp)

        # LIDAR
        publish_lidar_scan(autodrive.lidar_scan_rate, autodrive.lidar_range_array, autodrive.lidar_intensity_array, timestamp)
        if real_mode:
            publish_scan(autodrive.lidar_scan_rate, autodrive.lidar_range_array, autodrive.lidar_intensity_array, timestamp)

        # Coordinate transforms
        broadcast_transforms(transform_broadcaster, autodrive, timestamp)

        # Cameras
        publish_camera_images(autodrive.front_camera_image, timestamp)
        # Lap data
        publish_lap_count_data(autodrive.lap_count)
        publish_lap_time_data(autodrive.lap_time)
        publish_last_lap_time_data(autodrive.last_lap_time)
        publish_best_lap_time_data(autodrive.best_lap_time)
        publish_collision_count_data(autodrive.collision_count)

        if real_mode:
            publish_raw_imu_data(autodrive.orientation_quaternion, autodrive.angular_velocity, autodrive.linear_acceleration, timestamp)
            #wheel_encoder_callback()
            publish_odometery_data(autodrive.position, autodrive.orientation_quaternion, autodrive.linear_velocity, autodrive.angular_velocity, timestamp)


        ########################################################################
        # OUTGOING DATA
        ########################################################################
        # Vehicle and simulation commands
        pace_bridge_emit()
        sio.emit('Bridge', data={'V1 Throttle': str(autodrive.throttle_command),
                                 'V1 Steering': str(autodrive.steering_command),
                                 'V1 Reset': str(autodrive.reset_command)
                                 }
                 )

def publish_odometery_data(position, orientation_quaternion, linear_velocity, angular_velocity, timestamp):
    publishers['pub_odom'].publish(create_odom_msg(msg_odom, position, orientation_quaternion, linear_velocity, angular_velocity, timestamp))

def create_odom_msg(odom, position, orientation_quaternion, linear_velocity, angular_velocity, timestamp = None):
    odom.header = Header()
    if not timestamp:
        timestamp = rospy.Time.now()
    odom.header.stamp = timestamp
    odom.header.frame_id = "odom"
    odom.child_frame_id = "base_link"


    odom.pose.pose.position.x = position[0]
    odom.pose.pose.position.y = position[1]
    odom.pose.pose.position.z = position[2]
    odom.pose.pose.orientation.x = orientation_quaternion[0]
    odom.pose.pose.orientation.y = orientation_quaternion[1]
    odom.pose.pose.orientation.z = orientation_quaternion[2]
    odom.pose.pose.orientation.w = orientation_quaternion[3]
    odom.pose.covariance = [0.0025, 0.0, 0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0025, 0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0025, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0, 0.0025, 0.0, 0.0,
                            0.0, 0.0, 0.0, 0.0, 0.0025, 0.0,
                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0025]
    odom.twist.twist.linear.x = linear_velocity[0]
    odom.twist.twist.linear.y = linear_velocity[1]
    odom.twist.twist.linear.z = linear_velocity[2]
    odom.twist.twist.angular.x = angular_velocity[0]
    odom.twist.twist.angular.y = angular_velocity[1]
    odom.twist.twist.angular.z = angular_velocity[2]
    odom.twist.covariance = [0.0025, 0.0, 0.0, 0.0, 0.0, 0.0,
                             0.0, 0.0025, 0.0, 0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0025, 0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0, 0.0025, 0.0, 0.0,
                             0.0, 0.0, 0.0, 0.0, 0.0025, 0.0,
                             0.0, 0.0, 0.0, 0.0, 0.0, 0.0025]
    return odom
'''
def normalize_angle(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle    

def wheel_encoder_callback():
    global autodrive, publishers
    # Choose the latest timestamp
    autodrive.current_time = rospy.Time.now()

    autodrive.last_steering_angle = autodrive.steering
    #autodrive.first_steering_msg = True

    autodrive.current_left_position = autodrive.encoder_angles[0]
    autodrive.current_right_position = autodrive.encoder_angles[1]

    delta_time = (autodrive.current_time - autodrive.prev_time).to_sec()
    if delta_time <= 0.0:
        rospy.logwarn("Non-positive delta time. Skipping odometry update.")
        return

    delta_left = autodrive.current_left_position - autodrive.prev_left_position
    delta_right = autodrive.current_right_position - autodrive.prev_right_position

    autodrive.prev_left_position = autodrive.current_left_position
    autodrive.prev_right_position = autodrive.current_right_position
    autodrive.prev_time = autodrive.current_time

    speed_left = (
        (2.0 * math.pi * autodrive.wheel_radius) * (delta_left / autodrive.counts_per_rev) / delta_time
    )
    speed_right = (
        (2.0 * math.pi * autodrive.wheel_radius) * (delta_right / autodrive.counts_per_rev) / delta_time
    )

    estimated_speed = (speed_left + speed_right) / 2.0
    if abs(autodrive.est_speed - estimated_speed) < autodrive.max_acceleration:
        autodrive.est_speed = estimated_speed

    print(delta_time, delta_left, delta_right, speed_left, speed_right,  estimated_speed)
    angular_speed = 0.0
    if abs(autodrive.last_steering_angle) > 1e-3:
        angular_speed = autodrive.est_speed * math.tan(autodrive.last_steering_angle) / autodrive.wheel_base

    autodrive.x += autodrive.est_speed * math.cos(autodrive.theta) * delta_time
    autodrive.y += autodrive.est_speed * math.sin(autodrive.theta) * delta_time
    autodrive.theta += angular_speed * delta_time
    autodrive.theta = normalize_angle(autodrive.theta)

    publish_current_pose()
'''

#########################################################
# AUTODRIVE ROS 1 BRIDGE INFRASTRUCTURE
#########################################################


def main():
    # Global declarations
    global autodrive, autodrive_bridge, cv_bridge, publishers, transform_broadcaster, static_broadcaster, real_mode
    global initpose_map_tx, initpose_map_ty, initpose_yaw_offset
    global reinit_on_collision, bridge_rate_hz, bridge_period_sec, next_bridge_emit_time

    parser = argparse.ArgumentParser(description='AutoDRIVE ROS 1 bridge')
    parser.add_argument('--real', action='store_true', help='Publish additional real-robot topics (/scan, /vesc/odom, /vesc/sensors/imu/raw)')
    parser.add_argument('--bridge-rate-hz', dest='bridge_rate_hz', type=float, default=None,
                        help='Maximum Socket.IO Bridge response rate. Use <= 0 to disable pacing.')
    parser.add_argument('--initpose-map-tx', '--initpose-map-from-ips-x',
                        dest='initpose_map_tx', type=float, default=None,
                        help='X translation from AutoDRIVE V1/world frame to Cartographer map frame')
    parser.add_argument('--initpose-map-ty', '--initpose-map-from-ips-y',
                        dest='initpose_map_ty', type=float, default=None,
                        help='Y translation from AutoDRIVE V1/world frame to Cartographer map frame')
    parser.add_argument('--initpose-yaw-offset', '--initpose-map-from-ips-yaw',
                        dest='initpose_yaw_offset', type=float, default=None,
                        help='Yaw rotation in radians from AutoDRIVE V1/world frame to Cartographer map frame')
    args, _ = parser.parse_known_args()
    real_mode = args.real

    # ROS 1 infrastructure
    rospy.init_node('autodrive_bridge_ros1')  # Initialize ROS 1 node
    default_initpose_map_tx = (
        args.initpose_map_tx if args.initpose_map_tx is not None else initpose_map_tx
    )
    default_initpose_map_ty = (
        args.initpose_map_ty if args.initpose_map_ty is not None else initpose_map_ty
    )
    default_initpose_yaw_offset = (
        args.initpose_yaw_offset if args.initpose_yaw_offset is not None else initpose_yaw_offset
    )
    initpose_map_tx = rospy.get_param(
        '~initpose_map_tx',
        rospy.get_param('~initpose_map_from_ips_x', default_initpose_map_tx)
    )
    initpose_map_ty = rospy.get_param(
        '~initpose_map_ty',
        rospy.get_param('~initpose_map_from_ips_y', default_initpose_map_ty)
    )
    initpose_yaw_offset = rospy.get_param(
        '~initpose_yaw_offset',
        rospy.get_param('~initpose_map_from_ips_yaw', default_initpose_yaw_offset)
    )
    reinit_on_collision = rospy.get_param('~reinit_on_collision', True)
    default_bridge_rate_hz = (
        args.bridge_rate_hz if args.bridge_rate_hz is not None else bridge_rate_hz
    )
    bridge_rate_hz = float(rospy.get_param('~bridge_rate_hz', default_bridge_rate_hz))
    if bridge_rate_hz > 0.0:
        bridge_period_sec = 1.0 / bridge_rate_hz
        next_bridge_emit_time = monotonic()
    else:
        bridge_period_sec = 0.0
    rospy.loginfo("AutoDRIVE bridge emit pacing: %.3f Hz", bridge_rate_hz)
    rospy.loginfo(
        "AutoDRIVE /initialpose transform: map_x=-v1_y+%.3f map_y=v1_x+%.3f yaw=v1_yaw+%.3f rad",
        initpose_map_tx, initpose_map_ty, initpose_yaw_offset
    )

    autodrive_bridge = rospy  # Retain handle for symmetry with ROS 2 version
    cv_bridge = CvBridge()  # ROS bridge object for OpenCV library to handle image data
    transform_broadcaster = tf2_ros.TransformBroadcaster()  # Initialize transform broadcaster
    static_broadcaster = tf2_ros.StaticTransformBroadcaster()  # One-time static transforms
    publishers = {e['name']: rospy.Publisher(e['topic'], e['type'], queue_size=1)
                  for e in config.pub_sub_dict.publishers}  # Publishers
    publishers['pub_initpose'] = rospy.Publisher('/initialpose', PoseWithCovarianceStamped, queue_size=1)
    if real_mode:
        autodrive.init_ackermann_to_autodrive_params()
        #autodrive.init_wheel_odometry()
        publishers['pub_odom'] = rospy.Publisher('/vesc/odom', Odometry, queue_size=10)

        publishers['pub_scan'] = rospy.Publisher('/scan', LaserScan, queue_size=10)
        publishers['pub_raw_imu'] = rospy.Publisher('/vesc/sensors/imu/raw', Imu, queue_size=1)
        #publishers['pub_ips_pose'] = rospy.Publisher('/ips/pose', PoseWithCovarianceStamped, queue_size=1)

    callbacks = {
        '/autodrive/roboracer_1/throttle_command': callback_throttle_command,
        '/autodrive/roboracer_1/steering_command': callback_steering_command,
        '/autodrive/reset_command': callback_reset_command
    }  # Subscriber callback functions
    [rospy.Subscriber(e['topic'], e['type'], callbacks[e['topic']], queue_size=1) for e in config.pub_sub_dict.subscribers]  # Subscribers

    if real_mode:
        rospy.Subscriber("/vesc/low_level/ackermann_cmd_mux/output", AckermannDriveStamped, drive_callback, queue_size=1)

    # Spin ROS callbacks in a background thread so the gevent server can own the main thread
    process = Thread(target=rospy.spin, daemon=True)
    process.start()

    app = socketio.WSGIApp(sio)  # Create socketio WSGI application
    pywsgi.WSGIServer(('', 4567), app, handler_class=WebSocketHandler).serve_forever()  # Deploy as a gevent WSGI server

    # Cleanup
    rospy.signal_shutdown('Shutting down autodrive_bridge_ros1')


################################################################################

if __name__ == '__main__':
    main()  # Call main function of AutoDRIVE ROS 1 bridge
