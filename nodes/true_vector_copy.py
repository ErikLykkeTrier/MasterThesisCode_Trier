#!/usr/bin/env python
import rospy, tf2_ros
import cv2
import numpy as np
from utilities import model_up

from image_geometry.cameramodels import PinholeCameraModel
from tf2_geometry_msgs import PointStamped
from sensor_msgs.msg import CameraInfo, CompressedImage
from geometry_msgs.msg import PoseStamped, Pose2D

import os
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
#0 = all messages are logged (default behavior)
#1 = INFO messages are not printed
#2 = INFO and WARNING messages are not printed
#3 = INFO, WARNING, and ERROR messages are not printed

#GPU id to use
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time, json, cv2

from pathlib import Path

import tensorflow as tf
tf.config.optimizer.set_jit(True)

from tensorflow import keras

from keras import layers
from keras.callbacks import EarlyStopping, ModelCheckpoint, TerminateOnNaN
from keras.mixed_precision import loss_scale_optimizer

from keras import mixed_precision
# mixed_precision.set_global_policy('mixed_float16')

# Check if GPU is available and set memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

tf.config.experimental.set_virtual_device_configuration(
    gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3800)])

IMAGE_SIZE  = 256
MOBILE = False
net_id  = 'erik_logic_1'
NUM_CLASSES = 2 # as per training data
#color for mask viz
colormap = np.asarray([[ 0,  0, 64], [64,  0, 0], [ 0, 64,  0], [0, 0, 200], [200,  0, 0], [0, 200, 0], [255,255,255]])

robot_pose = PoseStamped()
camera = None
line = Pose2D()

# List over start and end points
pt_A_list = [( -0.68, -21, 2.72), (0.98, -21, 2.84), (2.57, -21, 2.96), (4.2, -21, 3)]
pt_B_list = [(-0.68, -9, 2.72), (0.98, -9, 2.84), (2.5, -9, 2.96), (4.137, -9, 3)]

model_ = "thorvald"
model_pose = PoseStamped()
model_pose.header.frame_id = "world"
global param_value
cv_image = None
def get_ros_parameter():
    try:
        param_value = rospy.get_param('/line_nr', 0) # 'default_value' is used if the parameter doesn't exist
        # rospy.loginfo(f"Value of '/line_nr': {param_value}")
        return param_value
    except KeyError:
        rospy.logwarn("Parameter '/line_nr' does not exist.")

def set_line_number(n):
    rospy.set_param('/line_nr', n)
    rospy.loginfo(f"Parameter '/line_nr' set to {n}")

def compressed_img_callback(msg):
    global cv_image
    np_arr = np.frombuffer(msg.data, np.uint8)
    cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

def get_caminfo():
    global camera
    cam_info = None
    cam_info = rospy.wait_for_message('/r200/camera/color/camera_info',CameraInfo)
    # cam_info = rospy.wait_for_message('/camera/camera/color/camera_info',CameraInfo)
    if cam_info is not None:
        camera = PinholeCameraModel()
        camera.fromCameraInfo(cam_info)

def point_conversion(point,cur_frame,dest_frame):
    cur_point = PointStamped()
    dest_point = PointStamped()
    cur_point.header.stamp = rospy.Time.now()
    cur_point.header.frame_id = cur_frame
    cur_point.point.x = point[0]
    cur_point.point.y = point[1]
    cur_point.point.z = point[2]

    try:
        dest_point = mytfBuffer.transform(cur_point, dest_frame, rospy.Duration(0.2))
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as err:
        print (err)
        pass

    return (dest_point.point.x,dest_point.point.y,dest_point.point.z)

def get_border_intersection(xA, yA, theta_rad, W, H):
    # Direction vector: theta is with respect to vertical (downward)
    dx = -np.sin(theta_rad)
    dy = -np.cos(theta_rad)

    intersections = []

    # Check both directions
    for sign in [+1, -1]:
        dx_dir = dx * sign
        dy_dir = dy * sign

        # Top edge (y = 0)
        if not np.isclose(dy_dir, 0):
            t = (0 - yA) / dy_dir
            x = xA + t * dx_dir
            if 0 <= x <= W:
                intersections.append(((x, 0), abs(t)))

        # Bottom edge (y = H)
        if not np.isclose(dy_dir, 0):
            t = (H - yA) / dy_dir
            x = xA + t * dx_dir
            if 0 <= x <= W:
                intersections.append(((x, H), abs(t)))

        # Left edge (x = 0)
        if not np.isclose(dx_dir, 0):
            t = (0 - xA) / dx_dir
            y = yA + t * dy_dir
            if 0 <= y <= H:
                intersections.append(((0, y), abs(t)))

        # Right edge (x = W)
        if not np.isclose(dx_dir, 0):
            t = (W - xA) / dx_dir
            y = yA + t * dy_dir
            if 0 <= y <= H:
                intersections.append(((W, y), abs(t)))

    if not intersections:
        return None

    # Return the intersection with the smallest distance from A
    (x_int, y_int), _ = min(intersections, key=lambda tup: tup[1])
    return int(x_int), int(y_int)

def draw_arrow(image, pt_A, pt_B, reference_point):
    A = np.array(pt_A)
    B = np.array(pt_B)
    P = np.array(reference_point)

    AB = B - A
    AB_unit = AB / np.linalg.norm(AB)

    AP = P - A

    t = np.dot(AP, AB_unit)
    closest = A + t * AB_unit

    pt_start = closest - AB_unit/2
    pt_end = closest + AB_unit/2

    (x, y, z) = point_conversion(tuple(pt_start), 'world', 'color')
    xA_p, yA_p = camera.project3dToPixel((x, y, z))

    (x, y, z) = point_conversion(tuple(pt_end), 'world', 'color')
    xB_p, yB_p = camera.project3dToPixel((x, y, z))

    pt1 = np.array([xA_p, yA_p])
    pt2 = np.array([xB_p, yB_p])

    # Calculate the angle between vec and vertical, signed
    theta_rad = np.arctan2(pt2[0] - pt1[0], pt2[1] - pt1[1])

    #cv2.arrowedLine(image, tuple(pt1.astype(int)), tuple(pt2.astype(int)), (255, 0, 255), 2, tipLength=0.2)
    #cv2.circle(image, tuple(pt1.astype(int)), 5, (255, 0, 0), -1)  # Start point in blue
    #cv2.circle(image, tuple(pt2.astype(int)), 5, (0, 0, 255), -1)  # end point in red

    bp = get_border_intersection(xA_p, yA_p, theta_rad, image.shape[1], image.shape[0])

    if bp is not None:

        final_angle = 180-np.degrees(theta_rad)
        if final_angle > 180:
            final_angle -= 360

        #print("Border intersection point:", bp, "Angle with vertical (deg):", final_angle)

        #ep = (int(bp[0] + 60 * np.sin(theta_rad)), int(bp[1] + 60 * np.cos(theta_rad)))
        #cv2.arrowedLine(image, bp, ep, (0, 0, 0), 5, tipLength=0.15)
        #cv2.arrowedLine(image, bp, ep, (0, 255, 0), 3, tipLength=0.15)
        #cv2.circle(image, bp, 7, (0, 255, 0), -1)
        #cv2.circle(image, bp, 3, (0, 0, 0), -1)
    else:

        #print("No line found.")
        return -1000, -1000, -1000

    #cv2.arrowedLine(image, tuple(pt1.astype(int)), tuple(pt2.astype(int)), (255, 0, 255), 2, tipLength=0.2)
    #cv2.circle(image, tuple(pt1.astype(int)), 5, (255, 0, 0), -1)

    #cv2.imshow("Arrow AB", image)
    #cv2.waitKey(1)

    return bp[0], bp[1], final_angle

def main():
    point2d = Pose2D()
    point2d.x = 0.0
    point2d.y = 0.0
    point2d.theta = 0.0

    while not rospy.is_shutdown():
        
        # pt_A = ( -0.5, -20.8, 2.7) #point_conversion((0.8, 0, 0), 'base_link', 'world')
        # pt_B = (-0.5, 11.8, 2.7) #point_conversion((1.2, 0, 0), 'base_link', 'world')
        index = get_ros_parameter()
        if index % 2 == 0:
            pt_A = pt_A_list[index]
            pt_B = pt_B_list[index]
        else:
            pt_A = pt_B_list[index]
            pt_B = pt_A_list[index]

        reference_point = point_conversion((1.0, 0, 0), 'base_link', 'world')

        if cv_image is not None:

            image = cv_image.copy()
            x, y , theta = draw_arrow(image, pt_A, pt_B, reference_point)

            point2d.x = x
            point2d.y = y
            point2d.theta = theta
            
            pub.publish(point2d)

        rate.sleep()

if __name__ == '__main__':
    # rospy.init_node('line_setter_node', anonymous=True)
    # rospy.init_node('param_getter_node', anonymous=True)
    # set_line_number(0)
    model = model_up()
    rospy.init_node('true_line_node')
    get_caminfo()

    mytfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(mytfBuffer)
    
    pub = rospy.Publisher('/true_line', Pose2D, queue_size=10)
    rospy.Subscriber('/r200/camera/color/image_raw/compressed' , CompressedImage , compressed_img_callback)
    # rospy.Subscriber('/camera/camera/color/image_raw/compressed' , CompressedImage , compressed_img_callback)

    rate = rospy.Rate(50)
    main()