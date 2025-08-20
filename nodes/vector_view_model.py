import os
import datetime
import time
import json
from pathlib import Path
import csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import math

import rospy
import tf2_ros

from nav_msgs.msg import Odometry
from sensor_msgs.msg import CameraInfo, CompressedImage
from geometry_msgs.msg import PoseStamped, Pose2D, Twist
from tf2_geometry_msgs import PointStamped
from image_geometry.cameramodels import PinholeCameraModel
from tf.transformations import quaternion_from_euler
from reset_robot import reset_robot#, pub2

import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.mixed_precision import loss_scale_optimizer
import utilities

import cv2
import numpy as np
from cv_bridge import CvBridge
### TensorFlow and GPU configuration
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # 0: all messages, 1: no INFO, 2: no INFO/WARNING, 3: no INFO/WARNING/ERROR
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Enable XLA JIT compilation
tf.config.optimizer.set_jit(True)

gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     for gpu in gpus:
#         tf.config.experimental.set_memory_growth(gpu, True)

# GPU memory growth and limit
# gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3800)]
        )
    except RuntimeError as e:
        print(e)

# Optional: set mixed precision policy
# mixed_precision.set_global_policy('mixed_float16')

colormap = np.asarray([[ 0,  0, 64], [64,  0, 0], [ 0, 64,  0], [0, 0, 200], [200,  0, 0], [0, 200, 0], [255,255,255]])

IMAGE_SIZE = 256
BATCH_SIZE  = 4
NUM_CLASSES = 2
L_RATE   = 0.00001
RATIO = 1.76666666666666666

MOBILE = False
# MODEL_PATH = "../models/256px_100ep_erik_logic_2.hdf5"
MODEL_PATH = "../models/256px_500ep_erik_supplemented_2.hdf5"

# ---- Model Architecture Matching Training ----
def convolution_block(
    block_input,
    num_filters=256,
    kernel_size=3,
    dilation_rate=1,
    padding="same",
    use_bias=False,
):
    x = layers.Conv2D(
        num_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="same",
        use_bias=use_bias,
        kernel_initializer=keras.initializers.HeNormal(),
    )(block_input)
    x = layers.BatchNormalization()(x)
    return tf.nn.relu(x)

def DilatedSpatialPyramidPooling(dspp_input):
    dims = dspp_input.shape
    x = layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
    x = convolution_block(x, kernel_size=1, use_bias=True)
    out_pool = layers.UpSampling2D(
        size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]), interpolation="bilinear",
    )(x)

    out_1 = convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
    out_6 = convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
    out_12 = convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
    out_18 = convolution_block(dspp_input, kernel_size=3, dilation_rate=18)

    x = layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
    output = convolution_block(x, kernel_size=1)
    return output

def get_x0(x):
    shape = x.shape

    flatten = layers.Flatten(input_shape=shape)(x)
    hidden_layer = layers.Dense(25, activation='relu', use_bias=True)(flatten)
    hidden_layer = layers.Dense(20, activation='relu', use_bias=True)(hidden_layer)
    output_x0 = layers.Dense(1, activation='sigmoid', use_bias=True, name="x0")(hidden_layer)

    return output_x0

def get_x1(x):
    shape = x.shape

    flatten = layers.Flatten(input_shape=shape)(x)
    hidden_layer = layers.Dense(25, activation='relu', use_bias=True)(flatten)
    hidden_layer = layers.Dense(20, activation='relu', use_bias=True)(hidden_layer)
    output_x1 = layers.Dense(1, activation='sigmoid', use_bias=True, name="x1")(hidden_layer)

    return output_x1

def DeeplabV3Plus(image_size, num_classes):
    model_input = keras.Input(shape=(image_size, image_size, 3))
    resnet50 = keras.applications.ResNet50(
        weights="imagenet", include_top=False, input_tensor=model_input
    )
    x = resnet50.get_layer("conv4_block6_2_relu").output

    x1_out = get_x1(x)
    x0_out = get_x0(x)

    x = DilatedSpatialPyramidPooling(x)

    input_a = layers.UpSampling2D(
        size=(image_size // 4 // x.shape[1], image_size // 4 // x.shape[2]),
        interpolation="bilinear",
    )(x)
    input_b = resnet50.get_layer("conv2_block3_2_relu").output
    input_b = convolution_block(input_b, num_filters=48, kernel_size=1)

    x = layers.Concatenate(axis=-1)([input_a, input_b])
    x = convolution_block(x)
    x = convolution_block(x)

    x = layers.UpSampling2D(
        size=(image_size // x.shape[1], image_size // x.shape[2]),
        interpolation="bilinear",
    )(x)

    model_output = layers.Conv2D(num_classes, kernel_size=(1, 1), padding="same")(x)
    model_output = tf.keras.layers.Activation("linear", dtype="float32", name="mask")(model_output)
    
    return keras.Model(inputs=model_input, outputs=[model_output, x0_out, x1_out])

losses = {
    "mask": keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    "x0": keras.losses.MeanAbsoluteError(),
    "x1": keras.losses.MeanAbsoluteError()
}

loss_weights = {
    "mask": 1.0,
    "x0": 5.0,
    "x1": 5.0
}

opt = keras.optimizers.Adam(learning_rate=L_RATE)
opt = loss_scale_optimizer.LossScaleOptimizerV3(opt)

model = DeeplabV3Plus(image_size=IMAGE_SIZE, num_classes=NUM_CLASSES)
model.compile(
optimizer=opt,
loss=losses,
loss_weights=loss_weights,
metrics={'mask': ['accuracy'], 
            'x0': ['mae'],
            'x1': ['mae']})
print('ResNet model created')

model = DeeplabV3Plus(image_size=IMAGE_SIZE, num_classes=2)
model.load_weights(MODEL_PATH)
print('weights loaded')

def border_to_point(image_width, image_height, val1, val2):
    w = image_width
    h = image_height

    if val2<h/2:
        x1 = 0
        y1 = h/2-val2
    elif val2>h/2+w:
        x1 = w
        y1 = val2-h/2-w
    else:
        x1 = val2-h/2
        y1 = 0
    
    if val1<h/2:
        x2 = 0
        y2 = val1 + h/2
    elif val1>h/2+w:
        x2 = w
        y2 = h+w-(val1 - h/2)
    else:
        x2 = val1-h/2
        y2 = h
    
    return (x1, y1), (x2, y2)

# def predict_from_image(image):
#     img = image / 255.0 # Normalize the image
#     input_tensor = tf.convert_to_tensor(img, dtype=tf.float32) # tensor for prediction
#     input_tensor = tf.expand_dims(input_tensor, axis=0)
#     _, val1, val2 = model(input_tensor, training=False)

#     val1 = float(np.clip(val1.numpy(), 1e-6, 1 - 1e-6))
#     val2 = float(np.clip(val2.numpy(), 1e-6, 1 - 1e-6))

#     print("Start", val1, val2)

#     val1 = int(np.round(- (IMAGE_SIZE / 3.5) * np.log((1 - val1) / val1) + 1.0 * IMAGE_SIZE))
#     val2 = int(np.round(- (IMAGE_SIZE / 3.5) * np.log((1 - val2) / val2) + 1.0 * IMAGE_SIZE))
#     print("End", val1, val2)
#     (x1, y1), (x2, y2) = border_to_point(IMAGE_SIZE, IMAGE_SIZE, val1, val2)
#     # center_x = (x1 + x2) / 2
#     # center_y = (y1 + y2) / 2
#     # theta = np.arctan2(x2 - x1, y2 - y1)
#     # return center_x, center_y, np.degrees(theta)
#     return (x1, y1), (x2, y2)
def predict_from_image(image):
    img = image / 255.0
    input_tensor = tf.convert_to_tensor(img, dtype=tf.float32)
    input_tensor = tf.expand_dims(input_tensor, axis=0)
    val1, val2 = predict_from_image_optimized(input_tensor)
    
    val1 = float(np.clip(val1.numpy(), 1e-6, 1 - 1e-6))
    val2 = float(np.clip(val2.numpy(), 1e-6, 1 - 1e-6))
    
    val1 = int(np.round(- (IMAGE_SIZE / 3.5) * np.log((1 - val1) / val1) + 1.0 * IMAGE_SIZE))
    val2 = int(np.round(- (IMAGE_SIZE / 3.5) * np.log((1 - val2) / val2) + 1.0 * IMAGE_SIZE))
    
    return border_to_point(IMAGE_SIZE, IMAGE_SIZE, val1, val2)

bridge = CvBridge()

# Global variable to store the latest message
latest_pose = None
cv_image = None  
model_pose_ = None

def callback(msg):
    """Callback for line features"""
    global latest_pose
    latest_pose = msg
def robotPoseCallback(msg):
    global model_pose_
    model_pose_ = msg.pose
    model_pose_.pose.position.z = model_pose_.pose.position.x * 0.08 + 2.75 # world slope lr
def norm(x, max):
    return 2 * ((x) / (max)) - 1

def shift(x,max):
    if x > max:
        x = np.sign(x) * max
    elif x < 0:
        x = 0
    return (2*x - max)/2

def lim_w(w, max):
    if np.abs(w) > max:
        return np.sign(w)*max
    else:
        return w

def point_conversion(point,cur_frame,dest_frame):
    cur_point = PointStamped()
    dest_point = PointStamped()
    cur_point.header.stamp = rospy.Time.now()
    cur_point.header.frame_id = cur_frame
    cur_point.point.x = point[0]
    cur_point.point.y = point[1]
    cur_point.point.z = point[2]

    try:
        # dest_point = mytfBuffer.transform(cur_point, dest_frame, rospy.Duration(0.2))
        print("lel")
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as err:
        print (err)
        pass

    return (dest_point.point.x,dest_point.point.y,dest_point.point.z)

def compressed_img_callback(msg):
    global cv_image
    np_arr = np.frombuffer(msg.data, np.uint8)
    cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


@tf.function(jit_compile=True)
def infer_tensor(inputs):
    mask, val1, val2 = model(inputs, training=False)
    return tf.squeeze(mask), tf.squeeze(val1), tf.squeeze(val2)

@tf.function(jit_compile=True)
def predict_from_image_optimized(image_tensor):
    _, val1, val2 = model(image_tensor, training=False)
    return val1, val2

def infer(model, image_tensor):
    inputs = tf.expand_dims(image_tensor, axis=0)
    # print(f'\n\n THIS IS THE SHAPE OF CALAAR: {np.shape(inputs)}\n\n\n')
    return infer_tensor(image_tensor)

def decode_segmentation_masks(mask, colormap):
    rgb = colormap[mask]
    return rgb.astype(np.uint8)
def get_overlay(image, colored_mask):
    image = tf.keras.preprocessing.image.array_to_img(image)
    colored_mask = tf.keras.preprocessing.image.array_to_img(colored_mask)
    
    image = np.array(image).astype(np.uint8)
    colored_mask = np.array(colored_mask).astype(np.uint8)

    overlay = cv2.addWeighted(image, 0.5, colored_mask, 0.5, 0)
    return overlay

def angle_with_neg_y(p1, p2, signed=True):
    dx, dy = p2[0]-p1[0], p2[1]-p1[1]
    ang = math.atan2(dy, dx) + math.pi/2
    if ang > math.pi:
        ang -= 2*math.pi
    return ang

def enhance(img, clip_limit=1.2, h_factor=1.0, s_factor=1.15, v_factor=1.05):
    # Convert image from BGR to HSV directly, skip LAB conversion
    hsv_enhanced = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Apply CLAHE directly to V channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(3, 3))
    hsv_enhanced[:,:,2] = clahe.apply(hsv_enhanced[:,:,2])
    
    # Adjust channels using vectorized operations
    hsv_enhanced = hsv_enhanced.astype(np.float32)
    hsv_enhanced[:,:,1] *= s_factor
    hsv_enhanced[:,:,2] *= v_factor
    hsv_enhanced = np.clip(hsv_enhanced, [0,0,0], [179,255,255]).astype(np.uint8)
    
    return cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)

# def enhance(img, clip_limit=1.2, h_factor=1.0, s_factor=1.15, v_factor=1.05):
#     r_factor = 1 + 0.08
#     g_factor = 1 + 0.09
#     b_factor = 1 + 0.07

#     # Convert image from BGR to LAB color space
#     lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

#     # Split LAB channels
#     l, a, b = cv2.split(lab)

#     # Applying CLAHE to L-channel
#     clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(3, 3))
#     cl = clahe.apply(l)

#     # Merge enhanced L-channel with a and b channels
#     lab_enhanced = cv2.merge((cl, a, b))

#     # Convert enhanced LAB image to RGB color space
#     rgb_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)

#     # Convert RGB enhanced image to HSV color space
#     hsv_enhanced = cv2.cvtColor(rgb_enhanced, cv2.COLOR_RGB2HSV)
#     #hsv_enhanced = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#     # Adjust H, S, and V channels in-place
#     hsv_enhanced[:, :, 0] = np.asarray(np.clip(hsv_enhanced[:, :, 0] * h_factor, 0, 179), dtype=np.uint8)
#     hsv_enhanced[:, :, 1] = np.asarray(np.clip(hsv_enhanced[:, :, 1] * s_factor, 0, 255), dtype=np.uint8)
#     hsv_enhanced[:, :, 2] = np.asarray(np.clip(hsv_enhanced[:, :, 2] * v_factor, 0, 255), dtype=np.uint8)

#     # Convert HSV enhanced image back to RGB color space
#     rgb_plus = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2RGB)

#     r,g,b = cv2.split(rgb_plus)
#     r = np.asarray(np.clip(r * r_factor, 0, 255), dtype=np.uint8)
#     g = np.asarray(np.clip(g * g_factor, 0, 255), dtype=np.uint8)
#     b = np.asarray(np.clip(b * b_factor, 0, 255), dtype=np.uint8)
#     rgb_plus2 = cv2.merge((r, g, b))

#     # Convert RGB enhanced image to BGR color space
#     color_plus = cv2.cvtColor(rgb_plus2, cv2.COLOR_RGB2BGR)

    # return color_plus

def sq_to_rec(x,y, w,h):
    x_rec = int(x * w / IMAGE_SIZE)
    y_rec = int(y * h /IMAGE_SIZE)
    return (x_rec, y_rec)


def main():
    global cv_image
    pub = rospy.Publisher('/twist_mux/cmd_vel', Twist, queue_size=10)



    v_d = 1.0

    controller = utilities.RobustController(1, v_d, dt=1/30, robust=False)
    control_output = Twist()
    control_output.linear.x = v_d
    
    # Pre-allocate arrays and objects
    img = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)

    # Create the OpenCV window before the loop
    # Set size of window to match IMAGE_SIZE

    cv2.namedWindow("Camera Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Camera Image", 848, 480)


    # Setup the data collection 
    name = "robust" if controller.robust else "proportional"
    dtdt = datetime.datetime.now()
    day_time = dtdt.strftime("%d%m%H%M")
    # file_name = f"../data/final/slope_lr_smooth_{name}_{day_time}.csv"
    start_pos = "right"
    file_name = f"ML_pitch_bumpy_{start_pos}_{name}_{day_time}.csv"
    folder = "prop_vs_robust" #"true_vs_ML" 
    full_path = f"../DATAFINAL/{folder}/{file_name}"
    try:
        k = open(full_path, "x")
        k.close()
    except FileExistsError:
        print(f"File {full_path} already exists, overwriting it.")
    
    f = open(full_path, "w", newline='')
    writer = csv.writer(f)
    writer.writerow(["xv", "yv", "thetav", "x", "y", "z", "r", "p", "y", "v", "w"])


    while not rospy.is_shutdown():
        if cv_image is not None and model_pose_ is not None:
            # Make a copy for visualization to avoid modifying the original image
            vis_image = cv_image.copy()
            # Resize first to reduce processing time
            img = cv2.resize(vis_image, (IMAGE_SIZE, IMAGE_SIZE))
            img = enhance(img)
            
            pt1, pt2 = predict_from_image(img)
            theta_pred = angle_with_neg_y(pt2, pt1)
            
            # Combine calculations
            x, y = pt2
            visual_theta = theta_pred - np.pi/2
            x_rec, y_rec = sq_to_rec(x, y, 848, 480)
            
            # Update control output
            omega = controller.control_step(norm(x_rec, 848),norm(y_rec, 480), theta_pred)
            omega = lim_w(omega, np.pi/8)
            
            control_output.angular.z = omega

            pub.publish(control_output)
            
            # Visualization (consider reducing frequency if needed)
            if cv2.getWindowProperty("Camera Image", cv2.WND_PROP_VISIBLE) >= 0:
                cv2.circle(cv_image, (x_rec, y_rec), 7, (0, 255, 0), -1)
                cv2.circle(cv_image, (x_rec, y_rec), 3, (0, 0, 0), -1)
                end_point = (int(x_rec + 70 * np.cos(visual_theta)), 
                           int(y_rec + 70 * np.sin(visual_theta)))
                cv2.arrowedLine(cv_image, (x_rec, y_rec), end_point, (0, 255, 0), 7, tipLength=0.2)
                cv2.arrowedLine(cv_image, (x_rec, y_rec), end_point, (0, 0, 0), 3, tipLength=0.2)
                cv2.imshow("Camera Image", cv_image)
                cv2.waitKey(1)
            # Get robot pose for evaluation 
            x_robot = model_pose_.pose.position.x
            y_robot = model_pose_.pose.position.y
            z_robot = model_pose_.pose.position.z
            orientation = model_pose_.pose.orientation
            r_robot = orientation.x
            p_robot = orientation.y
            yaw_robot = orientation.z
            writer.writerow([x_rec,y_rec,theta_pred, x_robot, y_robot, z_robot, r_robot, p_robot, yaw_robot, controller.v_d,omega])

        rate.sleep()
    f.close()
if __name__ == '__main__':
    rospy.init_node('true_line_printer', anonymous=True)
    rospy.Subscriber('/r200/camera/color/image_raw/compressed' , CompressedImage , compressed_img_callback)
    rospy.Subscriber('/odometry/base_raw', Odometry, robotPoseCallback)
    rate = rospy.Rate(30)

    rospy.sleep(1)

    # Start positions flat world
    # x,y,z,r,p,yaw = (0.967481, -20.989647, 0.021569, 0.0, 0.0, 1,588244) # straight on
    # x,y,z,r,p,yaw = (1.482014, -20.851523, 0.021569, 0.0, 0.0, 2.337844) # from right
    # x,y,z,r,p,yaw = (0.416909, -20.792364, 0.021569, 0.0, 0.0, 0.817370) # from left
    # x,y,z,r,p,yaw = (0.000664, -20.573657, 0.021569, 0.0, 0, 1.188267) # same as in trueline worked!

    # Start positions slope left right
    # x,y,z,r,p,yaw = (1.438389, -20.826057, 2.881630, -0.052244, 0.044957, 2.280777) # from right
    # x,y,z,r,p,yaw = (1.005811, -20.772925, 2.851775, -0.068896, 0.001340, 1.590212) # straight
    # x,y,z,r,p,yaw = (0.267664, -20.869485, 2.793623, -0.058682, -0.053494, 0.775467) # from left, this failed for all controllers, too far :D 
    # x,y,z,r,p,yaw = (0.267664, -20.869485, 2.793623, -0.058682, -0.053494, 1.005467) # Testing new left, this was fine

    # Start positions slope back front
    # x,y,z,r,p,yaw = (-1.849903,-17.016098, 2.633105, -0.044410, -0.052710, 0.700699) # From right - old
    x,y,z,r,p,yaw = (-2.144370,-17.465810, 2.666442, -0.015858, -0.073668, 0.743598) # From right - this one
    # x,y,z,r,p,yaw = (-1.928843,-16.434568, 2.627657, 0.000047, -0.068910, -0.000672) # Straight on
    # x,y,z,r,p,yaw = (-1.919183, -15.870317, 2.628320, 0.047657, -0.049789, -0.764152) # From left

    # Translate rpy into these quoterniions that are used for positionsing
    qx, qy, qz, qw = quaternion_from_euler(r,p,yaw)


    reset_robot(x=x, y=y, z=z, qx=qx, qy=qy, qz=qz, qw=qw) 

    main()

