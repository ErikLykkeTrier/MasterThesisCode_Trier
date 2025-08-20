#!/usr/bin/env python
import rospy, tf2_ros
import cv2
import numpy as np

from image_geometry.cameramodels import PinholeCameraModel
# from tf2_geometry_msgs import PointStamped, TFMessage
from sensor_msgs.msg import CameraInfo, CompressedImage
from geometry_msgs.msg import PoseStamped, Pose2D
# from gazebo_msgs.msg import ModelStates, LinkStates
from tf.transformations import quaternion_matrix
from nav_msgs.msg import Odometry


robot_pose = PoseStamped()
camera = None
line = Pose2D()

# List over start(A) and end(B) points for each world
# For normal sloped sideways + bumpy
# pt_A_list = [( -0.68, -21, 2.72), (0.98, -21, 2.84), (2.57, -21, 2.96), (4.2, -21, 3)]
# pt_B_list = [(-0.68, -9, 2.72), (0.98, -9, 2.84), (2.5, -9, 2.96), (4.137, -9, 3)]
# For flat + bumpy
pt_A_list = [(-0.594422, -21, 0.021567), (0.967484, -21, 0.021567),(2.578926, -21, 0.021567),(4.217786, -21, 0.021567)]
pt_B_list = [(-0.594422, -8, 0.021567), (0.967484, -8, 0.021567),(2.578926, -8, 0.021567),(4.217786, -8, 0.021567)]

# pt_A_list = [(-0.594422, -21, 4.9), (0.967484, -21, 4.9),(2.578926, -21, 4.9),(4.217786, -21, 4.9)]
# pt_B_list = [(-0.594422, -8, 4.9), (0.967484, -8, 4.9),(2.578926, -8, 4.9),(4.217786, -8, 4.9)]

# For sloped front + bumpy
# pt_A_list = [(-3.012625, -14.848261, 2.552855), (-3.012625, 16.436688, 2.552855),(-3.012625, -18.037518, 2.552855), (-3.012625, -19.600686, 2.552855)]
# pt_B_list = [(12.263264, -14.848261, 3.628749), (12.263264, 16.436688, 3.628749),(12.263264, -18.037518, 3.628749), (12.263264, -19.600686, 3.628749)]


model_ = "thorvald"
# model_ = "plt_2"
model_pose_ = PoseStamped()
model_pose_.header.frame_id = "world"
model_idx = 0

global param_value
# cv_image = None
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

# Store static transforms at initialization
static_transforms = {}

def get_static_transform(cur_frame, dest_frame):
    key = (cur_frame, dest_frame)
    if key not in static_transforms:
        try:
            trans = mytfBuffer.lookup_transform(dest_frame, cur_frame, rospy.Time(0), rospy.Duration(1.0))
            translation = np.array([
                trans.transform.translation.x,
                trans.transform.translation.y,
                trans.transform.translation.z
            ])
            rotation = np.array([
                trans.transform.rotation.x,
                trans.transform.rotation.y,
                trans.transform.rotation.z,
                trans.transform.rotation.w
            ])
            matrix = quaternion_matrix(rotation)
            matrix[0:3, 3] = translation
            static_transforms[key] = matrix
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as err:
            print(err)
            return None
    return static_transforms[key]

matrix = None
def point_conversion(point, cur_frame, dest_frame):
    global matrix
    if matrix is None:
        matrix = get_static_transform(cur_frame, dest_frame)
    
    if matrix is None:
        return (0.0, 0.0, 0.0)
    
    pt = np.array([point[0], point[1], point[2], 1.0])
    transformed = matrix @ pt
    return (transformed[0], transformed[1], transformed[2])

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

def draw_arrow(pt_A, pt_B, reference_point):
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

    # (x, y, z) = point_conversion(tuple(pt_start), 'world', 'color')
    (x,y,z) = world_to_robot_conversion(tuple(pt_start))
    (x,y,z) = point_conversion((x, y, z), 'base_link', 'color')
    xA_p, yA_p = camera.project3dToPixel((x, y, z)) 

    # (x, y, z) = point_conversion(tuple(pt_end), 'world', 'color')
    (x,y,z) = world_to_robot_conversion(tuple(pt_end))
    (x,y,z) = point_conversion((x, y, z), 'base_link', 'color')
    xB_p, yB_p = camera.project3dToPixel((x, y, z))

    pt1 = np.array([xA_p, yA_p])
    pt2 = np.array([xB_p, yB_p])

    # Calculate the angle between vec and vertical, signed
    theta_rad = np.arctan2(pt2[0] - pt1[0], pt2[1] - pt1[1])

    # cv2.arrowedLine(image, tuple(pt1.astype(int)), tuple(pt2.astype(int)), (255, 0, 255), 2, tipLength=0.2)
    # cv2.circle(image, tuple(pt1.astype(int)), 5, (255, 0, 0), -1)  # Start point in blue
    # cv2.circle(image, tuple(pt2.astype(int)), 5, (0, 0, 255), -1)  # end point in red

    # bp = get_border_intersection(xA_p, yA_p, theta_rad, image.shape[1], image.shape[0])
    bp = get_border_intersection(xA_p, yA_p, theta_rad, 848, 480)

    if bp is not None:

        final_angle = 180-np.degrees(theta_rad)
        if final_angle > 180:
            final_angle -= 360

        # print("Border intersection point:", bp, "Angle with vertical (deg):", final_angle)

        # ep = (int(bp[0] + 60 * np.sin(theta_rad)), int(bp[1] + 60 * np.cos(theta_rad)))
        # cv2.arrowedLine(image, bp, ep, (0, 0, 0), 5, tipLength=0.15)
        # cv2.arrowedLine(image, bp, ep, (0, 255, 0), 3, tipLength=0.15)
        # cv2.circle(image, bp, 7, (0, 255, 0), -1)
        # cv2.circle(image, bp, 3, (0, 0, 0), -1)
    else:

        print("No line found.")
        return -1000, -1000, -1000

    # cv2.arrowedLine(image, tuple(pt1.astype(int)), tuple(pt2.astype(int)), (255, 0, 255), 2, tipLength=0.2)
    # cv2.circle(image, tuple(pt1.astype(int)), 5, (255, 0, 0), -1)

    # cv2.imshow("Arrow AB", image)
    # cv2.waitKey(1)

    return bp[0], bp[1], final_angle

# def stateCallback(msg):
#     global model_pose_,model_idx, t, t2
#     model_pose_.header.stamp = rospy.Time.now()
    
#     if msg.name[model_idx] == model_:
#         model_pose_.pose = msg.pose[model_idx]
#     else:
#         for idx, name in enumerate(msg.name):
#             if name == model_:
#                 model_pose_.pose = msg.pose[idx]
#                 model_idx = idx
    # print(model_pose_)
def robotPoseCallback(msg):
    global model_pose_
    model_pose_ = msg.pose
    # model_pose_.pose.position.z = 2.75 + model_pose_.pose.position.x * 0.08 # sloped worlds
    model_pose_.pose.position.z = 0.021569# flat world
    # print(msg.pose.pose.orientation)

# def linkStateCallback(msg):
#     global model_pose_, model_idx
#     model_pose_.header.stamp = rospy.Time.now()
    
#     # Modify the search string based on your robot's link name in Gazebo
#     link_name = f"{model_}::base_link"
    
#     try:
#         idx = msg.name.index(link_name)
#         model_pose_.pose = msg.pose[idx]
#     except ValueError:
#         rospy.logwarn(f"Link {link_name} not found in link_states")
#     print(model_pose_)

def robot_pose_conversion(pose):
    # Get robot position and orientation from model_pose_
    robot_pos = np.array([
        model_pose_.pose.position.x,
        model_pose_.pose.position.y,
        model_pose_.pose.position.z
    ])
    
    # Get quaternion components
    qx = model_pose_.pose.orientation.x
    qy = model_pose_.pose.orientation.y
    qz = model_pose_.pose.orientation.z
    qw = model_pose_.pose.orientation.w
    
    # Convert quaternion to rotation matrix
    R = np.array([
        [1 - 2*qy**2 - 2*qz**2,  2*qx*qy - 2*qz*qw,    2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw,      1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw,      2*qy*qz + 2*qx*qw,    1 - 2*qx**2 - 2*qy**2]
    ])
    
    # Convert input point to numpy array
    point = np.array([pose[0], pose[1], pose[2]])
    
    # Apply rotation and translation
    world_point = R @ point + robot_pos
    
    return tuple(world_point)

def world_to_robot_conversion(world_pose):
    """
    Convert a point from world frame to robot's base frame
    @param world_pose: tuple (x,y,z) representing point in world frame
    @return: tuple (x,y,z) representing point in robot's base frame
    """
    # Get robot position from model_pose_
    robot_pos = np.array([
        model_pose_.pose.position.x,
        model_pose_.pose.position.y,
        model_pose_.pose.position.z
    ])
    
    # Get quaternion components
    qx = model_pose_.pose.orientation.x
    qy = model_pose_.pose.orientation.y
    qz = model_pose_.pose.orientation.z
    qw = model_pose_.pose.orientation.w
    
    # Convert quaternion to rotation matrix
    R = np.array([
        [1 - 2*qy**2 - 2*qz**2,  2*qx*qy - 2*qz*qw,    2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw,      1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw,      2*qy*qz + 2*qx*qw,    1 - 2*qx**2 - 2*qy**2]
    ])
    
    # Convert input world point to numpy array
    world_point = np.array(world_pose)
    
    # Apply inverse transformation:
    # p_base = R^T * (p_world - t)
    # R^T is the transpose of R (inverse of rotation matrix)
    base_point = R.T @ (world_point - robot_pos)
    
    return tuple(base_point)

def main():
    point2d = Pose2D()
    point2d.x = 0.0
    point2d.y = 0.0
    point2d.theta = 0.0

    while not rospy.is_shutdown():
        
        index = get_ros_parameter()
        # If you want to start from bottom left and follow all rows
        # if index % 2 == 0:
        #     pt_A = pt_A_list[index]
        #     pt_B = pt_B_list[index]
        # else:
        #     pt_A = pt_B_list[index]
        #     pt_B = pt_A_list[index]

        # If you want to start left middel bottom and follow one row :D 
        pt_A = pt_A_list[index]
        pt_B = pt_B_list[index]

        #reference_point = point_conversion((1.0, 0, 0), 'base_link', 'world')'
        reference_point = robot_pose_conversion((1.0, 0, 0))
        # print("Reference point in world frame:", reference_point, "\nRobot pose:", model_pose_.pose.position)
        # if cv_image is not None:
        #     image = cv_image.copy()
        #     x, y , theta = draw_arrow(image, pt_A, pt_B, reference_point)

        x, y, theta = draw_arrow(pt_A, pt_B, reference_point)
        point2d.x = x
        point2d.y = y
        point2d.theta = theta
        
        pub.publish(point2d)

        rate.sleep()

if __name__ == '__main__':
    rospy.init_node('true_line_node')
    get_caminfo()

    mytfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(mytfBuffer)
    
    pub = rospy.Publisher('/true_line', Pose2D, queue_size=1)

    # rospy.Subscriber('/r200/camera/color/image_raw/compressed' , CompressedImage , compressed_img_callback)
    # rospy.Subscriber('/gazebo/model_states', ModelStates, stateCallback)
    # rospy.Subscriber('/gazebo/link_states', LinkStates, linkStateCallback)
    rospy.Subscriber('/odometry/base_raw', Odometry, robotPoseCallback) # faster but does not include height. We estimate :D 

    rate = rospy.Rate(30)
    main()