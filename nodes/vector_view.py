import rospy, utilities, cv2, numpy as np, datetime, csv
from geometry_msgs.msg import Pose2D, Twist
from sensor_msgs.msg import CompressedImage
from tf2_geometry_msgs import PointStamped
from nav_msgs.msg import Odometry
from tf.transformations import quaternion_from_euler
from cv_bridge import CvBridge
bridge = CvBridge()

from reset_robot import reset_robot#, pub2

# Global variable to store the latest message
latest_pose = None
cv_image = None
model_pose_ = None
    
np.random.seed(42)

def callback(msg):
    global latest_pose
    latest_pose = msg

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


def compressed_img_callback(msg):
    global cv_image
    np_arr = np.frombuffer(msg.data, np.uint8)
    cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

def robotPoseCallback(msg):
    global model_pose_
    model_pose_ = msg.pose
    model_pose_.pose.position.z = model_pose_.pose.position.x * 0.08 + 2.75 # from world gen

def normal_noise(mean=0.0, std_dev=1.0, size=None):
    return np.random.normal(loc=mean, scale=std_dev, size=size)

def main():
    global latest_pose, cv_image, model_pose_

    pub = rospy.Publisher('/twist_mux/cmd_vel', Twist, queue_size=10)
    v_d = 1.5
    controller = utilities.RobustController(lim=1, vd=v_d, dt=1/30, robust = False) # dt based on frequency of node, lim unused

    control_ouput = Twist()
    # control_ouput.angular.z = 0


    dtdt = datetime.datetime.now()
    day_time = dtdt.strftime("%d%m%H%M")
    name = "robust" if controller.robust else "proportional"
    
    # file_name = f"../data/final/slope_lr_smooth_{name}_{day_time}.csv"
    start_pos = "left_offset"
    cam_state = "certain"
    file_name = f"true_flat_bumpy_{start_pos}_{name}_{cam_state}_{day_time}.csv"
    folder = "cam_certainity/without_prime" #"prop_vs_robust" #"true_vs_ML"
    full_path = f"../DATAFINAL/{folder}/{file_name}"
    try:
        k = open(full_path, "x")
        k.close()
    except FileExistsError:
        print(f"File {full_path} already exists, overwriting it.")
    
    f = open(full_path, "w", newline='')
    writer = csv.writer(f)
    writer.writerow(["xv", "yv", "thetav", "x", "y", "z", "r", "p", "yaw", "v", "w_prime"])


    # ynorm_pose = utilities.BoundedList(n, "ynormv")

    while not rospy.is_shutdown():
        if latest_pose is not None and cv_image is not None and model_pose_ is not None:
            x = int(latest_pose.x)
            y = int(latest_pose.y)
            theta = np.deg2rad(latest_pose.theta) # For control

            x_norm = norm(x, cv_image.shape[1])
            y_norm = norm(y, cv_image.shape[0])

            # Add noise to image features
            # x_norm = np.clip(x_norm + normal_noise(mean=0, std_dev=0.0), -1, 1)
            # y_norm = np.clip(y_norm + normal_noise(mean=0, std_dev=0.0), -1, 1)
            # theta =  np.clip(theta + normal_noise(mean=0, std_dev=0.00*np.pi), -np.pi, np.pi)
            

            visual_theta = theta - np.pi/2

            # omega = controller.step_row(norm(x, cv_image.shape[1]),norm(y, cv_image.shape[0]), theta) # This did not work
            # omega = controller.control_step(x_norm, y_norm, theta)
            omega_prime = controller.control_step(x_norm, y_norm, theta)
            # omega = lim_w(omega, np.pi/8)

            control_ouput.linear.x = v_d
            control_ouput.angular.z = omega_prime
            pub.publish(control_ouput)

            cv2.circle(cv_image, (x, y), 7, (0, 255, 0), -1)
            cv2.circle(cv_image, (x, y), 3, (0, 0, 0), -1)
            cv2.arrowedLine(cv_image, (x, y), (int(x + 70 * np.cos(visual_theta)), int(y + 70 * np.sin(visual_theta))), (0, 255, 0), 7, tipLength=0.2)
            cv2.arrowedLine(cv_image, (x, y), (int(x + 70 * np.cos(visual_theta)), int(y + 70 * np.sin(visual_theta))), (0, 0, 0), 3, tipLength=0.2)

            cv2.imshow("Camera Image", cv_image)
            
            # Extract robot pose
            x_robot = model_pose_.pose.position.x
            y_robot = model_pose_.pose.position.y
            z_robot = model_pose_.pose.position.z
            orientation = model_pose_.pose.orientation
            r_robot = orientation.x
            p_robot = orientation.y
            yaw_robot = orientation.z


            writer.writerow([x,y,theta, x_robot, y_robot, z_robot, r_robot, p_robot, yaw_robot, controller.v_d, omega_prime.item()])

            fisk = cv2.waitKey(1)
            if fisk == ord('q'):
                print("HELLO")
                f.close()
                break
        else:
            rate.sleep()

if __name__ == '__main__':
    rospy.init_node('true_line_printer', anonymous=True)
    rospy.Subscriber('/true_line', Pose2D, callback)
    rospy.Subscriber('/r200/camera/color/image_raw/compressed' , CompressedImage , compressed_img_callback)
    rospy.Subscriber('/odometry/base_raw', Odometry, robotPoseCallback)

    # pub2 = rospy.Publisher('/twist_mux/cmd_vel', Twist, queue_size=10)
    
    rate = rospy.Rate(30)  # Hz

    # Set start pose
    rospy.sleep(1)

    # Start positions flat world
    # x,y,z,r,p,yaw = (0.967481, -20.989647, 0.021569, 0.0, 0.0, 1.588244) # straight on
    # x,y,z,r,p,yaw = (0.667481, -20.989647, 0.021569, 0.0, 0.0, 1.588244) # straight on
    x,y,z,r,p,yaw = (0.667481, -20.989647, 4.921569, 0.0, 0.0, 1.588244) # straight on bumpy road
    # x,y,z,r,p,yaw = (1.482014, -20.851523, 0.021569, 0.0, 0.0, 2.337844) # from right
    # x,y,z,r,p,yaw = (0.416909, -20.792364, 0.021569, 0.0, 0.0, 0.817370) # from left - failed
    # x,y,z,r,p,yaw = (0.416909, -20.792364, 0.021569, 0.0, -0.053494, 1.005467) # from left Works!
    # x,y,z,r,p,yaw = (0.000664, -20.573657, 0.021569, 0.0, 0, 1.188267) #Starts in column and works!


    # Start positions slope left right
    # x,y,z,r,p,yaw = (1.005811, -20.772925, 2.851775, -0.068896, 0.001340, 1.590212) # straight
    # x,y,z,r,p,yaw = (1.438389, -20.826057, 2.881630, -0.052244, 0.044957, 2.280777) # from right
    # x,y,z,r,p,yaw = (0.267664, -20.869485, 2.793623, -0.058682, -0.053494, 0.775467) # from left, this failed for all controllers, too far :D 

    # Start positions slope back front
    # x,y,z,r,p,yaw = (-1.849903,-17.016098, 2.633105, -0.044410, -0.052710, 0.700699) # From right
    # x,y,z,r,p,yaw = (-1.928843,-16.434568, 2.627657, 0.000047, -0.068910, -0.000672) # Straight on
    # x,y,z,r,p,yaw = (-1.919183, -15.870317, 2.628320, 0.047657, -0.049789, -0.764152) # From left

    # Translate rpy into these quoterniions that are used for positionsing
    qx, qy, qz, qw = quaternion_from_euler(r,p,yaw)


    reset_robot(x=x, y=y, z=z, qx=qx, qy=qy, qz=qz, qw=qw) 

    main()