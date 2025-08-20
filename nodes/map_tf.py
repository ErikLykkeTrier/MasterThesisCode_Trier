#!/usr/bin/env python  
import rospy

import numpy as np
import tf2_ros, tf_conversions
import geometry_msgs.msg
from geometry_msgs.msg import Twist
from tf2_geometry_msgs import PoseStamped
from nav_msgs.msg import Odometry
from gazebo_msgs.msg import ModelStates

model_ = "thorvald"
model_pose_ = PoseStamped()
model_pose_.pose.orientation.w = 1
model_pose_.header.frame_id = "world"
model_idx = 0

# pt_A_list = [( -0.68, -22, 2.72), (0.98, -22, 2.84), (2.57, -22, 2.96), (4.2, -22, 3)]
# pt_B_list = [(-0.68, -8, 2.72), (0.98, -8, 2.84), (2.5, -8, 2.96), (4.137, -8, 3)]

change_line_list = [(-0.68, 10, 2.72), (0.98, -22, 2.84), (2.5, -8, 2.96),(4.2, -22, 3)]

def get_ros_parameter():
    global param_value
    try:
        param_value = rospy.get_param('/line_nr', 0) # 'default_value' is used if the parameter doesn't exist
        # rospy.loginfo(f"Value of '/line_nr': {param_value}")
    except KeyError:
        rospy.logwarn("Parameter '/line_nr' does not exist.")

def set_line_number(n):
    rospy.set_param('/line_nr', n)
    rospy.loginfo(f"Parameter '/line_nr' set to {n}")

def stateCallback(msg):
    global model_pose_,model_idx, t, t2
    model_pose_.header.stamp = rospy.Time.now()
    
    if msg.name[model_idx] == model_:
        model_pose_.pose = msg.pose[model_idx]
    else:
        for idx, name in enumerate(msg.name):
            if name == model_:
                model_pose_.pose = msg.pose[idx]
                model_idx = idx
    #print(model_pose_)
    x_pose = model_pose_.pose.position.x
    y_pose = model_pose_.pose.position.y
    z_pose = model_pose_.pose.position.z
    x_rot = model_pose_.pose.orientation.x
    y_rot = model_pose_.pose.orientation.y
    z_rot = model_pose_.pose.orientation.z
    w_rot = model_pose_.pose.orientation.w

    t.header.stamp = rospy.Time.now()
    t.header.frame_id = "world" #world #base_link_control
    t.child_frame_id = "base_link" #base_link #base_link
    t.transform.translation.x = x_pose
    t.transform.translation.y = y_pose
    t.transform.translation.z = z_pose
    t.transform.rotation.x = x_rot
    t.transform.rotation.y = y_rot
    t.transform.rotation.z = z_rot
    t.transform.rotation.w = w_rot


def main():
    # Set up the manouver output
    # control_ouput = Twist()
    # control_ouput.linear.x = 0.5

    while not rospy.is_shutdown():
    
        pose_publisher.publish(model_pose_)
        br.sendTransform(t)

        # Check if change of line is necessary
        # get_ros_parameter()
        # if np.abs(model_pose_.pose.position.x - change_line_list[param_value][0]) < 1 and np.abs(model_pose_.pose.position.y - change_line_list[param_value][1]) < 1:
        #     print(model_pose_.pose.position.x, model_pose_.pose.position.y)
        #     print(change_line_list[param_value])
        #     print(np.abs(model_pose_.pose.position.x - change_line_list[param_value][0]))
        #     print(np.abs(model_pose_.pose.position.y - change_line_list[param_value][1]))
        #     set_line_number(param_value+1)
        #     if param_value == 4:
        #         rospy.signal_shutdown()
        #     elif param_value % 2 == 0:
        #         control_ouput.angular.z = -0.6
        #     else:
        #         control_ouput.angular.z = 0.6
        #     for _ in range(53*5):
        #         pub.publish(control_ouput)
        #         rate.sleep()

        rate.sleep()

if __name__ == '__main__':
    rospy.init_node('tf2_model_broadcaster')

    rospy.loginfo("Starting tf2 model broadcaster...")
    rospy.Subscriber('/gazebo/model_states', ModelStates, stateCallback)

    pose_publisher = rospy.Publisher(model_ + "_pose", PoseStamped, queue_size=10)
    pub = rospy.Publisher('/twist_mux/cmd_vel', Twist, queue_size=10)

    br = tf2_ros.TransformBroadcaster()

    t  = geometry_msgs.msg.TransformStamped()
    # rate = rospy.Rate(1)
    rate = rospy.Rate(500) # Fast as fluff 

    try:
        main()
    except rospy.ROSInterruptException:
        pass