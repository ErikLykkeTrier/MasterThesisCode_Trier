#!/usr/bin/env python
import rospy
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Pose, Twist, Quaternion
# pub2 = None

def set_line_number(n):
    rospy.set_param('/line_nr', n)
    rospy.loginfo(f"Parameter '/line_nr' set to {n}")

# def reset_robot(x=-0.5, y=-21, z=2.72, yaw=0.7071, yaw2=0.7071): # Straight ahead, line starts to the left in the image
# def reset_robot(x=-1.3, y=-21, z=2.72, yaw=0.5071, yaw2=1): # way left pointing in
def reset_robot(x, y, z, qx, qy, qz, qw):
    global pub2
    pub2 = rospy.Publisher('/twist_mux/cmd_vel', Twist, queue_size=10)
    rospy.sleep(1)
    
    rospy.wait_for_service('/gazebo/set_model_state')
    try:
        set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        
        state_msg = ModelState()
        state_msg.model_name = 'thorvald'
        state_msg.pose.position.x = x
        state_msg.pose.position.y = y
        state_msg.pose.position.z = z
        state_msg.pose.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)
        # state_msg.pose.orientation.x = roll
        # state_msg.pose.orientation.y = pitch
        # state_msg.pose.orientation.z = yaw
        # state_msg.pose.orientation.w = omega
        state_msg.twist = Twist()
        state_msg.reference_frame = 'world'
        
        resp = set_state(state_msg)

        control_ouput = Twist()
        pub2.publish(control_ouput)
        set_line_number(1)
        print(resp.status_message)
    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)

if __name__ == '__main__':
    rospy.init_node('reset_robot_node')
    pub2 = rospy.Publisher('/twist_mux/cmd_vel', Twist, queue_size=10)
    rospy.sleep(1)
    # reset_robot(x=-1.3, y=-21, z=2.72, roll=0, pitch=0, yaw=0.5071, omega=1)
