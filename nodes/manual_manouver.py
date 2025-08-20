# Importos
import rospy
from geometry_msgs.msg import Pose2D, Twist

# Testcombo of v and w to drive robot one turn :D

n = 53

def main(v,w):
    pub = rospy.Publisher('/twist_mux/cmd_vel', Twist, queue_size=10)

    control_ouput = Twist()
    control_ouput.linear.x = v
    control_ouput.angular.z = w

    # while not rospy.is_shutdown():
    for i in range(n):
        pub.publish(control_ouput)
        rate.sleep()


if __name__ == '__main__':
    rospy.init_node('manouver_time')
    rate = rospy.Rate(10)

    main(0.5, 0.6)