from moveit_commander import MoveGroupCommander
import rospy
import math

rospy.init_node('test_moveit', anonymous=True)

arm = MoveGroupCommander("arm", ns="/doris_arm", robot_description="/robot_description")
pose = arm.get_current_pose()
pose.pose.position.y += 0.1
p_list = [
pose.pose.position.x,
pose.pose.position.y,
pose.pose.position.z,
0.0,
0.0,
math.atan2(pose.pose.position.y, pose.pose.position.x)
]
arm.set_position_target(p_list[:3])
arm.go(wait=True)
