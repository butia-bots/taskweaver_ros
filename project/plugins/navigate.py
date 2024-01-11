from operator import itemgetter
from butia_world_msgs.srv import GetKey, GetPose, GetPoseResponse
from actionlib.simple_action_client import SimpleActionClient
import rospy
from typing import Dict
from taskweaver.plugin import Plugin, register_plugin
from move_base_msgs.msg import MoveBaseAction, MoveBaseActionGoal
from nav_msgs.msg import OccupancyGrid
from copy import deepcopy
from ros_numpy import numpify
import cv2
import numpy as np
from std_msgs.msg import Float64
import random
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from moveit_commander import MoveGroupCommander
import tf

@register_plugin
class Navigate(Plugin):
    def __init__(self, name, ctx, config):
        super().__init__(name=name, ctx=ctx, config=config)
        rospy.init_node("taskweaver_navigate", anonymous=True)
        self.get_key_srv = rospy.ServiceProxy(self.config.get('get_key_srv', '/butia_world/get_closest_key'), GetKey)
        self.get_pose_srv = rospy.ServiceProxy(self.config.get('get_pose_srv', '/butia_world/get_pose'), GetPose)
        self.move_base_client = SimpleActionClient(self.config.get('move_base_ns', 'move_base'), MoveBaseAction)
        self.neck_pan_publisher = rospy.Publisher(self.config.get("neck_pan_topic", "/doris_head/head_pan_position_controller/command"), Float64)
        self.neck_tilt_publisher = rospy.Publisher(self.config.get("neck_tilt_topic", "/doris_head/head_tilt_position_controller/command"), Float64)
        self._occupancy_grid: OccupancyGrid = None
        self.map_subscriber = rospy.Subscriber(self.config.get("map_topic", "/map"), OccupancyGrid, self._update_occupancy_grid)
        self.arm = MoveGroupCommander("arm", ns="/doris_arm")
        self.tfl = tf.TransformListener()

    def __call__(
        self,
        location_name: str
    ):
        self.arm.set_named_target("Home")
        self.arm.go(wait=True)
        rate = rospy.Rate(50)
        start = rospy.Time.now()
        while rospy.Time.now() - start < rospy.Duration(5):
            self.neck_pan_publisher.publish(self.config.get("pan_angle", 0.0))
            self.neck_tilt_publisher.publish(self.config.get("tilt_angle", 0.0))
            rate.sleep()
        rate = rospy.Rate(1.0)
        rate.sleep()
        response = self.get_key_srv.call(query=location_name, threshold=self.config.get('multimodal_similarity_threshold', 0.8))
        if response.success == False:
            return False
        else:
            response: GetPoseResponse = self.get_pose_srv.call(key=response.key+"/pose")
            pose = response.pose
            size = response.size
            quat = [
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            ]
            roll, pitch, yaw = euler_from_quaternion(quat)
            offset = 1.0
            poses = []
            for i in range(4):
                pose_side = deepcopy(pose)
                pose_side.position.x += np.cos((yaw + i*np.pi/2))*(size.x+offset)
                pose_side.position.y += np.sin((yaw + i*np.pi/2))*(size.y+offset)
                pose_side.position.z = 0.0
                quat = quaternion_from_euler(0.0, 0.0, (yaw + (i+2)*np.pi/2))
                pose.orientation.x = quat[0]
                pose.orientation.y = quat[1]
                pose.orientation.z = quat[2]
                pose.orientation.w = quat[3]
                poses.append(pose)
            self.tfl.waitForTransform("base_footprint", "map", rospy.Time(), rospy.Duration(1.0))
            try:
                position, orientation = self.tfl.lookupTransform("map", "base_footprint", rospy.Time.now())
            except:
                position = None
            if position is not None:
                poses = sorted(poses, key=lambda p: self._euclidean_distance((p.position.x, p.position.y, p.position.z), position))
            nav_goal = MoveBaseActionGoal()
            nav_goal.header.frame_id = self.config.get("map_frame", "map")
            nav_goal.goal.target_pose.header.frame_id = self.config.get("map_frame", "map")
            nav_goal.goal.target_pose.pose.position = poses[0].position
            nav_goal.goal.target_pose.pose.orientation = poses[0].orientation
            self.move_base_client.send_goal_and_wait(nav_goal.goal)
            return True

    def _update_occupancy_grid(self, msg: OccupancyGrid):
        self._occupancy_grid = msg

    def _euclidean_distance(self, p0, p1):
        return np.sqrt((p0[0]-p1[0])**2 + (p0[1]-p1[1])**2 + (p0[2]-p1[2])**2)

    @property
    def occupancy_grid(self)->OccupancyGrid:
        return deepcopy(self._occupancy_grid)
