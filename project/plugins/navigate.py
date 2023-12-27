from operator import itemgetter
from butia_world_msgs.srv import GetKey, GetPose
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

@register_plugin
class Navigate(Plugin):
    def __init__(self, name, ctx, config):
        super().__init__(name=name, ctx=ctx, config=config)
        self.get_key_srv = rospy.ServiceProxy(self.config.get('get_key_srv', '/butia_world/get_closest_key'), GetKey)
        self.get_pose_srv = rospy.ServiceProxy(self.config.get('get_pose_srv', '/butia_world/get_pose'), GetPose)
        self.move_base_client = SimpleActionClient(self.config.get('move_base_ns', 'move_base'), MoveBaseAction)
        self.neck_pan_publisher = rospy.Publisher(self.config.get("neck_pan_topic", "/doris_head/head_pan_position_controller/command"), Float64)
        self.neck_tilt_publisher = rospy.Publisher(self.config.get("neck_tilt_topic", "/doris_head/head_tilt_position_controller/command"), Float64)
        self._occupancy_grid: OccupancyGrid = None

    def __call__(
        self,
        location_name: str
    ):
        self.neck_pan_publisher.publish(self.config.get("pan_angle", 0.0))
        self.neck_tilt_publisher.publish(self.config.get("tilt_angle", -np.pi/4))
        if not self._explore_until_find(location_name=location_name):
            return False
        response = self.get_key_srv.call(query=location_name, threshold=self.config.get('multimodal_similarity_threshold', 0.8))
        if response.success == False:
            return False
        else:
            response = self.get_pose_srv.call(key=response.key)
            pose = response.pose
            nav_goal = MoveBaseActionGoal()
            nav_goal.header.frame_id = self.config.get("map_frame", "map")
            nav_goal.goal.target_pose.header.frame_id = self.config.get("map_frame", "map")
            nav_goal.goal.target_pose.pose = pose
            self.move_base_client.send_goal_and_wait(nav_goal)
            return True

    def _explore_until_find(self, location_name: str):
        frontier_centers = []
        last_frontier = None
        while not self.get_key_srv.call(query=location_name, threshold=self.config.get('multimodal_similarity_threshold', 0.8)).success:
            occupancy_grid_msg: OccupancyGrid = self.occupancy_grid
            occupancy_grid = numpify(occupancy_grid_msg)
            occupied = occupancy_grid > 0
            free = occupancy_grid == 0
            free_edges = cv2.Canny(free*255, 100, 200)
            shape = cv2.MORPH_RECT
            element = cv2.getStructuringElement(shape, (5,5), (2,2))
            occupied_expanded = cv2.dilate(occupied*255, element)
            frontiers = (free_edges > 0) and not (occupied_expanded > 0)
            frontiers_expanded = cv2.dilate(frontiers*255, element)
            _, frontiers, _ = cv2.findContours(frontiers_expanded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for frontier in frontiers:
                frontier_center = np.mean(frontier, axis=1)
                x = occupancy_grid_msg.info.origin.position.x + frontier_center[0]*occupancy_grid_msg.info.resolution
                y = occupancy_grid_msg.info.origin.position.y + frontier_center[1]*occupancy_grid_msg.info.resolution
                lower_x = self.config.get('lower_x', -2.0)
                upper_x = self.config.get('upper_x', 2.0)
                lower_y = self.config.get('lower_y', -2.0)
                upper_y = self.config.get('upper_y', 2.0)
                if lower_x < x and x < upper_x and lower_y < y and y < upper_y:
                    rospy.loginfo(frontier_center)
                    frontier_centers.append(frontier_center)
            if len(frontier_centers) > 0:
                frontier_center = frontier_centers[0]
                x = occupancy_grid_msg.info.origin.position.x + frontier_center[0]*occupancy_grid_msg.info.resolution
                y = occupancy_grid_msg.info.origin.position.y + frontier_center[1]*occupancy_grid_msg.info.resolution
                nav_goal = MoveBaseActionGoal()
                nav_goal.header.frame_id = self.config.get("map_frame", "map")
                nav_goal.goal.target_pose.header.frame_id = self.config.get("map_frame", "map")
                nav_goal.goal.target_pose.pose.position.x = x
                nav_goal.goal.target_pose.pose.position.y = y
                self.move_base_client.send_goal_and_wait(nav_goal)
                last_frontier = frontier_center
                frontier_centers.pop(0)
            else:
                return False
        return True

    def _update_occupancy_grid(self, msg: OccupancyGrid):
        self._occupancy_grid = msg

    @property
    def occupancy_grid(self)->OccupancyGrid:
        return deepcopy(self._occupancy_grid)

@test_plugin(name="test Navigate", description="test")
def test_call(api_call):
    result = api_call(location_name="table")
    assert result