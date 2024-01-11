from operator import itemgetter
from butia_world_msgs.srv import GetKey, GetPose
import rospy
from typing import Dict
from taskweaver.plugin import Plugin, register_plugin
from copy import deepcopy
from ros_numpy import numpify
import cv2
import numpy as np
from butia_world_msgs.srv import GetKey, GetPose
from butia_vision_msgs.msg import Recognitions3D, Description3D
import os
import moveit_commander
from moveit_msgs.msg import Grasp as GraspMsg
from trajectory_msgs.msg import JointTrajectoryPoint
from std_msgs.msg import Float64
from sensor_msgs.msg import Image, PointCloud2
import math
from geometry_msgs.msg import PoseStamped, PointStamped
from std_srvs.srv import Empty
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import tf
import weaviate
import weaviate.classes as wvc
from butia_vision_msgs.msg import Recognitions3D
import base64
from io import BytesIO
from PIL import Image as PILImage
from taskweaver.plugin.manipulate import Manipulate
import random

@register_plugin
class Place(Manipulate):
    def __call__(
        self,
        container_name: str
    ):
        self.arm.set_named_target("Home")
        self.arm.go(wait=True)
        self.scene.clear()
        rate = rospy.Rate(50)
        start = rospy.Time.now()
        while rospy.Time.now() - start < rospy.Duration(5):
            self.neck_pan_publisher.publish(self.config.get("pan_angle", 0.0))
            self.neck_tilt_publisher.publish(self.config.get("tilt_angle", np.pi/4.0))
            rate.sleep()
        vision_collection, recognitions = self.get_vision_collection()
        results = vision_collection.query.near_text(query=container_name, certainty=self.config.get("multimodal_similarity_threshold", 0.2))
        filtered_objects = []
        size_limit = 10.0
        for obj in results.objects:
            description_id = int(obj.properties['descriptionId'])
            description = recognitions.descriptions[description_id]
            if description.bbox.size.x > size_limit or description.bbox.size.y > size_limit or description.bbox.size.z > size_limit:
                continue
            filtered_objects.append(obj)
        if len(filtered_objects) == 0:
            return False
        description_id = int(filtered_objects[0].properties['descriptionId'])
        description: Description3D = recognitions.descriptions[description_id]
        cloud = numpify(description.filtered_cloud)
        self.arm.set_named_target("CraneHigh")
        self.arm.go(wait=True)
        self.arm.stop()
        success = False
        while not success:
            point = random.choice(cloud)
            pose = PoseStamped()
            pose.header.frame_id = description.header.frame_id
            pose.pose.position = description.bbox.center.position
            pose.pose.orientation = description.bbox.center.orientation
            arm_reference_frame = self.arm.get_planning_frame()
            self.tfl.waitForTransform(arm_reference_frame, pose.header.frame_id, rospy.Time(), rospy.Duration(10.0))
            try:
                ps = self.tfl.transformPose(arm_reference_frame, pose)
            except:
                return False
            transformed_points = []
            for i in range(len(cloud['x'])):
                point = PointStamped()
                point.header.frame_id = description.filtered_cloud.header.frame_id
                point.point.x = cloud['x'][i]
                point.point.y = cloud['y'][i]
                point.point.z = cloud['z'][i]
                transformed_points.append(self.tfl.transformPoint(arm_reference_frame, point))
            transformed_points.sort(key=lambda ps: ps.point.z, reverse=True)
            cps = self.arm.get_current_pose()
            position_approach = [cps.pose.position.x, cps.pose.position.y, cps.pose.position.z]
            position_place = position_approach.copy()
            position_place[2] = transformed_points[0].point.z
            rospy.loginfo(position_place)
            self.arm.set_position_target(position_place)
            if self.arm.go(wait=True) != -1:
                success = True
            success = True
        self.gripper.set_named_target("Open")
        self.gripper.go(wait=True)
        self.arm.set_position_target(position_approach)
        self.arm.go(wait=True)
        self.arm.set_named_target("Home")
        self.arm.go(wait=True)
        return True
