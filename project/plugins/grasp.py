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
from sensor_msgs.msg import Image
import math
from geometry_msgs.msg import PoseStamped
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

@register_plugin
class Grasp(Manipulate):
    def __call__(
        self,
        object_name: str
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
        results = vision_collection.query.near_text(query=object_name, certainty=self.config.get("multimodal_similarity_threshold", 0.2))
        filtered_objects = []
        size_limit = 0.2
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

        pose = PoseStamped()
        pose.header.frame_id = description.header.frame_id
        pose.pose.position = description.bbox.center.position
        pose.pose.orientation = description.bbox.center.orientation
        grasp_pose = deepcopy(pose)
        box_pose = deepcopy(pose)
        cloud = numpify(description.filtered_cloud)
        #grasp_pose.pose.position.z = min(cloud['z'][~np.isnan(cloud['z'])])
        #grasp_pose.pose.position.z += 0.1
        arm_reference_frame = self.arm.get_planning_frame()
        self.tfl.waitForTransform(arm_reference_frame, pose.header.frame_id, rospy.Time(), rospy.Duration(10.0))
        try:
            grasp_pose = self.tfl.transformPose(arm_reference_frame, grasp_pose)
            box_pose = self.tfl.transformPose(arm_reference_frame, box_pose)
        except:
            return False
        #ps.pose.orientation.x = 0.0
        #ps.pose.orientation.y = 0.0
        #ps.pose.orientation.z = 0.0
        #ps.pose.orientation.w = 1.0
        #ps.pose.position.x -= description.bbox.size.x
        self.arm.set_named_target("Home")
        self.arm.go(wait=True)
        self.arm.stop()
        self.gripper.set_named_target("Open")
        self.gripper.go(wait=True)
        position_grasp = [grasp_pose.pose.position.x, grasp_pose.pose.position.y, grasp_pose.pose.position.z]
        size = [description.bbox.size.x, description.bbox.size.y, description.bbox.size.z]
        #self.scene.add_box(object_name, ps, [0.05, 0.05, 0.05])
        pose_table = deepcopy(grasp_pose)
        pose_table.pose.position.z = pose_table.pose.position.z/2.0 - 0.1
        pose_table.pose.orientation.x = 0
        pose_table.pose.orientation.y = 0
        pose_table.pose.orientation.z = 0
        pose_table.pose.orientation.w = 1.0
        size_table = [1.0, 2.0, pose_table.pose.position.z*2]
        self.scene.add_box("table", pose_table, size_table)
        position_approach = position_grasp.copy()
        position_approach[2] += 0.3
        #self.arm.set_position_target(position_approach)
        #self.arm.go(wait=True)
        self.arm.set_position_target(position_grasp)
        for i in range(5):
            #size = [s*0.75 for s in size]
            #size = [0.1, 0.1, 0.1]
            self.scene.add_box(object_name, grasp_pose, size)
            if self.arm.go(wait=True):
                break
            break
        touch_links = ["doris_arm/left_finger_link", "doris_arm/right_finger_link"]
        self.scene.attach_box("doris_arm/ee_gripper_link", object_name, touch_links=touch_links)
        self.gripper.set_named_target("Closed")
        self.gripper.go(wait=True)
        rospy.Rate(1.0/2.0).sleep()
        self.arm.set_named_target("Home")
        self.arm.go(wait=True)
        return True
