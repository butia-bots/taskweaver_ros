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

def toBase64(img: PILImage.Image):
  buffered = BytesIO()
  img.save(buffered, format="JPEG")
  img_str = base64.b64encode(buffered.getvalue())
  return img_str

class Manipulate(Plugin):
    def __init__(self, name, ctx, config):
        rospy.init_node("taskweaver_manipulate", anonymous=True)
        super().__init__(name=name, ctx=ctx, config=config)
        self.scene = moveit_commander.PlanningSceneInterface(ns="/doris_arm")
        self.arm = moveit_commander.MoveGroupCommander("arm", ns="/doris_arm")
        self.arm.set_end_effector_link("doris_arm/ee_gripper_link")
        self.arm.allow_replanning(True)
        self.arm.set_goal_position_tolerance(0.05)
        self.arm.set_goal_orientation_tolerance(math.pi/24)
        #self.arm.set_planning_time(1.0)
        self.gripper = moveit_commander.MoveGroupCommander("gripper", ns="/doris_arm")
        self.neck_pan_publisher = rospy.Publisher(self.config.get("neck_pan_topic", "/doris_head/head_pan_position_controller/command"), Float64)
        self.neck_tilt_publisher = rospy.Publisher(self.config.get("neck_tilt_topic", "/doris_head/head_tilt_position_controller/command"), Float64)
        self.trigger_forget = rospy.ServiceProxy("/butia_world/trigger_remove", Empty)
        self.tfl = tf.TransformListener()
        self.vector_client = weaviate.connect_to_local()
        self.vision_subscriber = rospy.Subscriber("/butia_vision/br/object_recognition3d", Recognitions3D, self._update_recognitions)

    def _update_recognitions(self, msg):
        self._recognitions: Recognitions3D = msg

    def get_vision_collection(self):
        recognitions = deepcopy(self._recognitions)
        if self.vector_client.collections.exists("VisionObjects"):
            self.vector_client.collections.delete("VisionObjects")
        vision_collection = self.vector_client.collections.create(
            name="VisionObjects",
            vectorizer_config=wvc.Configure.Vectorizer.multi2vec_clip(image_fields=["image"])
        )
        for i, description in enumerate(recognitions.descriptions):
            if description.bbox.size.x > 0 and description.bbox.center.position.z < 1.5:
                vision_collection.data.insert(
                    properties={
                        "image": toBase64(PILImage.fromarray(numpify(recognitions.image_rgb)[int(description.bbox2D.center.y-description.bbox2D.size_y//2):int(description.bbox2D.center.y+description.bbox2D.size_y//2),int(description.bbox2D.center.x-description.bbox2D.size_x//2):int(description.bbox2D.center.x+description.bbox2D.size_x//2)][:,:,::-1])).decode(),
                        "descriptionId": i
                    },
                )
        return vision_collection, recognitions