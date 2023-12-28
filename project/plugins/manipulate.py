from operator import itemgetter
from butia_world_msgs.srv import GetKey, GetPose
import rospy
from typing import Dict
from taskweaver.plugin import Plugin, register_plugin
from copy import deepcopy
from ros_numpy import numpify
import cv2
import numpy as np
from octo.model.octo_model import OctoModel
from butia_vision_msgs.msg import Recognitions3D
import os
import jax
from interbotix_xs_modules.arm import InterbotixManipulatorXS
import interbotix_common_modules.angle_manipulation as ang
from std_msgs.msg import Float64

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

@register_plugin
class Manipulate(Plugin):
    def __init__(self, name, ctx, config):
        rospy.init_node("taskweaver_manipulate", anonymous=True)
        super().__init__(name=name, ctx=ctx, config=config)
        self.model = OctoModel.load_pretrained(self.config.get("octo_model", "hf://rail-berkeley/octo-small"))
        self.recognitions_subscriber = rospy.Subscriber(self.config.get("recognitions3d_topic", "/butia_vision/br/object_recognition3d"), Recognitions3D, callback=self._update_recognitions)
        self._recognitions3d: Recognitions3D = None
        self.bot = InterbotixManipulatorXS(
            self.config.get("arm_model", "doris_arm"),
            self.config.get("arm_group", "arm"),
            self.config.get("gripper_group", "gripper")
        )
        self.ctrl_rate = rospy.Rate(self.config.get("control_rate", 10))
        self.neck_pan_publisher = rospy.Publisher(self.config.get("neck_pan_topic", "/doris_head/head_pan_position_controller/command"), Float64)
        self.neck_tilt_publisher = rospy.Publisher(self.config.get("neck_tilt_topic", "/doris_head/head_tilt_position_controller/command"), Float64)

    def __call__(
        self,
        manipulation_command: str
    ):
        self.neck_pan_publisher.publish(self.config.get("pan_angle", 0.0))
        self.neck_tilt_publisher.publish(self.config.get("tilt_angle", -np.pi/8))
        task = self.model.create_tasks(texts=[manipulation_command])
        imgs = None
        self.bot.arm.go_to_home_pose()
        previous_action = None
        for i in range(self.config.get("n_steps", 50)):
            img = numpify(self.recognitions3d.image_rgb)
            img = img[np.newaxis,np.newaxis,...]
            if imgs == None:
                imgs = img
            else:
                imgs = np.concatenate([imgs, img], axis=1)
            if imgs.shape[1] >= self.config.get("window_size", 2):
                imgs = imgs[...,-self.config.get("window_size", 2):,...]
            observation = {
                "image_primary": imgs,
                "pas_mask": np.full((1, imgs.shape[1]), True, dtype=bool)
            }
            norm_actions = self.model.sample_actions(observation, task, rng=jax.random.PRNGKey(0))
            norm_actions = norm_actions[0]
            actions = (
                norm_actions * self.model.dataset_statistics["bridge_dataset"]["action"]["std"]
                + self.model.dataset_statistics["bridge_dataset"]["action"]["mean"]
            )
            rospy.loginfo(actions)
            for action in actions:
                if action == previous_action:
                    return
                current_ee_pose_matrix = self.bot.arm.get_ee_pose()
                rotation_matrix = current_ee_pose_matrix[:3,:3]
                roll, pitch, yaw = ang.rotationMatrixToEulerAngles(rotation_matrix)
                x, y, z = current_ee_pose_matrix[:3,3].flatten()
                x += action[0]
                y += action[1]
                z += action[2]
                roll += action[3]
                pitch += action[4]
                roll = roll % 2*np.pi
                pitch = pitch % 2*np.pi
                self.bot.arm.set_ee_pose_components(x=x, y=y, z=z, roll=roll, pitch=pitch, blocking=False)
                if action[6] >= 0.5:
                    self.bot.gripper.close(delay=0.0)
                else:
                    self.bot.gripper.open(delay=0.0)
                self.ctrl_rate.sleep()
                previous_action = action

    def _update_recognitions(self, msg: Recognitions3D):
        self._recognitions3d = msg

    @property
    def recognitions3d(self):
        return deepcopy(self._recognitions3d)

@test_plugin(name="test Manipulate", description="test")
def test_call(api_call):
    api_call(manipulation_command="grasp the pringles")