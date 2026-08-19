# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Controller for the PiperX ROS topics exposed by intervene_piper_x.launch."""

from __future__ import annotations

import threading
from collections import deque

import numpy as np
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState

from rlinf.envs.realworld.piper.piper_controller import PiperController


class PiperXController(PiperController):
    """Connect RLinf to an already-running PiperX HIL ROS stack.

    This class never starts, stops, enables, or resets ROS hardware. The external
    ``intervene_piper_x.launch`` process owns the robot lifecycle.
    """

    def __init__(
        self,
        *,
        left_state_topic: str = "/puppet/joint_left",
        right_state_topic: str = "/puppet/joint_right",
        left_end_pose_topic: str = "/puppet/end_pose_left",
        right_end_pose_topic: str = "/puppet/end_pose_right",
        left_policy_topic: str = "/policy/joint_left",
        right_policy_topic: str = "/policy/joint_right",
        use_robot_base: bool = False,
        robot_base_topic: str = "/odom",
        joint_speed_pct: int = 50,
        publish_policy_actions: bool = False,
    ) -> None:
        self._joint_history_lock = threading.Lock()
        self._left_joint_history = deque(maxlen=2000)
        self._right_joint_history = deque(maxlen=2000)
        self._publish_policy_actions = bool(publish_policy_actions)
        self._piperx_topics = {
            "left_state": left_state_topic,
            "right_state": right_state_topic,
            "left_end_pose": left_end_pose_topic,
            "right_end_pose": right_end_pose_topic,
            "left_policy": left_policy_topic,
            "right_policy": right_policy_topic,
        }
        super().__init__(
            ns_left="/puppet_left",
            ns_right="/puppet_right",
            use_robot_base=use_robot_base,
            robot_base_topic=robot_base_topic,
            joint_speed_pct=joint_speed_pct,
        )

    @staticmethod
    def _joint_sample(msg: JointState) -> tuple[float, np.ndarray]:
        return msg.header.stamp.to_sec(), np.asarray(msg.position[:7], dtype=np.float64)

    def _on_left_joint_state(self, msg: JointState) -> None:
        with self._joint_history_lock:
            self._left_joint_history.append(self._joint_sample(msg))
        super()._on_left_joint_state(msg)

    def _on_right_joint_state(self, msg: JointState) -> None:
        with self._joint_history_lock:
            self._right_joint_history.append(self._joint_sample(msg))
        super()._on_right_joint_state(msg)

    def get_synchronized_qpos(self, frame_time: float) -> np.ndarray | None:
        """Return the first dual-arm sample at or after the image timestamp."""
        with self._joint_history_lock:
            samples = []
            for history in (self._left_joint_history, self._right_joint_history):
                sample = next(
                    (position for stamp, position in history if stamp >= frame_time),
                    None,
                )
                if sample is None:
                    return None
                samples.append(sample.copy())
        return np.concatenate(samples)

    def _publish_joint_ctrl(
        self, topic: str, action: np.ndarray, speed_pct: int
    ) -> None:
        """Publish only when hardware action output is explicitly enabled."""
        if not self._publish_policy_actions:
            return
        super()._publish_joint_ctrl(topic, action, speed_pct)

    def move_arm(
        self,
        left_action: np.ndarray,
        right_action: np.ndarray,
        left_speed_pct: int | None = None,
        right_speed_pct: int | None = None,
    ) -> None:
        """Publish PiperX's native 7-D positions without legacy gripper clipping."""
        left_action = np.asarray(left_action, dtype=np.float64)
        right_action = np.asarray(right_action, dtype=np.float64)
        if left_action.shape != (7,) or right_action.shape != (7,):
            raise ValueError("PiperX arm actions must each contain 7 values")
        self._publish_joint_ctrl(
            self._left_joint_ctrl_topic,
            left_action,
            self._joint_speed_pct if left_speed_pct is None else left_speed_pct,
        )
        self._publish_joint_ctrl(
            self._right_joint_ctrl_topic,
            right_action,
            self._joint_speed_pct if right_speed_pct is None else right_speed_pct,
        )

    def _init_ros_channels(self) -> None:
        topics = self._piperx_topics
        self._left_joint_state_topic = topics["left_state"]
        self._right_joint_state_topic = topics["right_state"]
        self._left_end_pose_topic = topics["left_end_pose"]
        self._right_end_pose_topic = topics["right_end_pose"]
        self._left_joint_ctrl_topic = topics["left_policy"]
        self._right_joint_ctrl_topic = topics["right_policy"]

        self._ros.connect_ros_channel(
            self._left_joint_state_topic, JointState, self._on_left_joint_state
        )
        self._ros.connect_ros_channel(
            self._right_joint_state_topic, JointState, self._on_right_joint_state
        )
        self._ros.connect_ros_channel(
            self._left_end_pose_topic, PoseStamped, self._on_left_end_pose
        )
        self._ros.connect_ros_channel(
            self._right_end_pose_topic, PoseStamped, self._on_right_end_pose
        )
        if self._use_robot_base:
            self._ros.connect_ros_channel(
                self._robot_base_topic, Odometry, self._on_robot_base
            )
        self._ros.create_ros_channel(
            self._left_joint_ctrl_topic, JointState, queue_size=1
        )
        self._ros.create_ros_channel(
            self._right_joint_ctrl_topic, JointState, queue_size=1
        )

    def enable_arm(self, ns=None) -> bool:
        """Leave enable state under the external PiperX launch process."""
        return True

    def disable_arm(self, ns=None) -> bool:
        """Leave disable state under the external PiperX launch process."""
        return True

    def go_zero(self, is_mit_mode: bool = False, ns=None) -> bool:
        """Do not issue lifecycle or reset commands to PiperX hardware."""
        del is_mit_mode, ns
        return True

    def stop_arm(self, ns=None) -> bool:
        """Do not stop the externally managed PiperX ROS process."""
        return True

    def reset_arm(self, ns=None) -> bool:
        """Do not reset the externally managed PiperX ROS process."""
        return True
