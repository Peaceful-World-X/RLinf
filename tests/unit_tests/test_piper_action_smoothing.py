# Copyright 2026 The GIGA Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
from unittest.mock import MagicMock

import numpy as np

if "rospy" not in sys.modules:
    sys.modules["rospy"] = MagicMock()
if "cv_bridge" not in sys.modules:
    cv_bridge = MagicMock()
    cv_bridge.CvBridge = MagicMock
    sys.modules["cv_bridge"] = cv_bridge
if "sensor_msgs" not in sys.modules:
    sys.modules["sensor_msgs"] = MagicMock()
if "sensor_msgs.msg" not in sys.modules:
    msg = MagicMock()
    msg.Image = MagicMock
    msg.JointState = MagicMock
    sys.modules["sensor_msgs.msg"] = msg
if "geometry_msgs" not in sys.modules:
    sys.modules["geometry_msgs"] = MagicMock()
if "geometry_msgs.msg" not in sys.modules:
    msg = MagicMock()
    msg.PoseStamped = MagicMock
    sys.modules["geometry_msgs.msg"] = msg
if "nav_msgs" not in sys.modules:
    sys.modules["nav_msgs"] = MagicMock()
if "nav_msgs.msg" not in sys.modules:
    msg = MagicMock()
    msg.Odometry = MagicMock
    sys.modules["nav_msgs.msg"] = msg

from rlinf.envs.realworld.piper.piper_env import PiperEnv, PiperRobotConfig


def _make_smoothing_env():
    env = object.__new__(PiperEnv)
    cfg = PiperRobotConfig(is_dummy=True)
    cfg.sliding_window_action_buffer = True
    cfg.smooth_action_chunk = True
    cfg.action_smooth_cutoff_freq = 1.0
    cfg.action_smooth_sampling_freq = 25.0
    cfg.action_smooth_order = 2
    cfg.sliding_window_inference_trigger_remaining = 10
    cfg.sliding_window_max_action_execute_horizon = 35
    cfg.sliding_window_distance_thresh = 0.5
    cfg.sliding_window_latency_steps = 0
    env.config = cfg
    env._sliding_window_action_buffer = None
    return env


def test_default_smooth_sampling_freq_matches_control_frequency():
    cfg = PiperRobotConfig(is_dummy=True)

    assert cfg.action_smooth_sampling_freq == cfg.step_frequency


def test_sliding_window_blends_short_15_step_chunk_boundary():
    env = _make_smoothing_env()
    first = np.zeros((15, 14), dtype=np.float64)
    second = np.ones((15, 14), dtype=np.float64)

    env.process_action_chunk(first)
    processed = env.process_action_chunk(second)

    assert np.all(processed[0, :6] < 0.5)
    assert np.all(processed[1, :6] > processed[0, :6])
    assert np.all(processed[-1, :6] > 0.9)


def test_boundary_blend_preserves_grippers():
    env = _make_smoothing_env()
    previous_action = np.zeros(14, dtype=np.float64)
    actions = np.ones((15, 14), dtype=np.float64)

    blended = env._blend_action_chunk_start(actions, previous_action)

    np.testing.assert_allclose(blended[:5, 0], np.linspace(1 / 6, 5 / 6, 5))
    np.testing.assert_allclose(blended[:5, 6], 1.0)
    np.testing.assert_allclose(blended[:5, 13], 1.0)
