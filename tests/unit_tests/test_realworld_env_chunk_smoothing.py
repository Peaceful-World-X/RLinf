# Copyright 2026 The RLinf Authors.
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

"""Unit test: RealWorldEnv.chunk_step() calls process_action_chunk on sub-envs that support it."""

import sys
from unittest.mock import MagicMock

# Mock rospy before any realworld imports happen
if "rospy" not in sys.modules:
    sys.modules["rospy"] = MagicMock()

import numpy as np


def _make_mock_realworld_env(num_envs=1):
    """Build a minimal RealWorldEnv-like object with a mocked NoAutoResetSyncVectorEnv."""
    # Mock sub-env with process_action_chunk
    sub_env = MagicMock()
    sub_env.process_action_chunk.side_effect = (
        lambda x: x * 0.9
    )  # returns modified array

    # Mock venv
    venv = MagicMock()
    venv.envs = [sub_env]

    # Import here so the test file can be collected even without full ROS deps
    from rlinf.envs.realworld.realworld_env import RealWorldEnv

    env = object.__new__(RealWorldEnv)
    env.env = venv
    env.num_envs = num_envs
    env.break_chunk_on_intervention_change = False
    env.save_substep_obs = False
    env.auto_reset = False
    env.ignore_terminations = False
    env.manual_episode_control_only = True
    env._elapsed_steps = 0
    env.cfg = MagicMock()
    env.cfg.max_episode_steps = 10000

    import torch

    def fake_step(actions, auto_reset=True):
        obs = {}
        reward = torch.zeros(num_envs)
        term = torch.zeros(num_envs, dtype=torch.bool)
        trunc = torch.zeros(num_envs, dtype=torch.bool)
        info = {"chunk_valid_mask": None}
        return obs, reward, term, trunc, info

    env.step = fake_step
    env._record_metrics = MagicMock(return_value={})
    return env, sub_env


def test_chunk_step_calls_process_action_chunk():
    """process_action_chunk must be called once per sub-env per chunk_step call."""
    env, sub_env = _make_mock_realworld_env()
    chunk_actions = np.ones((1, 10, 14), dtype=np.float64)

    env.chunk_step(chunk_actions)

    assert sub_env.process_action_chunk.call_count == 1
    called_arg = sub_env.process_action_chunk.call_args[0][0]
    np.testing.assert_array_equal(called_arg, chunk_actions[0])


def test_chunk_step_uses_processed_actions():
    """chunk_step must feed the processed (smoothed) actions into per-step execution, not the originals."""
    env, sub_env = _make_mock_realworld_env()
    chunk_actions = np.ones((1, 10, 14), dtype=np.float64)

    received_actions = []
    original_step = env.step

    def capturing_step(actions, auto_reset=True):
        received_actions.append(
            actions.copy() if hasattr(actions, "copy") else np.array(actions)
        )
        return original_step(actions, auto_reset=auto_reset)

    env.step = capturing_step
    env.chunk_step(chunk_actions)

    # sub_env.process_action_chunk returns x * 0.9, so all received actions should be ~0.9
    for a in received_actions:
        assert np.allclose(a, 0.9), f"Expected smoothed value 0.9, got {a}"


def test_chunk_step_skips_envs_without_process_action_chunk():
    """Envs lacking process_action_chunk must pass through unchanged (no AttributeError)."""
    env, sub_env = _make_mock_realworld_env()
    del sub_env.process_action_chunk  # remove the method

    chunk_actions = np.ones((1, 10, 14), dtype=np.float64)

    received_actions = []
    original_step = env.step

    def capturing_step(actions, auto_reset=True):
        received_actions.append(
            actions.copy() if hasattr(actions, "copy") else np.array(actions)
        )
        return original_step(actions, auto_reset=auto_reset)

    env.step = capturing_step
    # Must not raise
    env.chunk_step(chunk_actions)

    for a in received_actions:
        assert np.allclose(a, 1.0), f"Expected original value 1.0, got {a}"
