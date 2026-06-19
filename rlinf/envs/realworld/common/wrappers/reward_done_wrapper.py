# Copyright 2025 The RLinf Authors.
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

import time
from typing import Any, SupportsFloat

import gymnasium as gym
from gymnasium.core import ActType, ObsType

from rlinf.envs.realworld.common.keyboard.keyboard_listener import KeyboardListener


class BaseKeyboardRewardDoneWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, reward_mode: str = "always_replace"):
        super().__init__(env)
        self.reward_modifier = 0
        self.listener = KeyboardListener()
        self.reward_mode = reward_mode
        self._reset_armed = False
        self._reset_enter_keys = ("Key.enter", "Key.kpenter", "enter")
        assert self.reward_mode in ["always_replace"]

    def _check_keypress(self) -> tuple[bool, bool, float]:
        raise NotImplementedError

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Modifies the :attr:`env` :meth:`step` reward using :meth:`self.reward`."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        last_intervened, updated_reward, updated_terminated = self.reward_terminated()
        if last_intervened or self.reward_mode == "always_replace":
            reward = updated_reward
        return observation, reward, updated_terminated, truncated, info

    def reward_terminated(
        self,
    ) -> tuple[float, bool]:
        last_intervened, terminated, keyboard_reward = self._check_keypress()
        if terminated:
            self._reset_armed = True
            self._hold_current_position()
        return last_intervened, keyboard_reward, terminated

    def reset(self, *, seed=None, options=None):
        if self._reset_armed:
            self._wait_for_reset_confirmation()
        obs, info = super().reset(seed=seed, options=options)
        self.listener.clear_keys(("a", "b", "c", *self._reset_enter_keys))
        self._reset_armed = False
        return obs, info

    def _wait_for_reset_confirmation(self) -> None:
        print(
            "Episode ended by keyboard reward. Press Enter to reset/start next episode...",
            flush=True,
        )
        self.listener.clear_keys(self._reset_enter_keys)
        while (
            self.listener.consume_any_press(self._reset_enter_keys, include_held=False)
            is None
        ):
            time.sleep(0.05)
        print("Reset confirmation received from keyboard Enter.", flush=True)
        self._exit_human_intervention()
        self.listener.clear_keys(("a", "b", "c", *self._reset_enter_keys))

    def _exit_human_intervention(self) -> None:
        unwrapped = self.env.unwrapped
        if hasattr(unwrapped, "_policy_enabled"):
            unwrapped._policy_enabled = True
        try:
            import rospy

            rospy.set_param("/enable_message_publish", False)
        except Exception as exc:
            print(f"Failed to disable teleop on reset confirmation: {exc}", flush=True)

    def _hold_current_position(self) -> None:
        unwrapped = self.env.unwrapped
        if hasattr(unwrapped, "_policy_enabled"):
            unwrapped._policy_enabled = False
        controller = getattr(unwrapped, "_controller", None)
        config = getattr(unwrapped, "config", None)
        if controller is None or config is None or getattr(config, "is_dummy", False):
            return
        try:
            import rospy

            if bool(rospy.get_param("/enable_message_publish", False)):
                print(
                    "Keyboard reward event: teleop is active; skip policy hold target.",
                    flush=True,
                )
                return
            qpos = controller.get_qpos()
            controller.move_arm(qpos[:7], qpos[7:14])
            print("Keyboard reward event: holding current robot position.", flush=True)
        except Exception as exc:
            print(f"Failed to hold current robot position: {exc}", flush=True)


class KeyboardRewardDoneWrapper(BaseKeyboardRewardDoneWrapper):
    def _check_keypress(self) -> tuple[bool, bool, float]:
        last_intervened = False
        done = False
        reward = 0
        key = self.listener.consume_any_press(("a", "b", "c"), include_held=False)
        if key is not None:
            print(f"Key pressed: {key}")
        if key not in ["a", "b", "c"]:
            return last_intervened, done, reward

        last_intervened = True
        if key == "a":
            reward = 0
            done = True
            last_intervened = True
        elif key == "b":
            reward = 0
            last_intervened = True
        elif key == "c":
            reward = 1
            done = True
            last_intervened = True
        print(
            f"Keyboard reward event: key={key}, reward={reward}, done={done}",
            flush=True,
        )
        return last_intervened, done, reward


class KeyboardRewardDoneMultiStageWrapper(BaseKeyboardRewardDoneWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.stage_rewards = [0, 0.1, 1]

    def reset(self, *, seed=None, options=None):
        self.reward_stage = 0
        return super().reset(seed=seed, options=options)

    def _check_keypress(self) -> tuple[bool, bool, float]:
        last_intervened = False
        done = False
        reward = 0
        key = self.listener.consume_any_press(("a", "b", "c", "q"), include_held=False)
        if key is not None:
            print(f"Key pressed: {key}")
        if key == "a":
            self.reward_stage = 0
        elif key == "b":
            self.reward_stage = 1
        elif key == "c":
            self.reward_stage = 2

        if self.reward_stage == 2:
            done = True

        reward = self.stage_rewards[self.reward_stage]
        if key == "q":
            reward = -1
            done = False
        return last_intervened, done, reward
