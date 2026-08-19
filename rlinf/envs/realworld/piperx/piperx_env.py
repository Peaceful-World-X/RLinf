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

"""PiperX dual-arm environment using the existing HIL ROS interfaces."""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Header

from rlinf.envs.realworld.piper.piper_env import PiperEnv, PiperRobotConfig

from .piperx_controller import PiperXController


@dataclass
class PiperXRobotConfig(PiperRobotConfig):
    """Topics and task settings for ``intervene_piper_x.launch``."""

    task_name: str = "Open the box, put two items in, then close it."
    chunk_size: int = 50
    pos_lookahead_step: int = 50
    step_frequency: float = 30.0
    publish_rate: int = 30
    continuous_publish_timing: bool = True
    inference_trigger_remaining: int = 0
    align_search_window: int = 0
    continuity_blend_steps: int = 0
    joint_action_mode: str = "absolute"
    enable_human_intervention: bool = True

    left_state_topic: str = "/puppet/joint_left"
    right_state_topic: str = "/puppet/joint_right"
    left_end_pose_topic: str = "/puppet/end_pose_left"
    right_end_pose_topic: str = "/puppet/end_pose_right"
    left_policy_topic: str = "/policy/joint_left"
    right_policy_topic: str = "/policy/joint_right"
    intervention_mode_topic: str = "/piper/intervene_mode"
    subtask_reward_topic: str = "/piper/subtask_reward"
    subtask_reward_value: float = 0.0
    publish_policy_actions: bool = False
    external_reset: bool = True
    wait_enter_after_reset: bool = True
    reset_to_zero: bool = True
    actor_enabled_at_episode_start: bool = True
    pause_poll_sec: float = 0.05


class PiperXEnv(PiperEnv):
    """Piper environment whose ROS lifecycle is managed outside RLinf."""

    def __init__(
        self,
        override_cfg: dict | None = None,
        worker_info=None,
        hardware_info=None,
        env_idx: int = 0,
        **kwargs,
    ) -> None:
        config = PiperXRobotConfig(**(override_cfg or {}))
        self._piperx_mode_lock = threading.Lock()
        self._piperx_intervening = False
        self._piperx_reward_lock = threading.Lock()
        self._pending_subtask_rewards = 0
        self._subtask_reward_event_count = 0
        self._piperx_paused = False
        self._piperx_actor_enabled = bool(config.actor_enabled_at_episode_start)
        self._piperx_force_actor = False
        self._piperx_startup_ready_logged = False
        self._operator_terminal_in = None
        self._operator_terminal_out = None
        self._operator_key_fd = None
        self._operator_keys = deque()
        self._piperx_image_history = {
            cam_name: deque(maxlen=2000) for cam_name in config.camera_names
        }
        terminal_path = os.environ.get("RLINF_PIPERX_TERMINAL")
        if terminal_path:
            self._operator_terminal_out = open(terminal_path, "w", buffering=1)
        key_fifo = os.environ.get("RLINF_PIPERX_KEY_FIFO")
        if key_fifo:
            self._operator_key_fd = os.open(key_fifo, os.O_RDONLY | os.O_NONBLOCK)
        super().__init__(config, worker_info, hardware_info, env_idx)
        self._piperx_subscribers = []
        if not config.is_dummy:
            import rospy

            self._piperx_subscribers = [
                rospy.Subscriber(
                    config.intervention_mode_topic,
                    Bool,
                    self._on_intervention_mode,
                    queue_size=1,
                ),
                rospy.Subscriber(
                    config.subtask_reward_topic,
                    Header,
                    self._on_subtask_reward,
                    queue_size=20,
                ),
            ]

    def _make_img_callback(self, cam_name: str):
        """Keep native geometry and match the reference client's RGB decoding."""

        def _callback(msg: Image) -> None:
            try:
                image = self._bridge.imgmsg_to_cv2(msg, "rgb8")
                with self._img_lock:
                    self._latest_images[cam_name] = image
                    self._piperx_image_history[cam_name].append(
                        (msg.header.stamp.to_sec(), image)
                    )
            except Exception as exc:
                self._logger.warning(f"Image callback error ({cam_name}): {exc}")

        return _callback

    def _get_observation(self) -> dict:
        """Build a timestamp-aligned image and dual-arm observation."""
        if self.config.is_dummy:
            return super()._get_observation()

        deadline = time.monotonic() + 5.0
        while True:
            with self._img_lock:
                histories = [
                    self._piperx_image_history[name]
                    for name in self.config.camera_names
                ]
                if all(histories):
                    frame_time = min(history[-1][0] for history in histories)
                    selected = []
                    for history in histories:
                        frame = next(
                            (image for stamp, image in history if stamp >= frame_time),
                            None,
                        )
                        if frame is None:
                            break
                        selected.append(frame.copy())
                    if len(selected) == len(histories):
                        frames = dict(zip(self.config.camera_names, selected))
                    else:
                        frames = None
                else:
                    frame_time = None
                    frames = None

            qpos = (
                None
                if frame_time is None
                else self._controller.get_synchronized_qpos(frame_time)
            )
            if frames is not None and qpos is not None:
                return {"state": {"qpos": qpos}, "frames": frames}
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "Timed out waiting for timestamp-aligned PiperX images and joint states"
                )
            time.sleep(0.005)

    def _create_controller(self) -> PiperXController:
        cfg = self.config
        return PiperXController(
            left_state_topic=cfg.left_state_topic,
            right_state_topic=cfg.right_state_topic,
            left_end_pose_topic=cfg.left_end_pose_topic,
            right_end_pose_topic=cfg.right_end_pose_topic,
            left_policy_topic=cfg.left_policy_topic,
            right_policy_topic=cfg.right_policy_topic,
            use_robot_base=cfg.use_robot_base,
            robot_base_topic=cfg.robot_base_topic,
            joint_speed_pct=cfg.joint_speed_pct,
            publish_policy_actions=cfg.publish_policy_actions,
        )

    def _on_intervention_mode(self, msg: Bool) -> None:
        with self._piperx_mode_lock:
            self._piperx_intervening = bool(msg.data)

    def _on_subtask_reward(self, _msg: Header) -> None:
        with self._piperx_reward_lock:
            self._pending_subtask_rewards += 1
            self._subtask_reward_event_count += 1
            event_index = self._subtask_reward_event_count
        self._operator_print(
            "[PIPERX REWARD PEDAL] "
            f"topic={self.config.subtask_reward_topic} event={event_index} "
            f"reward={float(self.config.subtask_reward_value):.6f}"
        )

    def _teleop_active(self) -> bool:
        if self.config.is_dummy:
            return False
        with self._piperx_mode_lock:
            return self._piperx_intervening

    def _calc_step_reward(self, observation: dict) -> float:
        del observation
        with self._piperx_reward_lock:
            if self._pending_subtask_rewards <= 0:
                return 0.0
            self._pending_subtask_rewards -= 1
        return float(self.config.subtask_reward_value)

    def _consume_control_key(self, keys) -> str | None:
        if self._operator_key_fd is not None:
            try:
                pending = os.read(self._operator_key_fd, 64)
            except BlockingIOError:
                pending = b""
            key_names = {
                ord("\n"): "Key.enter",
                ord(" "): "Key.space",
                ord("q"): "q",
                ord("x"): "x",
                ord("f"): "f",
                ord("0"): "0",
                ord("1"): "1",
            }
            self._operator_keys.extend(
                key_names[value] for value in pending if value in key_names
            )
            for index, key in enumerate(self._operator_keys):
                if key in keys:
                    del self._operator_keys[index]
                    return key
            return None
        if self._keyboard is None:
            return None
        return self._keyboard.consume_any_press(keys, include_held=False)

    def _clear_control_keys(self) -> None:
        if self._keyboard is not None:
            self._keyboard.clear_keys()
        self._operator_keys.clear()
        if self._operator_key_fd is not None:
            while True:
                try:
                    if not os.read(self._operator_key_fd, 64):
                        break
                except BlockingIOError:
                    break

    def _operator_print(self, message: str) -> None:
        print(message, flush=True)
        if self._operator_terminal_out is not None:
            print(message, file=self._operator_terminal_out, flush=True)

    def _wait_for_episode_start(self) -> None:
        if self.config.is_dummy or not self.config.wait_enter_after_reset:
            return
        self._clear_control_keys()
        self._operator_print(
            "\n"
            "################################################################\n"
            "### PIPERX ROBOT IS ZEROED - PRESS ENTER TO START EPISODE   ###\n"
            "################################################################"
        )
        poll = max(float(self.config.pause_poll_sec), 0.01)
        while self._consume_control_key(("Key.enter",)) is None:
            time.sleep(poll)
        closed = np.zeros(7, dtype=np.float64)
        self._move_to_home_continuous(closed, closed)
        self._operator_print("[PIPERX START] Episode execution started.")

    def _move_to_home_continuous(
        self, left_target: np.ndarray, right_target: np.ndarray
    ) -> None:
        """Match the reference PiperX client's bounded-step reset trajectory."""
        current = np.asarray(self._controller.get_qpos(), dtype=np.float64).reshape(
            2, 7
        )
        targets = np.stack((left_target, right_target)).astype(np.float64)
        max_step = np.asarray([0.01] * 6 + [0.2], dtype=np.float64)
        sleep_sec = 1.0 / max(float(self.config.publish_rate), 1.0)
        while True:
            delta = targets - current
            complete = bool(np.all(np.abs(delta) <= 1e-6))
            current += np.clip(delta, -max_step, max_step)
            self._controller.move_arm(current[0], current[1])
            if complete:
                return
            time.sleep(sleep_sec)

    def _wait_while_paused(self) -> str:
        poll = max(float(self.config.pause_poll_sec), 0.01)
        self._operator_print("[PIPERX PAUSED] Press Space to resume or q to end.")
        while True:
            key = self._consume_control_key(("Key.space", "q", "x", "f"))
            if key == "x":
                self._piperx_actor_enabled = not self._piperx_actor_enabled
                owner = "ACTOR" if self._piperx_actor_enabled else "BASEPOLICY"
                self._operator_print(f"[PIPERX OWNER] {owner}")
            elif key == "f":
                self._piperx_force_actor = not self._piperx_force_actor
                mode = "ON" if self._piperx_force_actor else "OFF"
                self._operator_print(f"[PIPERX FORCE ACTOR] {mode}")
            elif key == "q":
                return "quit"
            elif key == "Key.space":
                self._piperx_paused = False
                self._operator_print(
                    "[PIPERX RESUMED] Discarding the old action chunk and requesting "
                    "new inference from the current state."
                )
                return "resume"
            time.sleep(poll)

    def _wait_for_episode_result(self) -> float:
        self._operator_print(
            "[PIPERX END] Episode stopped. Press 1 for success or 0 for failure."
        )
        poll = max(float(self.config.pause_poll_sec), 0.01)
        while True:
            key = self._consume_control_key(("1", "0"))
            if key is not None:
                reward = float(key == "1")
                self._operator_print(
                    f"[PIPERX RESULT] "
                    f"{'SUCCESS' if reward else 'FAILURE'} reward={reward:.1f}"
                )
                return reward
            time.sleep(poll)

    def step(self, action: np.ndarray):
        """Handle PiperX episode keys before publishing the next action."""
        key = self._consume_control_key(("Key.space", "q", "x", "f"))
        if key == "f":
            self._piperx_force_actor = not self._piperx_force_actor
            mode = "ON" if self._piperx_force_actor else "OFF"
            self._operator_print(f"[PIPERX FORCE ACTOR] {mode}")
            return (
                self._get_observation(),
                0.0,
                False,
                False,
                {
                    "piperx_break_chunk": True,
                    "piperx_actor_enabled": self._piperx_actor_enabled,
                    "piperx_force_actor": self._piperx_force_actor,
                    "piperx_action_executed": False,
                },
            )
        if key == "x":
            self._piperx_actor_enabled = not self._piperx_actor_enabled
            owner = "ACTOR" if self._piperx_actor_enabled else "BASEPOLICY"
            self._operator_print(f"[PIPERX OWNER] {owner}")
            observation = self._get_observation()
            return (
                observation,
                0.0,
                False,
                False,
                {
                    "piperx_break_chunk": True,
                    "piperx_actor_enabled": self._piperx_actor_enabled,
                    "piperx_force_actor": self._piperx_force_actor,
                    "piperx_action_executed": False,
                },
            )
        if key == "Key.space":
            self._piperx_paused = True
            outcome = self._wait_while_paused()
            if outcome == "quit":
                reward = self._wait_for_episode_result()
                return (
                    self._get_observation(),
                    reward,
                    True,
                    False,
                    {
                        "piperx_actor_enabled": self._piperx_actor_enabled,
                        "piperx_force_actor": self._piperx_force_actor,
                    },
                )
            return (
                self._get_observation(),
                0.0,
                False,
                False,
                {
                    "piperx_break_chunk": True,
                    "piperx_actor_enabled": self._piperx_actor_enabled,
                    "piperx_force_actor": self._piperx_force_actor,
                    "piperx_action_executed": False,
                },
            )
        if key == "q":
            reward = self._wait_for_episode_result()
            return (
                self._get_observation(),
                reward,
                True,
                False,
                {
                    "piperx_actor_enabled": self._piperx_actor_enabled,
                    "piperx_force_actor": self._piperx_force_actor,
                },
            )

        observation, reward, terminated, truncated, info = super().step(action)
        info["piperx_actor_enabled"] = self._piperx_actor_enabled
        info["piperx_force_actor"] = self._piperx_force_actor
        info["piperx_action_executed"] = True
        return observation, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        """Reset RL episode state without managing externally owned hardware."""
        if self.config.is_dummy or not self.config.external_reset:
            return super().reset(seed=seed, options=options)
        self._num_steps = 0
        self._success_hold_counter = 0
        self._sliding_window_action_buffer = None
        self._last_teleop_active = self._teleop_active()
        self._piperx_paused = False
        self._last_action_publish_time = None
        self._piperx_actor_enabled = bool(self.config.actor_enabled_at_episode_start)
        with self._piperx_reward_lock:
            self._pending_subtask_rewards = 0
        if self.config.publish_policy_actions:
            self._wait_for_teleop_release_before_reset()
        if not self._piperx_startup_ready_logged:
            self._operator_print(
                "\n"
                "################################################################\n"
                "### PIPERX ONLINE RL IS READY - FIRST EPISODE STARTING NOW  ###\n"
                "### ROBOT ZEROING STARTS NEXT, THEN EXECUTION WAITS FOR ENTER ###\n"
                "################################################################"
            )
            self._piperx_startup_ready_logged = True
        if self.config.reset_to_zero:
            home = np.asarray([0.0] * 6 + [0.3], dtype=np.float64)
            self._operator_print("[PIPERX RESET] Moving both arms to zero.")
            self._move_to_home_continuous(home, home)
            self._controller._wait_for_joint(
                target_pos=home[:6], side="left", timeout=10.0, atol=0.05
            )
            self._controller._wait_for_joint(
                target_pos=home[:6], side="right", timeout=10.0, atol=0.05
            )
            self._operator_print("[PIPERX RESET] Both arms reached zero.")
        self._wait_for_episode_start()
        return self._get_observation(), {
            "piperx_actor_enabled": self._piperx_actor_enabled
        }

    def close(self) -> None:
        for subscriber in getattr(self, "_piperx_subscribers", []):
            subscriber.unregister()
        self._piperx_subscribers = []
        if self._operator_key_fd is not None:
            os.close(self._operator_key_fd)
            self._operator_key_fd = None
        if self._operator_terminal_out is not None:
            self._operator_terminal_out.close()
            self._operator_terminal_out = None
        super().close()

    @property
    def task_description(self) -> str:
        return self.config.task_name
