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

"""Piper dual-arm robot Gym environment.

Unified RLinf interface coordinating ``PiperController`` and ``PiperRobotState``,
providing standard ``gymnasium.Env`` API (step / reset / observation_space / action_space).

Architecture aligned with ``rlinf.envs.realworld.franka.franka_env.FrankaEnv``.

**Action Space (Joint space):**

- 14D absolute joint positions: left arm 7D (6 joints + 1 gripper) + right arm 7D

**Observation Space:**

- ``state``: qpos(14), qvel(14), effort(14), base_vel(2)
- ``frames``: cam_high(480,640,3), cam_left_wrist(480,640,3), cam_right_wrist(480,640,3)

**Human-in-the-loop:**

- ``page_down``: toggle teleoperation (handled by ROS node via ``/enable_message_publish`` param)
- ``page_up``: toggle policy output; when disabled, env holds current qpos
"""

import copy
import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import cv2
import gymnasium as gym
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, JointState

from rlinf.utils.logging import get_logger

from .piper_controller import PiperController
from .utils import (
    split_dual_arm_action,
)

# =========================================================================
# Configuration
# =========================================================================


@dataclass
class PiperRobotConfig:
    """Piper dual-arm robot environment configuration.

    Attributes:
        ns_left: Left arm ROS namespace.
        ns_right: Right arm ROS namespace.
        use_robot_base: Whether to use mobile base.
        robot_base_topic: Mobile base odometry topic.
        camera_names: Camera name list.
        img_topics: Camera ROS topic list, one-to-one with camera_names.
        img_resolution: Camera image resolution (H, W).
        obs_img_resolution: Observation image resolution (H, W).
        step_frequency: Control frequency (Hz).
        publish_rate: Publishing frequency (Hz).
        joint_speed_pct: Joint motion speed percentage (0-100).
        is_dummy: Whether in dummy mode (no real hardware).
        max_num_steps: Maximum steps per episode.
        min_qpos: Per-arm joint lower limits (7D: 6 joints + 1 gripper).
        max_qpos: Per-arm joint upper limits.
        enable_camera_player: Whether to enable camera display.
        target_qpos: Target joint positions for reward (14D).
        reward_threshold: Per-joint error threshold for target zone.
        use_dense_reward / dense_reward_scale / success_hold_steps: See ``_calc_step_reward``.
        enable_human_intervention: Whether to enable human-in-the-loop intervention.
        intervention_trigger_key: Key that toggles teleoperation in the ROS node (page_down).
        policy_enable_key: Key that toggles policy output in piper_env (page_up).
        puppet_joint_topic_left: ROS topic for left puppet (slave) arm joint states (7D).
        puppet_joint_topic_right: ROS topic for right puppet (slave) arm joint states (7D).
        wait_teleop_release_before_reset: If True, ``reset`` waits until ``/enable_message_publish`` is False.
        teleop_release_poll_sec: Poll interval while waiting for teleop release.
        teleop_release_reset_timeout_sec: Max seconds to wait (None or <=0 = wait indefinitely).
        wait_enter_after_reset: If True, ``reset`` waits for Enter before returning.
    """

    # ---- ROS namespaces ----
    ns_left: str = "/puppet_left"
    ns_right: str = "/puppet_right"

    # ---- Mobile base ----
    use_robot_base: bool = False
    robot_base_topic: str = "/odom"

    # ---- Camera configuration ----
    camera_names: list[str] = field(
        default_factory=lambda: ["cam_high", "cam_left_wrist", "cam_right_wrist"]
    )
    img_topics: list[str] = field(
        default_factory=lambda: [
            "/camera_f/color/image_raw",
            "/camera_l/color/image_raw",
            "/camera_r/color/image_raw",
        ]
    )
    img_resolution: tuple[int, int] = (480, 640)  # (H, W) raw image resolution
    obs_img_resolution: tuple[int, int] = (480, 640)  # (H, W) observation resolution

    # ---- Control parameters ----
    step_frequency: float = 25.0  # Hz
    publish_rate: int = 25
    joint_speed_pct: int = 50
    pos_lookahead_step: int = 25
    chunk_size: int = 50
    smooth_action_chunk: bool = True
    action_smooth_cutoff_freq: float = 1.0
    action_smooth_sampling_freq: float = 25.0
    action_smooth_order: int = 2
    sliding_window_action_buffer: bool = True
    sliding_window_inference_trigger_remaining: int = 10
    sliding_window_max_action_execute_horizon: int = 35
    sliding_window_distance_thresh: float = 0.5
    sliding_window_latency_steps: int = 0
    sliding_window_boundary_blend_steps: int = 5
    gripper_action_threshold: Optional[float] = 0.03
    gripper_action_scale: float = 1.1

    # ---- Environment parameters ----
    task_name: str = "task"
    is_dummy: bool = False
    max_num_steps: int = 10000
    enable_camera_player: bool = False

    # ---- Joint limits (per arm: 6 joints + 1 gripper) ----
    min_qpos: list[float] = field(
        default_factory=lambda: [
            -2.618,
            0.0,
            -2.967,
            -1.745,
            -1.22,
            -2.7925,
            0.0,
        ]
    )
    max_qpos: list[float] = field(
        default_factory=lambda: [
            2.618,
            3.14,
            0.0,
            1.745,
            1.22,
            2.7925,
            0.07,
        ]
    )

    # ---- Human-in-the-loop configuration ----
    enable_human_intervention: bool = True
    # page_down: handled by ROS node, toggles /enable_message_publish param
    intervention_trigger_key: str = "Key.pagedown"
    # page_up: toggles policy output inside piper_env
    policy_enable_key: str = "Key.pageup"
    # Topics publishing puppet (slave) arm actual joint states (7D each).
    # During teleoperation, these reflect what the slave arms are actually doing,
    # which is what we record as the intervene_action for imitation learning.
    # /puppet/joint_left and /puppet/joint_right are remapped from /puppet/joint_states
    # in start_ms_piper_double_agilex_delta_qpose.launch.
    puppet_joint_topic_left: str = "/puppet/joint_left"
    puppet_joint_topic_right: str = "/puppet/joint_right"
    # Before reset: block until teleop is off (same ROS param as step ``teleop_active``),
    # so ``move_arm`` does not fight master arms during human intervention.
    wait_teleop_release_before_reset: bool = True
    teleop_release_poll_sec: float = 0.1
    # None or <=0: wait indefinitely until teleop releases.
    teleop_release_reset_timeout_sec: Optional[float] = None
    wait_enter_after_reset: bool = False
    use_tcp_reset_pose_pool: bool = False
    tcp_reset_pose_pool_path: str = ""
    tcp_reset_random_seed: int = 1234
    tcp_reset_interpolate_steps: int = 20
    tcp_reset_interpolate_sleep: Optional[float] = None

    # ---- ZMQ inference service (reserved) ----
    inference_host: str = "127.0.0.1"
    inference_port: int = 8080

    # ---- Joint action semantics ----
    # absolute: policy outputs absolute joint positions
    # delta: policy outputs [-1,1]^14, scaled by delta_action_scale and added to current qpos
    joint_action_mode: str = "absolute"
    delta_action_scale: float = 0.05

    # ---- Reward (joint space, aligned with FrankaEnv) ----
    # Target joint positions (14D: left 7 + right 7), calibrate per task.
    target_qpos: np.ndarray = field(
        default_factory=lambda: np.zeros(14, dtype=np.float64)
    )
    # Per-joint absolute error threshold; all joints within threshold = target zone.
    reward_threshold: np.ndarray = field(
        default_factory=lambda: np.full(14, 0.1, dtype=np.float64)
    )
    use_dense_reward: bool = False
    # Hold success_hold_steps consecutive steps in target zone to terminate with success.
    success_hold_steps: int = 1
    # Dense reward: exp(-dense_reward_scale * sum_i (qpos_i - target_qpos_i)^2)
    dense_reward_scale: float = 50.0


# =========================================================================
# Environment
# =========================================================================


class PiperEnv(gym.Env):
    """Piper dual-arm robot Gymnasium environment.

    Aligned with ``FrankaEnv`` interface. Communicates with piper_ros via
    ``PiperController``. Images are received via ROS topics using ``cv_bridge``.

    **Data flow:**

    1. ``__init__``: create PiperController → subscribe joint/pose/image topics → wait for robot
    2. ``step(action)``: split 14D action → clip → publish joint command → rate control → return obs
    3. ``reset()``: enable → go_zero → wait for joint → return initial obs
    4. ``_get_observation()``: concatenate qpos/qvel/effort/base_vel + camera images

    Args:
        config: PiperRobotConfig instance.
        worker_info: RLinf WorkerInfo (optional).
        hardware_info: Hardware info (optional, reserved).
        env_idx: Environment instance index.
    """

    def __init__(
        self,
        config: PiperRobotConfig,
        worker_info: Any = None,
        hardware_info: Any = None,
        env_idx: int = 0,
    ) -> None:
        super().__init__()
        self._logger = get_logger()
        self.config = config
        self.env_idx = env_idx
        self._num_steps = 0
        self._success_hold_counter = 0
        self._sliding_window_action_buffer = None
        self._last_teleop_active = False
        self._ensure_reward_config_arrays()
        self._tcp_reset_rng = np.random.default_rng(
            int(getattr(config, "tcp_reset_random_seed", 1234))
        )
        self._tcp_reset_pose_pool: list[dict[str, Any]] = []
        if getattr(config, "use_tcp_reset_pose_pool", False):
            self._tcp_reset_pose_pool = self._load_tcp_reset_pose_pool(
                getattr(config, "tcp_reset_pose_pool_path", "")
            )

        # ---- Initialize controller ----
        if not self.config.is_dummy:
            self._controller = PiperController(
                ns_left=config.ns_left,
                ns_right=config.ns_right,
                use_robot_base=config.use_robot_base,
                robot_base_topic=config.robot_base_topic,
                joint_speed_pct=config.joint_speed_pct,
            )
        else:
            self._controller = None

        # ---- Image buffer (thread-safe, written by ROS callback thread) ----
        self._bridge = CvBridge()
        self._img_lock = threading.Lock()
        self._latest_images: dict[str, np.ndarray] = {}
        for cam_name in config.camera_names:
            h, w = config.obs_img_resolution
            self._latest_images[cam_name] = np.zeros((h, w, 3), dtype=np.uint8)

        # ---- Subscribe camera topics ----
        if not self.config.is_dummy:
            self._img_subscribers: list[rospy.Subscriber] = []
            for cam_name, topic in zip(config.camera_names, config.img_topics):
                sub = rospy.Subscriber(
                    topic,
                    Image,
                    callback=self._make_img_callback(cam_name),
                    queue_size=1,
                    tcp_nodelay=True,
                )
                self._img_subscribers.append(sub)

        # ---- Initialize action/observation spaces ----
        self._init_action_obs_spaces()
        self._joint_limit_low: np.ndarray | None = None
        self._joint_limit_high: np.ndarray | None = None
        if self.config.joint_action_mode in ("delta", "absolute_normalized"):
            min_q = np.array(self.config.min_qpos, dtype=np.float32)
            max_q = np.array(self.config.max_qpos, dtype=np.float32)
            if len(min_q) < 14:
                self._joint_limit_low = np.tile(min_q[:7], 2).astype(np.float64)
                self._joint_limit_high = np.tile(max_q[:7], 2).astype(np.float64)
            else:
                self._joint_limit_low = min_q[:14].astype(np.float64)
                self._joint_limit_high = max_q[:14].astype(np.float64)

        # ---- Human-in-the-loop: keyboard + puppet (slave) joint feedback ----
        # _master_action_left/right store latest 7D puppet joint states per arm (naming kept
        # for minimal diff). Combined to 14D when teleoperation is active for intervene_action.
        if config.enable_human_intervention and not config.is_dummy:
            from rlinf.envs.realworld.common.keyboard.keyboard_listener import (
                KeyboardListener,
            )

            self._keyboard = KeyboardListener()
            self._policy_enabled = True
            self._master_action_left: np.ndarray | None = None
            self._master_action_right: np.ndarray | None = None
            self._master_action_lock = threading.Lock()
            rospy.Subscriber(
                config.puppet_joint_topic_left,
                JointState,
                self._on_master_joint_state_left,
                queue_size=1,
            )
            rospy.Subscriber(
                config.puppet_joint_topic_right,
                JointState,
                self._on_master_joint_state_right,
                queue_size=1,
            )
        else:
            self._keyboard = None
            self._policy_enabled = True
            self._master_action_left = None
            self._master_action_right = None
            self._master_action_lock = threading.Lock()

        if self.config.is_dummy:
            self._logger.info("PiperEnv initialized in dummy mode.")
            return

        # ---- Wait for robot ready ----
        self._logger.info("Waiting for Piper dual-arm robot to be ready...")
        ready = self._controller.wait_for_robot(timeout=30.0)
        if not ready:
            self._logger.warning("Robot wait timed out; some topics may not be ready.")

        self._logger.info(
            f"PiperEnv initialized: env_idx={env_idx}, "
            f"cameras={config.camera_names}, freq={config.step_frequency}Hz, "
            f"intervention={config.enable_human_intervention}"
        )

    # ==================================================================
    # ROS callbacks: puppet joint feedback + images
    # ==================================================================

    def _on_master_joint_state_left(self, msg: JointState) -> None:
        """Store latest left puppet arm joint state (7D: 6 joints + gripper)."""
        with self._master_action_lock:
            self._master_action_left = np.array(msg.position[:7], dtype=np.float64)

    def _on_master_joint_state_right(self, msg: JointState) -> None:
        """Store latest right puppet arm joint state (7D: 6 joints + gripper)."""
        with self._master_action_lock:
            self._master_action_right = np.array(msg.position[:7], dtype=np.float64)

    def _make_img_callback(self, cam_name: str):
        """Create a ROS image callback closure for the given camera name."""

        def _callback(msg: Image) -> None:
            try:
                cv_image = self._bridge.imgmsg_to_cv2(msg, "passthrough")
                h, w = self.config.obs_img_resolution
                if cv_image.shape[0] != h or cv_image.shape[1] != w:
                    cv_image = cv2.resize(cv_image, (w, h))
                with self._img_lock:
                    self._latest_images[cam_name] = cv_image
            except Exception as e:
                self._logger.warning(f"Image callback error ({cam_name}): {e}")

        return _callback

    def _teleop_active(self) -> bool:
        """True when ROS teleop is on (same signal as ``step`` / ``intervene_action``)."""
        if self.config.is_dummy:
            return False
        try:
            return bool(rospy.get_param("/enable_message_publish", False))
        except Exception:
            return False

    def _wait_for_teleop_release_before_reset(self) -> None:
        """Block physical reset until teleop releases, avoiding ``move_arm`` vs master conflict."""
        if self.config.is_dummy or not self.config.wait_teleop_release_before_reset:
            return
        poll = max(float(self.config.teleop_release_poll_sec), 0.05)
        timeout = self.config.teleop_release_reset_timeout_sec
        start = time.time()
        warned = False
        while self._teleop_active():
            if not warned:
                self._logger.info(
                    "reset: teleop active (/enable_message_publish=True); "
                    "waiting for release before moving arms to reset pose."
                )
                warned = True
            if timeout is not None and float(timeout) > 0.0:
                if time.time() - start > float(timeout):
                    self._logger.warning(
                        "reset: teleop still active after %.1f s timeout; proceeding with reset.",
                        float(timeout),
                    )
                    break
            time.sleep(poll)

    def _wait_for_enter_after_reset(self) -> None:
        """Pause reset completion until the operator confirms policy execution."""
        if self.config.is_dummy or not self.config.wait_enter_after_reset:
            return

        prompt = "Piper reset complete. Press Enter to start policy execution..."
        self._logger.info(prompt)
        try:
            input(prompt)
        except (EOFError, OSError) as exc:
            self._logger.warning(
                "Cannot wait for Enter in this process (%s). Continuing policy execution.",
                exc,
            )

    def _load_tcp_reset_pose_pool(self, path: str) -> list[dict[str, Any]]:
        if not path:
            self._logger.warning(
                "use_tcp_reset_pose_pool=True but tcp_reset_pose_pool_path is empty; "
                "falling back to zero reset."
            )
            return []
        pool_path = Path(path).expanduser()
        if not pool_path.exists():
            self._logger.warning(
                "TCP reset pose pool does not exist at %s; falling back to zero reset.",
                pool_path,
            )
            return []
        with open(pool_path, "r") as f:
            data = json.load(f)
        raw_poses = data.get("poses", data if isinstance(data, list) else [])
        metadata = data.get("metadata", [{} for _ in raw_poses])
        pool: list[dict[str, Any]] = []
        for idx, pose in enumerate(raw_poses):
            arr = np.asarray(pose, dtype=np.float64).reshape(-1)
            if arr.shape[0] < 14:
                self._logger.warning(
                    "Skipping TCP reset pose %d from %s: expected >=14 values, got %d.",
                    idx,
                    pool_path,
                    arr.shape[0],
                )
                continue
            item_meta = metadata[idx] if idx < len(metadata) else {}
            pool.append({"pose": arr[:14].copy(), "metadata": item_meta})
        self._logger.info(
            "Loaded %d TCP reset poses from %s.", len(pool), str(pool_path)
        )
        return pool

    def _sample_reset_pose(self) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        if self._tcp_reset_pose_pool:
            idx = int(self._tcp_reset_rng.integers(0, len(self._tcp_reset_pose_pool)))
            item = self._tcp_reset_pose_pool[idx]
            pose = np.asarray(item["pose"], dtype=np.float64).reshape(14)
            meta = dict(item.get("metadata", {}))
            meta["pool_index"] = idx
            left_reset = pose[:7].copy()
            right_reset = pose[7:14].copy()
            return left_reset, right_reset, meta

        left_reset = np.zeros(7, dtype=np.float64)
        right_reset = np.zeros(7, dtype=np.float64)
        return left_reset, right_reset, {"pool_index": None}

    def _move_to_reset_pose(
        self, left_reset: np.ndarray, right_reset: np.ndarray
    ) -> None:
        if self._controller is None:
            return

        steps = max(1, int(getattr(self.config, "tcp_reset_interpolate_steps", 1)))
        sleep_sec = getattr(self.config, "tcp_reset_interpolate_sleep", None)
        if sleep_sec is None:
            sleep_sec = 1.0 / max(float(self.config.step_frequency), 1.0)
        sleep_sec = max(0.0, float(sleep_sec))

        target = np.concatenate([left_reset, right_reset]).astype(np.float64)
        current = np.asarray(self._controller.get_qpos(), dtype=np.float64)
        if current.shape[0] < 14 or steps <= 1:
            self._controller.move_arm(left_reset, right_reset)
            return

        for alpha in np.linspace(1.0 / steps, 1.0, steps):
            waypoint = current + alpha * (target - current)
            self._controller.move_arm(waypoint[:7], waypoint[7:14])
            if sleep_sec > 0.0:
                time.sleep(sleep_sec)

    # ==================================================================
    # Action / observation spaces
    # ==================================================================

    def _init_action_obs_spaces(self) -> None:
        """Initialize action and observation spaces.

        Action space: 14D absolute joint positions (left 7 + right 7),
        each arm: 6 joints + 1 gripper.

        Observation space:
        - ``state``: qpos(14), qvel(14), effort(14), base_vel(2)
        - ``frames``: one (H, W, 3) image per camera
        """
        min_qpos = np.array(self.config.min_qpos, dtype=np.float32)
        max_qpos = np.array(self.config.max_qpos, dtype=np.float32)

        if self.config.joint_action_mode in ("delta", "absolute_normalized"):
            # Policy outputs [-1,1]^14; step() converts to absolute joint target
            self.action_space = gym.spaces.Box(
                low=-np.ones(14, dtype=np.float32),
                high=np.ones(14, dtype=np.float32),
                dtype=np.float32,
            )
        elif len(min_qpos) < 14:
            # Single-arm limits, tile to dual-arm
            action_low = np.tile(min_qpos[:7], 2).astype(np.float32)
            action_high = np.tile(max_qpos[:7], 2).astype(np.float32)
            self.action_space = gym.spaces.Box(
                low=action_low, high=action_high, dtype=np.float32
            )
        else:
            action_low = min_qpos[:14].astype(np.float32)
            action_high = max_qpos[:14].astype(np.float32)
            self.action_space = gym.spaces.Box(
                low=action_low, high=action_high, dtype=np.float32
            )

        h, w = self.config.obs_img_resolution
        state_space = gym.spaces.Dict(
            {
                "qpos": gym.spaces.Box(-np.inf, np.inf, shape=(14,), dtype=np.float64),
            }
        )
        frames_space = gym.spaces.Dict(
            {
                cam_name: gym.spaces.Box(0, 255, shape=(h, w, 3), dtype=np.uint8)
                for cam_name in self.config.camera_names
            }
        )
        self.observation_space = gym.spaces.Dict(
            {"state": state_space, "frames": frames_space}
        )
        self._base_observation_space = copy.deepcopy(self.observation_space)

    # ==================================================================
    # Action chunk preprocessing
    # ==================================================================

    def process_action_chunk(self, chunk_actions: np.ndarray) -> np.ndarray:
        """Apply Piper-specific smoothing to one env's action chunk."""
        actions = np.asarray(chunk_actions, dtype=np.float64).copy()
        if actions.ndim != 2 or actions.shape[-1] < 14:
            return actions

        if self.config.sliding_window_action_buffer:
            return self._apply_sliding_window_action_buffer(actions)
        if self.config.smooth_action_chunk:
            return self._smooth_action_sequence(actions)
        return actions

    def _apply_sliding_window_action_buffer(self, actions: np.ndarray) -> np.ndarray:
        execute_steps = actions.shape[0]
        trigger_remaining = int(self.config.sliding_window_inference_trigger_remaining)
        latency_steps = max(0, int(self.config.sliding_window_latency_steps))
        previous_action = None

        if self._sliding_window_action_buffer is None or trigger_remaining <= 0:
            smoothed_actions = self._smooth_action_sequence(actions)
        else:
            old_tail_count = min(
                max(1, execute_steps // 2),
                trigger_remaining,
                self._sliding_window_action_buffer.shape[0],
            )
            old_tail = self._sliding_window_action_buffer[-old_tail_count:]
            previous_action = old_tail[-1].copy()
            adaptive_horizon = self._calculate_adaptive_horizon(actions)
            new_start = min(latency_steps, max(0, adaptive_horizon - 1))
            new_end = min(max(adaptive_horizon, execute_steps), actions.shape[0])
            new_slice = actions[new_start:new_end]

            if new_slice.shape[0] == 0:
                new_slice = actions

            stitched_actions = np.concatenate([old_tail, new_slice], axis=0)
            stitched_actions = self._smooth_action_sequence(stitched_actions)
            smoothed_actions = stitched_actions[old_tail_count:]

            if smoothed_actions.shape[0] < execute_steps:
                fallback_actions = np.concatenate([old_tail, actions], axis=0)
                fallback_actions = self._smooth_action_sequence(fallback_actions)
                smoothed_actions = fallback_actions[old_tail_count:]

        smoothed_actions = smoothed_actions[:execute_steps]
        processed_actions = actions.copy()
        processed_actions[: smoothed_actions.shape[0]] = smoothed_actions
        processed_actions = self._blend_action_chunk_start(
            processed_actions,
            previous_action,
        )
        self._sliding_window_action_buffer = processed_actions.copy()
        return processed_actions

    def _blend_action_chunk_start(
        self, actions: np.ndarray, previous_action: np.ndarray | None
    ) -> np.ndarray:
        if previous_action is None or actions.shape[-1] < 14:
            return actions
        blend_steps = int(
            getattr(self.config, "sliding_window_boundary_blend_steps", 0)
        )
        if blend_steps <= 0:
            return actions
        blend_steps = min(blend_steps, actions.shape[0])
        ramp = np.linspace(
            1.0 / (blend_steps + 1),
            blend_steps / (blend_steps + 1),
            blend_steps,
            dtype=np.float64,
        ).reshape(blend_steps, 1)
        joint_mask = np.ones(actions.shape[-1], dtype=bool)
        joint_mask[[6, 13]] = False
        blended = (
            previous_action.reshape(1, -1) * (1.0 - ramp) + actions[:blend_steps] * ramp
        )
        actions = actions.copy()
        actions[:blend_steps, joint_mask] = blended[:, joint_mask]
        return actions

    def _calculate_adaptive_horizon(self, actions: np.ndarray) -> int:
        max_horizon = int(self.config.sliding_window_max_action_execute_horizon)
        trigger_remaining = int(self.config.sliding_window_inference_trigger_remaining)
        distance_thresh = float(self.config.sliding_window_distance_thresh)

        if actions.shape[0] < max_horizon:
            return actions.shape[0]

        actions_scaled = actions.copy()
        actions_scaled[:, 6] *= 10.0
        actions_scaled[:, 13] *= 10.0
        actions_scaled = actions_scaled[:, [6, 13]]

        base_action = actions_scaled[0]
        for check_point in (15, 20, 25, 30, 40):
            if check_point >= max_horizon or check_point >= actions_scaled.shape[0]:
                break
            l2_distance = np.linalg.norm(actions_scaled[check_point] - base_action)
            if l2_distance > distance_thresh:
                return min(check_point + trigger_remaining, actions.shape[0])

        return min(max_horizon, actions.shape[0])

    def _smooth_action_sequence(self, actions: np.ndarray) -> np.ndarray:
        if actions.shape[0] <= 9 or actions.shape[-1] < 14:
            return actions.copy()

        from scipy.signal import butter, filtfilt

        cutoff_freq = float(self.config.action_smooth_cutoff_freq)
        sampling_freq = float(
            getattr(
                self.config,
                "action_smooth_sampling_freq",
                getattr(self.config, "step_frequency", 25.0),
            )
        )
        if sampling_freq <= 0:
            sampling_freq = float(getattr(self.config, "step_frequency", 25.0))
        order = int(self.config.action_smooth_order)
        b, a = butter(
            order,
            cutoff_freq / (0.5 * sampling_freq),
            btype="low",
            analog=False,
        )

        smoothed_actions = actions.copy()
        left = smoothed_actions[:, :7].copy()
        right = smoothed_actions[:, 7:14].copy()
        left_gripper = left[:, -1].copy()
        right_gripper = right[:, -1].copy()
        left = filtfilt(b, a, left, axis=0)
        right = filtfilt(b, a, right, axis=0)
        left[:, -1] = left_gripper
        right[:, -1] = right_gripper
        smoothed_actions[:, :7] = left
        smoothed_actions[:, 7:14] = right
        return smoothed_actions

    # ==================================================================
    # Gym API: step
    # ==================================================================

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        """Execute one environment step.

        Args:
            action: 14D joint position array (left 7 + right 7),
                    each arm: 6 joint angles (rad) + 1 gripper (rad).

        Returns:
            (observation, reward, terminated, truncated, info) tuple.
            info["intervene_action"] is set to the master arm's 14D joint target
            when teleoperation is active.
        """
        start_time = time.time()

        action = np.asarray(action, dtype=np.float64)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # ---- Delta / absolute_normalized: policy output [-1,1]^14 -> absolute joint target ----
        if self.config.joint_action_mode == "absolute_normalized":
            assert (
                self._joint_limit_low is not None and self._joint_limit_high is not None
            )
            lo, hi = self._joint_limit_low, self._joint_limit_high
            action = (action + 1.0) / 2.0 * (hi - lo) + lo
            if self.config.delta_action_scale > 0:
                current_qpos = (
                    self._controller.get_qpos()
                    if not self.config.is_dummy and self._controller is not None
                    else np.zeros(14, dtype=np.float64)
                )
                scale = float(self.config.delta_action_scale)
                action = np.clip(action, current_qpos - scale, current_qpos + scale)
        elif self.config.joint_action_mode == "delta":
            assert (
                self._joint_limit_low is not None and self._joint_limit_high is not None
            )
            current_qpos = (
                self._controller.get_qpos()
                if not self.config.is_dummy and self._controller is not None
                else np.zeros(14, dtype=np.float64)
            )
            delta = action * float(self.config.delta_action_scale)
            action = np.clip(
                current_qpos + delta,
                self._joint_limit_low,
                self._joint_limit_high,
            )

        # ---- page_up: toggle policy output ----
        if self._keyboard is not None and self._keyboard.consume_press(
            self.config.policy_enable_key
        ):
            self._policy_enabled = not self._policy_enabled
            self._logger.info(
                f"Policy output {'enabled' if self._policy_enabled else 'disabled'}."
            )

        # ---- If policy disabled, hold current position ----
        if (
            not self._policy_enabled
            and not self.config.is_dummy
            and self._controller is not None
        ):
            action = self._controller.get_qpos()

        if self._policy_enabled and self.config.gripper_action_threshold is not None:
            action = action.copy()
            gripper_ids = [6, 13]
            grippers = action[gripper_ids]
            threshold = float(self.config.gripper_action_threshold)
            scale = float(self.config.gripper_action_scale)
            action[gripper_ids] = np.where(grippers < threshold, 0.0, grippers * scale)

        # ---- Split into left/right arm actions ----
        left_action, right_action = split_dual_arm_action(action)

        # ---- Check teleop state once (ROS node owns channel when active) ----
        teleop_active = not self.config.is_dummy and rospy.get_param(
            "/enable_message_publish", False
        )
        self._last_teleop_active = bool(teleop_active)

        # ---- Publish control command ----
        # Skip move_arm during teleoperation: the ROS node drives the puppet arms
        # directly via /master/joint_left|right; publishing here would conflict.
        if not self.config.is_dummy and not teleop_active:
            self._controller.move_arm(left_action, right_action)
        elif self.config.is_dummy:
            self._logger.debug(f"Dummy step: left={left_action}, right={right_action}")

        self._num_steps += 1

        # ---- Rate control ----
        step_time = time.time() - start_time
        sleep_time = max(0.0, (1.0 / self.config.step_frequency) - step_time)
        if sleep_time > 0:
            time.sleep(sleep_time)

        # ---- Get observation ----
        observation = self._get_observation()

        # ---- Compute reward ----
        reward = self._calc_step_reward(observation)

        # ---- Termination (aligned with FrankaEnv) ----
        terminated = (reward == 1.0) and (
            self._success_hold_counter >= self.config.success_hold_steps
        )
        truncated = self._num_steps >= self.config.max_num_steps

        # ---- Build info: record master arm action when teleop is active ----
        # Combine left (7D) + right (7D) master arm targets into a single 14D action.
        info: dict = {}
        if teleop_active:
            with self._master_action_lock:
                if (
                    self._master_action_left is not None
                    and self._master_action_right is not None
                ):
                    info["intervene_action"] = np.concatenate(
                        [self._master_action_left, self._master_action_right]
                    )

        return observation, reward, terminated, truncated, info

    # ==================================================================
    # Gym API: reset
    # ==================================================================

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[dict, dict]:
        """Reset environment to initial state.

        Aligned with ``FrankaEnv.reset()``:
        1. Enable arms
        2. Go to zero position
        3. Wait for joints to reach zero
        4. Return initial observation

        Args:
            seed: Random seed (reserved).
            options: Extra options (reserved).

        Returns:
            (observation, info) tuple.
        """
        self._num_steps = 0
        self._success_hold_counter = 0
        self._sliding_window_action_buffer = None
        self._last_teleop_active = False

        if self.config.is_dummy:
            observation = self._get_observation()
            return observation, {}

        self._wait_for_teleop_release_before_reset()

        # ---- Enable ----
        self._controller.enable_arm()
        time.sleep(0.5)

        # ---- Go to reset pose: either a sampled TCP-near pose or zero pose ----
        left_reset, right_reset, reset_meta = self._sample_reset_pose()
        if reset_meta.get("pool_index") is not None:
            self._logger.info(
                "PiperEnv reset sampled TCP pose pool_index=%s source=%s "
                "chunk=%s substep=%s tcp_distance=%.4f.",
                reset_meta.get("pool_index"),
                reset_meta.get("trajectory", ""),
                reset_meta.get("chunk_index", ""),
                reset_meta.get("substep_index", ""),
                float(reset_meta.get("tcp_distance", -1.0)),
            )
        self._move_to_reset_pose(left_reset, right_reset)

        # ---- Wait for joints to reach reset pose ----
        self._controller._wait_for_joint(
            target_pos=left_reset[:6],
            side="left",
            timeout=10.0,
            atol=0.05,
        )
        self._controller._wait_for_joint(
            target_pos=right_reset[:6],
            side="right",
            timeout=10.0,
            atol=0.05,
        )

        time.sleep(0.5)

        self._logger.info("PiperEnv reset complete.")
        self._wait_for_enter_after_reset()

        observation = self._get_observation()
        return observation, {}

    def _ensure_reward_config_arrays(self) -> None:
        """Normalize target_qpos / reward_threshold to 14D float64 (compatible with YAML lists)."""
        cfg = self.config
        cfg.target_qpos = np.asarray(cfg.target_qpos, dtype=np.float64).reshape(14)
        cfg.reward_threshold = np.asarray(
            cfg.reward_threshold, dtype=np.float64
        ).reshape(14)

    # ==================================================================
    # Observation
    # ==================================================================

    def _get_observation(self) -> dict:
        """Build the full observation dictionary.

        - ``state.qpos``: 14D dual-arm joint positions (6 joints + 1 gripper) x2
        - ``frames``: camera images

        Returns:
            Dict conforming to observation_space.
        """
        if not self.config.is_dummy:
            state = {
                "qpos": self._controller.get_qpos(),
            }
            frames = self._get_camera_frames()
            return copy.deepcopy({"state": state, "frames": frames})
        else:
            return self._base_observation_space.sample()

    def _get_camera_frames(self) -> dict[str, np.ndarray]:
        """Thread-safe retrieval of the latest frame from each camera."""
        frames = {}
        with self._img_lock:
            for cam_name in self.config.camera_names:
                frames[cam_name] = self._latest_images[cam_name].copy()
        return frames

    # ==================================================================
    # Reward
    # ==================================================================

    def _calc_step_reward(self, observation: dict, **kwargs: Any) -> float:
        """Compute per-step reward.

        Aligned with ``FrankaEnv._calc_step_reward``: compares current ``qpos``
        against ``target_qpos`` in joint space. All per-joint absolute errors
        within ``reward_threshold`` = target zone (reward 1.0, increment hold counter);
        otherwise reset counter and optionally return dense reward.

        Returns 0.0 in dummy mode.

        Args:
            observation: Current observation (must contain ``state.qpos``).

        Returns:
            Scalar reward.
        """
        if self.config.is_dummy:
            return 0.0

        qpos = np.asarray(observation["state"]["qpos"], dtype=np.float64).reshape(14)
        target = np.asarray(self.config.target_qpos, dtype=np.float64).reshape(14)
        thr = np.asarray(self.config.reward_threshold, dtype=np.float64).reshape(14)

        target_delta = np.abs(qpos - target)
        is_in_target_zone = bool(np.all(target_delta <= thr))

        if is_in_target_zone:
            self._success_hold_counter += 1
            reward = 1.0
        else:
            self._success_hold_counter = 0
            if self.config.use_dense_reward:
                reward = float(
                    np.exp(
                        -self.config.dense_reward_scale
                        * np.sum(np.square(target_delta))
                    )
                )
            else:
                reward = 0.0
            self._logger.debug(
                "Joint target not met: max_delta=%s, threshold=%s, reward=%s",
                float(np.max(target_delta)),
                thr,
                reward,
            )

        return reward

    # ==================================================================
    # Properties and utilities
    # ==================================================================

    @property
    def num_steps(self) -> int:
        """Number of steps executed in the current episode."""
        return self._num_steps

    @property
    def task_description(self) -> str:
        """Task description string, used by RealWorldEnv wrapper."""
        return self.config.task_name

    def close(self) -> None:
        """Close environment and release resources."""
        if not self.config.is_dummy and hasattr(self, "_img_subscribers"):
            for sub in self._img_subscribers:
                sub.unregister()
            self._img_subscribers = []
        self._logger.info("PiperEnv closed.")
