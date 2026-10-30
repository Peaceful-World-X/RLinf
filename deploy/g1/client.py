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

"""Reset G1, press Enter for VLA, and Space to switch the RLT actor."""

import argparse
import logging
import os
import select
import sys
import termios
import time
import tty
from pathlib import Path

import numpy as np
import yaml

if __package__:
    from .hardware import ActionLimits, Cameras, DryRobot, G1Robot
    from .protocol import PolicyClient
    from .scheduler import ChunkScheduler
else:
    from hardware import ActionLimits, Cameras, DryRobot, G1Robot
    from protocol import PolicyClient
    from scheduler import ChunkScheduler

logger = logging.getLogger("g1-client")
CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")


def load_config(path: str | Path) -> dict:
    with open(path) as file:
        config = yaml.safe_load(file)
    ActionLimits().validate(config["reset"]["state"])
    positive = [
        ("rtc", "max_response_age_s"),
        ("control", "hz"),
        ("reset", "timeout_s"),
        ("reset", "joint_tolerance_rad"),
        ("reset", "gripper_tolerance"),
        ("reset", "max_joint_velocity_rad_s"),
        ("reset", "gripper_speed"),
        ("robot", "max_joint_velocity_rad_s"),
        ("robot", "arm_hz"),
        ("robot", "gripper_hz"),
        ("robot", "state_timeout_s"),
        ("robot", "connect_timeout_s"),
        ("server", "timeout_s"),
        ("server", "connect_timeout_s"),
        ("cameras", "max_age_s"),
        ("cameras", "connect_timeout_s"),
        ("cameras", "fps"),
    ]
    for section, key in positive:
        value = config[section][key]
        if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{section}.{key} must be positive and finite")
    if set(config["cameras"]["serials"]) != set(CAMERAS):
        raise ValueError(
            "Exactly cam_high, cam_left_wrist and cam_right_wrist are required"
        )
    if len(set(config["cameras"]["serials"].values())) != 3:
        raise ValueError("Camera serial numbers must be distinct")
    if not isinstance(config["rtc"]["enabled"], bool):
        raise ValueError("rtc.enabled must be boolean")
    if not isinstance(config["control"]["clip_gripper"], bool):
        raise ValueError("control.clip_gripper must be boolean")
    for key in ("prefetch_remaining_steps", "blend_steps"):
        if type(config["rtc"][key]) is not int:
            raise ValueError(f"rtc.{key} must be an integer")
    prefetch, blend = (
        config["rtc"]["prefetch_remaining_steps"],
        config["rtc"]["blend_steps"],
    )
    if not 1 <= prefetch < 10 or not 0 <= blend <= prefetch:
        raise ValueError("Require 1 <= prefetch < 10 and 0 <= blend <= prefetch")
    speed = config["robot"]["gripper"]["max_velocity_rad_s"]
    if not np.isfinite(speed) or speed <= 0:
        raise ValueError("Gripper velocity must be positive and finite")
    return config


class Keyboard:
    """Read terminal keys without waiting for a newline; restore on every exit."""

    def __enter__(self):
        self.fd = sys.stdin.fileno()
        if not os.isatty(self.fd):
            raise RuntimeError("客户端交互需要终端 TTY")
        self.old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def poll(self) -> list[str]:
        keys = []
        while select.select([self.fd], [], [], 0)[0]:
            value = os.read(self.fd, 1)
            if not value:
                keys.append("quit")
                break
            char = value.decode(errors="ignore")
            key = {
                "\r": "enter",
                "\n": "enter",
                " ": "space",
                "q": "quit",
                "Q": "quit",
                "\x03": "quit",
            }.get(char)
            if key:
                keys.append(key)
        return keys

    def __exit__(self, *_):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)


class DryCameras:
    def __init__(self, images=None):
        self.images = images or {k: np.zeros((480, 640, 3), np.uint8) for k in CAMERAS}

    def connect(self) -> None:
        pass

    def snapshot(self) -> dict[str, np.ndarray]:
        return {k: v.copy() for k, v in self.images.items()}

    def close(self) -> None:
        pass


class Session:
    """Keyboard and control state machine. Inference is always a background task."""

    def __init__(
        self, robot, cameras, policy, config, notify=logger.info, executor=None
    ):
        self.robot, self.cameras, self.config = robot, cameras, config
        self.notify = notify
        self.scheduler = ChunkScheduler(
            policy, self.observe, self.prepare_actions, config["rtc"], notify, executor
        )
        self.mode = "resetting"
        self.reset_done = False
        self.holding = False
        self.begin_reset()

    def prepare_actions(self, actions: np.ndarray) -> np.ndarray:
        return self.robot.limits.prepare(
            actions, self.config["control"]["clip_gripper"]
        )

    def observe(self) -> dict:
        return {
            "images": self.cameras.snapshot(),
            "state": self.robot.state().astype(np.float32),
            "prompt": self.config["prompt"],
        }

    def begin_reset(self) -> None:
        self.scheduler.stop()
        self.reset_start = time.monotonic()
        self.reset_origin = self.robot.state()
        self.reset_goal = np.array(self.config["reset"]["state"])
        speed = np.r_[
            np.full(14, self.config["reset"]["max_joint_velocity_rad_s"]),
            [self.config["reset"]["gripper_speed"]] * 2,
        ]
        self.reset_duration = max(
            0.1, float(np.max(np.abs(self.reset_goal - self.reset_origin) / speed))
        )
        self.mode = "resetting"
        self.notify("正在复位到扎带 episode 000 初始姿态；actor 关闭")

    def key(self, key: str) -> None:
        if key == "quit":
            self.scheduler.stop()
            self.robot.hold()
            self.mode = "closed"
            self.notify("收到退出按键：保持当前位置，关闭推理并释放双臂控制权")
        elif key == "enter":
            if self.mode == "resetting":
                self.notify("收到回车：仍在复位，等待到位后再按回车")
            elif self.mode == "running":
                self.notify("收到回车：连续推理已运行")
            else:
                try:
                    self.robot.clear_fault()
                    self.observe()
                    if not self.reset_done:
                        self.begin_reset()
                    else:
                        self.scheduler.start()
                        self.mode = "running"
                        self.holding = False
                        self.notify("收到回车：开启连续 VLA 推理，actor 关闭")
                except Exception as error:
                    self.fail(error)
        elif key == "space":
            if self.mode != "running":
                self.notify("收到空格：尚未运行；复位完成后按回车启动，actor 保持关闭")
            else:
                self.scheduler.toggle()
                # Keep the last installed target during a mode switch.
                self.holding = True
                self.notify(
                    f"收到空格：正在切换至 {'actor 开启' if self.scheduler.actor else 'VLA / actor 关闭'}；已清空旧动作"
                )

    def fail(self, error: Exception) -> None:
        self.scheduler.stop()
        self.robot.hold()
        self.mode = "fault"
        self.notify(f"保持当前位置，actor 已关闭：{error}；排除问题后按回车恢复")

    def tick(self) -> None:
        if self.mode in ("closed", "fault"):
            return
        try:
            self.robot.check()
            self.cameras.snapshot()
            if self.mode == "resetting":
                elapsed = time.monotonic() - self.reset_start
                if elapsed > self.config["reset"]["timeout_s"]:
                    raise TimeoutError("复位未到位，已超过允许时间")
                fraction = min(1.0, elapsed / self.reset_duration)
                self.robot.command(
                    self.reset_origin + fraction * (self.reset_goal - self.reset_origin)
                )
                error = np.abs(self.robot.state() - self.reset_goal)
                if (
                    fraction == 1
                    and np.max(error[:14])
                    <= self.config["reset"]["joint_tolerance_rad"]
                    and np.max(error[14:]) <= self.config["reset"]["gripper_tolerance"]
                ):
                    self.reset_done = True
                    self.mode = "ready"
                    self.notify(
                        "复位完成；按回车开始连续推理，空格切换 actor，q/Ctrl+C 退出"
                    )
            elif self.mode == "running":
                action = self.scheduler.tick()
                if action is None:
                    if not self.holding:
                        self.robot.hold()
                        self.holding = True
                else:
                    self.robot.command(action)
                    self.holding = False
                    self.scheduler.after_action()
        except Exception as error:
            self.fail(error)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("client.yaml")
    )
    parser.add_argument("--server", help="Override WebSocket URI")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Mock hardware; still use the real policy server",
    )
    parser.add_argument(
        "--observation",
        type=Path,
        help="Dry-run NPZ with state and three camera arrays",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S"
    )
    config = load_config(args.config)
    if args.server:
        config["server"]["uri"] = args.server
    if args.observation and not args.dry_run:
        parser.error("--observation requires --dry-run")
    robot = cameras = policy = session = None
    try:
        # Acquire terminal and protocol first, before any robot ownership.
        with Keyboard() as keyboard:
            policy = PolicyClient(config["server"])
            policy.connect()
            if args.dry_run:
                initial = config["reset"]["state"]
                images = None
                if args.observation:
                    with np.load(args.observation, allow_pickle=False) as sample:
                        initial = sample["state"].copy()
                        images = {k: sample[k].copy() for k in CAMERAS}
                robot = DryRobot(initial, config["robot"]["max_joint_velocity_rad_s"])
                cameras = DryCameras(images)
                logger.info("DRY-RUN：模拟硬件，不创建 DDS 发布者")
            else:
                robot = G1Robot(config["robot"])
                cameras = Cameras(config["cameras"])
            robot.connect()
            cameras.connect()
            logger.info("设备已连接，正在预热服务；预热动作不执行")
            policy.infer(
                {
                    "images": cameras.snapshot(),
                    "state": robot.state(),
                    "prompt": config["prompt"],
                    "rlt_switch": False,
                }
            )
            logger.info("服务预热完成：%.0f ms", policy.last_latency_s * 1000)
            robot.start()
            session = Session(robot, cameras, policy, config)
            period = 1 / config["control"]["hz"]
            deadline = time.monotonic()
            while session.mode != "closed":
                for key in keyboard.poll():
                    session.key(key)
                    if session.mode == "closed":
                        break
                now = time.monotonic()
                if now >= deadline:
                    session.tick()
                    deadline = max(deadline + period, time.monotonic() + 0.001)
                time.sleep(min(0.005, max(0, deadline - time.monotonic())))
    except KeyboardInterrupt:
        logger.info("收到 Ctrl+C：退出并释放双臂控制权")
    finally:
        if session:
            session.scheduler.stop()
        # Release each resource even when a disconnected device fails to close.
        resources = [robot, cameras, session.scheduler if session else policy]
        for resource in resources:
            if resource is not None:
                try:
                    resource.close()
                except Exception:
                    logger.exception("退出清理失败：%s", type(resource).__name__)


if __name__ == "__main__":
    main()
