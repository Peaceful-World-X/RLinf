# Copyright 2026 The RLinf Authors.
# Portions adapted from xr_teleoperate, Copyright 2025 Unitree Robotics.
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

"""G1 arm_sdk, Dex1 and RGB readers; SDK imports occur only in real hardware mode."""

import logging
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ARM_IDS = tuple(range(15, 29))
ARM_JOINTS = [
    f"{side}_{joint}_joint"
    for side in ("left", "right")
    for joint in (
        "shoulder_pitch",
        "shoulder_roll",
        "shoulder_yaw",
        "elbow",
        "wrist_roll",
        "wrist_pitch",
        "wrist_yaw",
    )
]
URDF = Path(__file__).parent / "assets/g1_body29_hand14.urdf"
logger = logging.getLogger("g1-client")


class ActionLimits:
    def __init__(self):
        joints = {
            j.attrib["name"]: j.find("limit") for j in ET.parse(URDF).findall("joint")
        }
        self.lower = np.array(
            [float(joints[n].attrib["lower"]) for n in ARM_JOINTS] + [0, 0]
        )
        self.upper = np.array(
            [float(joints[n].attrib["upper"]) for n in ARM_JOINTS] + [100, 100]
        )

    def validate(self, actions: np.ndarray) -> None:
        actions = np.asarray(actions)
        if (
            actions.ndim not in (1, 2)
            or actions.shape[-1] != 16
            or not np.isfinite(actions).all()
        ):
            raise ValueError("动作必须是有限的 16 维绝对目标")
        if np.any(actions < self.lower) or np.any(actions > self.upper):
            bad = np.argwhere((actions < self.lower) | (actions > self.upper)).tolist()
            raise ValueError(f"动作越界（关节按 URDF，夹爪 0～100），位置={bad[:8]}")

    def prepare(self, actions: np.ndarray, clip_gripper: bool) -> np.ndarray:
        actions = np.array(actions, dtype=float, copy=True)
        if actions.shape != (10, 16) or not np.isfinite(actions).all():
            raise ValueError("Expected finite actions[10,16]")
        if clip_gripper:
            actions[:, 14:] = np.clip(actions[:, 14:], 0, 100)
        self.validate(actions)
        return actions


class GripperMapping:
    def __init__(self, config):
        self.closed = np.array(config["closed_rad"], dtype=float)
        self.open = np.array(config["open_rad"], dtype=float)
        if (
            self.closed.shape != (2,)
            or self.open.shape != (2,)
            or not np.isfinite([self.closed, self.open]).all()
            or np.any(self.open == self.closed)
        ):
            raise ValueError(
                "Dex1 calibration requires two distinct finite endpoints per side"
            )

    def to_motor(self, value: np.ndarray) -> np.ndarray:
        return self.closed + np.asarray(value) / 100 * (self.open - self.closed)

    def to_policy(self, value: np.ndarray) -> np.ndarray:
        return (np.asarray(value) - self.closed) / (self.open - self.closed) * 100


class Gravity:
    """Use xr_teleoperate's reduced arm model without IK or mesh loading."""

    def __init__(self):
        import pinocchio as pin

        full = pin.buildModelFromUrdf(str(URDF))
        locked = [i for i in range(1, full.njoints) if full.names[i] not in ARM_JOINTS]
        self.model = pin.buildReducedModel(full, locked, pin.neutral(full))
        self.data = self.model.createData()
        self.q_indices = [
            self.model.joints[self.model.getJointId(n)].idx_q for n in ARM_JOINTS
        ]
        self.v_indices = [
            self.model.joints[self.model.getJointId(n)].idx_v for n in ARM_JOINTS
        ]
        self.pin = pin

    def __call__(self, joints: np.ndarray) -> np.ndarray:
        q = self.pin.neutral(self.model)
        q[self.q_indices] = joints
        return np.asarray(self.pin.computeGeneralizedGravity(self.model, self.data, q))[
            self.v_indices
        ].copy()


class G1Robot:
    """Subscribe first; acquire arm ownership only after explicit start()."""

    def __init__(self, config, gravity=None):
        self.config = config
        self.mapping = GripperMapping(config["gripper"])
        self.limits = ActionLimits()
        self.gravity = gravity if gravity is not None else Gravity()
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.channels = []
        self.threads = []
        self.samples = {}
        self.fault = None
        self.target = None
        self.started = False

    def _receive(self, key, value):
        with self.lock:
            self.samples[key] = (value, time.monotonic())

    def connect(self) -> None:
        from unitree_sdk2py.core.channel import (
            ChannelFactoryInitialize,
            ChannelPublisher,
            ChannelSubscriber,
        )
        from unitree_sdk2py.idl.default import (
            unitree_go_msg_dds__MotorCmd_,
            unitree_hg_msg_dds__LowCmd_,
        )
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
        from unitree_sdk2py.utils.crc import CRC

        interface = self.config["network_interface"]
        if not interface or not Path("/sys/class/net", interface).exists():
            raise ValueError(
                "在 client.yaml 的 robot.network_interface 指定 G1 DDS 网卡"
            )
        ChannelFactoryInitialize(self.config["domain_id"], interface)
        self.crc = CRC()
        self.arm_msg = unitree_hg_msg_dds__LowCmd_()
        self.grip_msgs = [MotorCmds_(), MotorCmds_()]
        for msg in self.grip_msgs:
            msg.cmds = [unitree_go_msg_dds__MotorCmd_()]
            msg.cmds[0].kp, msg.cmds[0].kd = 5.0, 0.05
            msg.cmds[0].dq, msg.cmds[0].tau = 0.0, 0.0
        for topic, typ, key in [
            ("rt/lowstate", LowState_, "arm"),
            ("rt/dex1/left/state", MotorStates_, "left"),
            ("rt/dex1/right/state", MotorStates_, "right"),
        ]:
            sub = ChannelSubscriber(topic, typ)
            self.channels.append(sub)
            sub.Init(lambda msg, k=key: self._receive(k, msg), 10)
        self.arm_pub = ChannelPublisher("rt/arm_sdk", LowCmd_)
        self.grip_pubs = [
            ChannelPublisher(f"rt/dex1/{side}/cmd", MotorCmds_)
            for side in ("left", "right")
        ]
        for pub in [self.arm_pub, *self.grip_pubs]:
            self.channels.append(pub)
            pub.Init()
        deadline = time.monotonic() + self.config["connect_timeout_s"]
        while time.monotonic() < deadline:
            try:
                self.state()
                return
            except RuntimeError:
                time.sleep(0.01)
        raise TimeoutError("等待 G1 双臂及两侧 Dex1 状态超时，尚未发布控制目标")

    def state(self, fresh: bool = True) -> np.ndarray:
        with self.lock:
            if not all(k in self.samples for k in ("arm", "left", "right")):
                raise RuntimeError("尚未收到完整 DDS 状态")
            if fresh and any(
                time.monotonic() - self.samples[k][1] > self.config["state_timeout_s"]
                for k in ("arm", "left", "right")
            ):
                raise RuntimeError("DDS 状态过期")
            arm = self.samples["arm"][0]
            q = [arm.motor_state[i].q for i in ARM_IDS]
            grip = self.mapping.to_policy(
                [self.samples[k][0].states[0].q for k in ("left", "right")]
            )
            state = np.r_[q, grip]
            if not np.isfinite(state).all():
                raise RuntimeError("DDS 状态含非有限值")
            return state

    def start(self) -> None:
        self.target = self.state().copy()
        # Non-arm entries remain disabled, including legs and waist.
        for i, motor in enumerate(self.arm_msg.motor_cmd):
            motor.mode = 0
            motor.q = motor.dq = motor.tau = motor.kp = motor.kd = 0.0
            if i in ARM_IDS:
                wrist = (i - 15) % 7 >= 4
                motor.mode = 1
                motor.kp, motor.kd = (40.0, 1.5) if wrist else (80.0, 3.0)
        self.arm_msg.mode_pr = 0
        self.arm_msg.mode_machine = self.samples["arm"][0].mode_machine
        self.started = True
        for kind, hz in [
            ("arm", self.config["arm_hz"]),
            ("gripper", self.config["gripper_hz"]),
        ]:
            thread = threading.Thread(
                target=self._loop, args=(kind, hz), daemon=True, name=f"g1-{kind}"
            )
            self.threads.append(thread)
            thread.start()

    def command(self, target: np.ndarray) -> None:
        self.limits.validate(target)
        with self.lock:
            if self.fault:
                raise RuntimeError(self.fault)
            self.target = np.asarray(target, dtype=float).copy()

    def hold(self) -> None:
        with self.lock:
            if self.target is not None:
                self.target = self.state(fresh=False)

    def check(self) -> None:
        self.state()
        if self.fault:
            raise RuntimeError(self.fault)

    def clear_fault(self) -> None:
        self.state()
        self.fault = None

    def _arm_write(self, q, weight):
        torque = self.gravity(q)
        if torque.shape != (14,) or not np.isfinite(torque).all():
            raise RuntimeError("重力补偿输出无效")
        for j, motor_id in enumerate(ARM_IDS):
            self.arm_msg.motor_cmd[motor_id].q = float(q[j])
            self.arm_msg.motor_cmd[motor_id].tau = float(torque[j])
        self.arm_msg.motor_cmd[29].q = float(weight)
        self.arm_msg.crc = self.crc.Crc(self.arm_msg)
        if self.arm_pub.Write(self.arm_msg) is False:
            raise RuntimeError("arm_sdk 发布失败")

    def _loop(self, kind, hz):
        period = 1.0 / hz
        while not self.stop_event.is_set():
            start = time.monotonic()
            try:
                try:
                    actual = self.state()
                except RuntimeError as error:
                    self.fault = str(error)
                    self.hold()
                    actual = self.state(fresh=False)
                with self.lock:
                    target = self.target.copy()
                if kind == "arm":
                    limit = self.config["max_joint_velocity_rad_s"] * period
                    delta = target[:14] - actual[:14]
                    q = actual[:14] + delta / max(1, np.max(np.abs(delta)) / limit)
                    self._arm_write(q, 1.0)
                else:
                    current = self.mapping.to_motor(actual[14:])
                    desired = self.mapping.to_motor(target[14:])
                    limit = self.config["gripper"]["max_velocity_rad_s"] * period
                    desired = current + np.clip(desired - current, -limit, limit)
                    for i, pub in enumerate(self.grip_pubs):
                        self.grip_msgs[i].cmds[0].q = float(desired[i])
                        if pub.Write(self.grip_msgs[i]) is False:
                            raise RuntimeError("Dex1 发布失败")
            except Exception as error:
                self.fault = f"{kind}: {error}"
                self.hold()
            self.stop_event.wait(max(0, period - (time.monotonic() - start)))

    def close(self) -> None:
        self.stop_event.set()
        for thread in self.threads:
            thread.join(timeout=2)
        try:
            if self.started:
                actual = self.state(fresh=False)
                q = actual[:14]
                for i, pub in enumerate(self.grip_pubs):
                    self.grip_msgs[i].cmds[0].q = float(
                        self.mapping.to_motor(actual[14:])[i]
                    )
                    try:
                        pub.Write(self.grip_msgs[i])
                    except Exception:
                        logger.exception("退出时保持夹爪失败")
                for weight in np.linspace(1.0, 0.0, 51):
                    self._arm_write(q, weight)
                    time.sleep(0.01)
                self.started = False
        finally:
            if self.started:
                # Release ownership even when gravity or the ramp failed.
                try:
                    self.arm_msg.motor_cmd[29].q = 0.0
                    self.arm_msg.crc = self.crc.Crc(self.arm_msg)
                    self.arm_pub.Write(self.arm_msg)
                except Exception:
                    logger.exception("释放双臂控制权失败")
                self.started = False
            for channel in reversed(self.channels):
                try:
                    channel.Close()
                except Exception:
                    logger.exception("关闭 DDS 通道失败")
            self.channels.clear()


class Cameras:
    """Each camera thread publishes its newest RGB frame without blocking control."""

    def __init__(self, config):
        self.config = config
        self.lock = threading.Lock()
        self.frames = {}
        self.pipelines = []
        self.threads = []
        self.stop_event = threading.Event()
        self.error = None

    def connect(self) -> None:
        import pyrealsense2 as rs

        for name, serial in self.config["serials"].items():
            pipeline, config = rs.pipeline(), rs.config()
            config.enable_device(str(serial))
            config.enable_stream(
                rs.stream.color,
                self.config["width"],
                self.config["height"],
                rs.format.rgb8,
                self.config["fps"],
            )
            pipeline.start(config)
            self.pipelines.append(pipeline)
            thread = threading.Thread(
                target=self._read,
                args=(name, pipeline),
                daemon=True,
                name=f"g1-camera-{name}",
            )
            self.threads.append(thread)
            thread.start()
        deadline = time.monotonic() + self.config["connect_timeout_s"]
        while time.monotonic() < deadline:
            try:
                self.snapshot()
                return
            except RuntimeError:
                time.sleep(0.01)
        raise TimeoutError(f"等待三相机超时: {self.error}")

    def _read(self, name, pipeline):
        try:
            while not self.stop_event.is_set():
                frame = pipeline.wait_for_frames(timeout_ms=1000).get_color_frame()
                if frame:
                    image = np.asanyarray(frame.get_data()).copy()
                    with self.lock:
                        self.frames[name] = (image, time.monotonic())
        except Exception as error:
            if not self.stop_event.is_set():
                self.error = str(error)

    def snapshot(self) -> dict[str, np.ndarray]:
        with self.lock:
            if self.error:
                raise RuntimeError(f"相机读取失败: {self.error}")
            if set(self.frames) != set(self.config["serials"]):
                raise RuntimeError("尚未收到完整三相机图像")
            if any(
                time.monotonic() - stamp > self.config["max_age_s"]
                for _, stamp in self.frames.values()
            ):
                raise RuntimeError("相机图像过期")
            return {k: image.copy() for k, (image, _) in self.frames.items()}

    def close(self) -> None:
        self.stop_event.set()
        for pipeline in self.pipelines:
            try:
                pipeline.stop()
            except Exception:
                logger.exception("关闭相机失败")
        for thread in self.threads:
            thread.join(timeout=2)
        self.pipelines.clear()


class DryRobot:
    """Local target follower. Never imports Unitree SDK or creates DDS channels."""

    def __init__(self, initial, velocity=1.0):
        self.q = np.array(initial, dtype=float)
        self.target = self.q.copy()
        self.last = time.monotonic()
        self.velocity = velocity
        self.limits = ActionLimits()

    def state(self) -> np.ndarray:
        now = time.monotonic()
        dt, self.last = now - self.last, now
        limits = np.r_[np.full(14, self.velocity), [100, 100]] * dt
        self.q += np.clip(self.target - self.q, -limits, limits)
        return self.q.copy()

    def command(self, target: np.ndarray) -> None:
        self.limits.validate(target)
        self.state()
        self.target = np.array(target, copy=True)

    def hold(self) -> None:
        self.target = self.state()

    def connect(self) -> None:
        pass

    def start(self) -> None:
        pass

    def check(self) -> None:
        pass

    def clear_fault(self) -> None:
        pass

    def close(self) -> None:
        self.hold()
