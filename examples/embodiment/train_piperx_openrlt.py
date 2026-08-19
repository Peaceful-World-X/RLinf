#!/usr/bin/env python3
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

"""Online OpenRLToken entrypoint for externally managed PiperX hardware."""

from __future__ import annotations

import json
import os
import select
import sys
import termios
import threading
import tty

os.environ["RLINF_SKIP_ROS_CLEANUP"] = "1"
if sys.stdin.isatty():
    os.environ["RLINF_PIPERX_TERMINAL"] = os.ttyname(sys.stdin.fileno())

import hydra
import torch.multiprocessing as mp
from omegaconf import OmegaConf

from rlinf.config import validate_cfg
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.env.async_env_worker import AsyncEnvWorker
from rlinf.workers.rollout.hf.async_huggingface_worker import (
    AsyncMultiStepRolloutWorker,
)

mp.set_start_method("spawn", force=True)

REQUIRED_PUBLISHERS = {
    "/puppet/joint_left",
    "/puppet/joint_right",
    "/puppet/end_pose_left",
    "/puppet/end_pose_right",
    "/camera_f/color/image_raw",
    "/camera_l/color/image_raw",
    "/camera_r/color/image_raw",
    "/piper/intervene_mode",
    "/piper/subtask_reward",
}
REQUIRED_SUBSCRIBERS = {"/policy/joint_left", "/policy/joint_right"}


class _TerminalKeyRelay:
    """Forward PiperX operator keys from the driver TTY to the EnvWorker."""

    def __init__(self) -> None:
        self._stdin_fd = sys.stdin.fileno()
        self._old_termios = termios.tcgetattr(self._stdin_fd)
        self._stop = threading.Event()
        self.path = f"/tmp/rlinf_piperx_keys_{os.getpid()}"
        os.mkfifo(self.path, 0o600)
        self._fifo_fd = os.open(self.path, os.O_RDWR | os.O_NONBLOCK)
        os.environ["RLINF_PIPERX_KEY_FIFO"] = self.path
        tty.setcbreak(self._stdin_fd)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            readable, _, _ = select.select([self._stdin_fd], [], [], 0.1)
            if not readable:
                continue
            key = os.read(self._stdin_fd, 1)
            if key == b"\r":
                key = b"\n"
            if key in (b"\n", b" ", b"q", b"x", b"f", b"0", b"1"):
                os.write(self._fifo_fd, key)

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        termios.tcsetattr(self._stdin_fd, termios.TCSADRAIN, self._old_termios)
        os.close(self._fifo_fd)
        os.unlink(self.path)
        os.environ.pop("RLINF_PIPERX_KEY_FIFO", None)


def validate_piperx_cfg(cfg) -> None:
    """Validate the dual-arm, no-gripper OpenRLToken contract."""
    model = cfg.actor.model
    if str(model.model_type) != "openpi_rl_token":
        raise ValueError("PiperX OpenRLT requires model_type=openpi_rl_token")
    if int(model.action_dim) != 12 or int(model.robot_state_dim) != 12:
        raise ValueError("PiperX OpenRLT must train 12 arm joints")
    if int(model.env_action_dim) != 14:
        raise ValueError("PiperX environment actions must remain 14-D")
    controlled = [int(value) for value in model.controlled_action_indices]
    if controlled != [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]:
        raise ValueError("PiperX OpenRLT must exclude both gripper dimensions")
    if int(cfg.env.train.override_cfg.chunk_size) != int(model.num_action_chunks):
        raise ValueError("PiperX chunk_size must match num_action_chunks")
    for split in (cfg.env.train, cfg.env.eval):
        if str(split.init_params.id) != "PiperXInterventionEnv-v1":
            raise ValueError("PiperX config must use PiperXInterventionEnv-v1")


def assert_external_ros_ready() -> None:
    """Verify the externally managed PiperX ROS topics."""
    import rosgraph

    publishers, subscribers, _ = rosgraph.Master(
        "/rlinf_piperx_readonly_preflight"
    ).getSystemState()
    published = {name for name, nodes in publishers if nodes}
    subscribed = {name for name, nodes in subscribers if nodes}
    missing_publishers = sorted(REQUIRED_PUBLISHERS - published)
    missing_subscribers = sorted(REQUIRED_SUBSCRIBERS - subscribed)
    if missing_publishers or missing_subscribers:
        raise RuntimeError(
            "PiperX ROS preflight failed: "
            f"missing publishers={missing_publishers}, "
            f"missing subscribers={missing_subscribers}"
        )


@hydra.main(
    version_base="1.1",
    config_path="config",
    config_name="realworld_piperx_td3_rl_token_online",
)
def main(cfg) -> None:
    cfg = validate_cfg(cfg)
    validate_piperx_cfg(cfg)
    assert_external_ros_ready()
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    from rlinf.runners.async_embodied_runner import AsyncEmbodiedRunner
    from rlinf.workers.actor.async_fsdp_td3_policy_worker import (
        AsyncEmbodiedTD3FSDPPolicy,
    )

    key_relay = _TerminalKeyRelay() if sys.stdin.isatty() else None
    try:
        cluster = Cluster(
            cluster_cfg=cfg.cluster,
            distributed_log_dir=cfg.runner.per_worker_log_path,
        )
        placement = HybridComponentPlacement(cfg, cluster)
        actor = AsyncEmbodiedTD3FSDPPolicy.create_group(cfg).launch(
            cluster,
            name=cfg.actor.group_name,
            placement_strategy=placement.get_strategy("actor"),
        )
        rollout = AsyncMultiStepRolloutWorker.create_group(cfg).launch(
            cluster,
            name=cfg.rollout.group_name,
            placement_strategy=placement.get_strategy("rollout"),
        )
        env = AsyncEnvWorker.create_group(cfg).launch(
            cluster,
            name=cfg.env.group_name,
            placement_strategy=placement.get_strategy("env"),
        )
        runner = AsyncEmbodiedRunner(
            cfg=cfg, actor=actor, rollout=rollout, env=env, reward=None
        )
        runner.init_workers()
        runner.run()
    finally:
        if key_relay is not None:
            key_relay.close()


if __name__ == "__main__":
    main()
