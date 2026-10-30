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

"""G1 reset, transport, mode generations and hardware command contracts."""

import copy
import os
import pty
import sys
import termios
import threading
import time
from concurrent.futures import Future
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
from robot_mocks.g1 import install_cameras, install_dds
from websockets.sync.server import serve

from deploy.g1.client import DryCameras, Keyboard, Session, load_config
from deploy.g1.hardware import ARM_IDS, Cameras, DryRobot, G1Robot
from deploy.g1.protocol import ACTION_NAMES, PolicyClient, pack, unpack

CONFIG = Path(__file__).parents[2] / "deploy/g1/client.yaml"


class ManualExecutor:
    def __init__(self):
        self.jobs = []

    def submit(self, fn, observation):
        future = Future()
        self.jobs.append((future, observation))
        return future

    def shutdown(self, **_):
        pass


class Policy:
    def close(self):
        pass

    def infer(self, observation):
        raise AssertionError("Use controlled futures")


@pytest.fixture
def session():
    config = load_config(CONFIG)
    robot = DryRobot(config["reset"]["state"])
    executor, messages = ManualExecutor(), []
    session = Session(robot, DryCameras(), Policy(), config, messages.append, executor)
    session.reset_start -= 1
    session.tick()
    assert session.mode == "ready"
    yield session, executor, messages
    session.scheduler.close()


def chunk(session, offset=0):
    actions = np.tile(session.robot.q, (10, 1))
    actions[:, 0] += offset
    return actions


def test_reset_keys_switch_and_direct_absolute_actions(session):
    s, e, messages = session
    s.key("space")
    assert not s.scheduler.actor and not e.jobs
    s.key("enter")
    s.tick()
    assert e.jobs[-1][1]["rlt_switch"] is False
    target = chunk(s, 0.1)
    e.jobs[-1][0].set_result(target)
    s.tick()
    np.testing.assert_allclose(
        s.robot.target, target[0]
    )  # no state/base action addition
    generation = s.scheduler.epoch
    s.key("enter")
    assert generation == s.scheduler.epoch
    old = e.jobs[-1][0]
    held_target = s.robot.target.copy()
    s.key("space")
    np.testing.assert_array_equal(s.robot.target, held_target)
    assert len(s.scheduler.queue) == 0 and s.scheduler.actor
    s.tick()
    assert len(e.jobs) == 2  # only one in-flight request
    old.set_result(chunk(s, 0.2))
    s.tick()
    assert e.jobs[-1][1]["rlt_switch"] is True
    assert s.scheduler.stats["discarded"] == 1
    e.jobs[-1][0].set_result(chunk(s, 0.03))
    s.tick()
    assert any("新模式已生效：actor 开启" in m for m in messages)
    s.key("space")
    s.key("space")
    s.key("space")
    assert not s.scheduler.actor and len(s.scheduler.queue) == 0
    e.jobs[-1][0].set_exception(ConnectionError("old mode"))
    s.tick()
    assert s.mode == "running" and not e.jobs[-1][1]["rlt_switch"]
    s.key("quit")
    assert s.mode == "closed" and not s.scheduler.active


def test_latency_alignment_blend_and_gripper_not_blended(session):
    s, e, _ = session
    s.key("enter")
    s.tick()
    old = chunk(s)
    old[:, 14:] = 20
    e.jobs[0][0].set_result(old)
    s.tick()
    s.tick()
    s.tick()
    new = chunk(s, 0.3)
    new[:, 1] += np.arange(10) * 0.01
    new[:, 14:] = 70
    e.jobs[1][0].set_result(new)
    s.tick()
    np.testing.assert_allclose(
        s.robot.target[:14], 2 / 3 * old[3, :14] + 1 / 3 * new[2, :14]
    )
    np.testing.assert_allclose(s.robot.target[14:], 70)
    assert s.scheduler.stats["skipped"] == 2


def test_queue_exhaustion_drops_late_response_and_reobserves(session):
    s, e, messages = session
    s.key("enter")
    s.tick()
    e.jobs[0][0].set_result(chunk(s))
    for _ in range(10):
        s.tick()
    s.tick()
    assert s.scheduler.stats["underruns"] == 1
    assert any("推理延迟" in m for m in messages)
    e.jobs[1][0].set_result(chunk(s, 0.3))
    s.tick()
    assert len(e.jobs) == 3 and s.scheduler.stats["discarded"] == 1
    assert len(s.scheduler.queue) == 0


@pytest.mark.parametrize("failure", ["network", "range", "stale_camera"])
def test_fault_disables_actor_and_enter_recovers(session, failure):
    s, e, _ = session
    s.key("enter")
    s.key("space")
    s.tick()
    if failure == "network":
        e.jobs[0][0].set_exception(ConnectionError("disconnected"))
    elif failure == "range":
        actions = chunk(s)
        actions[:, 0] = 100
        e.jobs[0][0].set_result(actions)
    else:
        s.cameras.snapshot = lambda: (_ for _ in ()).throw(RuntimeError("stale"))
    s.tick()
    assert s.mode == "fault" and not s.scheduler.actor and len(s.scheduler.queue) == 0
    s.cameras = DryCameras()
    s.key("enter")
    assert s.mode == "running" and not s.scheduler.actor


def test_reset_timeout_does_not_start_policy():
    c = load_config(CONFIG)
    q = np.array(c["reset"]["state"])
    q[0] += 0.5
    robot = DryRobot(q)
    e = ManualExecutor()
    s = Session(robot, DryCameras(), Policy(), c, lambda _: None, e)
    s.key("space")
    s.key("enter")
    assert s.mode == "resetting" and not s.scheduler.actor
    s.reset_start -= 20
    s.tick()
    assert s.mode == "fault" and not e.jobs and not s.reset_done
    np.testing.assert_allclose(robot.target, q)
    s.scheduler.close()


def test_reset_moves_smoothly_from_measured_pose(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    config = load_config(CONFIG)
    initial = np.array(config["reset"]["state"])
    initial[0] += 0.4
    robot = DryRobot(initial)
    s = Session(robot, DryCameras(), Policy(), config, lambda _: None, ManualExecutor())
    np.testing.assert_array_equal(robot.target, initial)
    previous = initial.copy()
    for _ in range(60):
        clock[0] += 0.02
        s.tick()
        current = robot.state()
        assert np.max(np.abs(current[:14] - previous[:14])) <= 0.020001
        previous = current
    assert s.mode == "ready" and s.reset_done
    np.testing.assert_allclose(robot.state(), config["reset"]["state"], atol=0.03)
    assert not s.scheduler.active
    s.scheduler.close()


@pytest.mark.parametrize("completed", [True, False])
def test_stale_response_requires_enter(session, completed):
    s, e, _ = session
    s.key("enter")
    s.tick()
    future, epoch, steps, started = s.scheduler.pending
    s.scheduler.pending = (future, epoch, steps, started - 2)
    if completed:
        future.set_result(chunk(s))
    s.tick()
    assert s.mode == "fault" and not s.scheduler.actor


def test_gripper_clipping_keeps_joint_coordinates(session):
    s, _, _ = session
    a = chunk(s, 0.1)
    a[:, 14:] = [-0.62, 101.2]
    prepared = s.prepare_actions(a)
    np.testing.assert_array_equal(prepared[:, :14], a[:, :14])
    np.testing.assert_array_equal(prepared[:, 14:], np.tile([0, 100], (10, 1)))
    a[0, 14] = np.inf
    with pytest.raises(ValueError):
        s.prepare_actions(a)


def test_no_async_prefetch_when_disabled(session):
    s, e, _ = session
    s.scheduler.enabled = False
    s.key("enter")
    s.tick()
    e.jobs[0][0].set_result(chunk(s))
    for _ in range(10):
        s.tick()
    assert len(e.jobs) == 1
    s.tick()
    assert len(e.jobs) == 2


def test_real_transport_numpy_schema():
    original = {
        "images": {"cam_high": np.arange(18, dtype=np.uint8).reshape(2, 3, 3)},
        "state": np.arange(16, dtype=np.float32),
        "rlt_switch": False,
    }
    decoded = unpack(pack(original))
    np.testing.assert_array_equal(original["state"], decoded["state"])
    # Fixed OpenPI byte-key schema, independently decoded without our object hook.
    import msgpack

    raw = msgpack.unpackb(pack(original))
    assert raw["state"][b"dtype"] == "<f4" and raw["state"][b"shape"] == [16]
    assert (
        raw["images"]["cam_high"][b"data"] == original["images"]["cam_high"].tobytes()
    )


@contextmanager
def policy_server(behavior):
    def handler(ws):
        ws.send(
            pack(
                {
                    "action_horizon": 10,
                    "action_dim": 16,
                    "action_names": ACTION_NAMES,
                    "default_rlt_switch": False,
                    "client_async_supported": True,
                }
            )
        )
        try:
            for request in ws:
                request = unpack(request)
                response = {
                    "actions": np.zeros((10, 16)),
                    "q_values": np.zeros(2),
                    "rlt_switch": request["rlt_switch"],
                }
                if behavior == "delay":
                    time.sleep(0.2)
                if behavior == "disconnect":
                    ws.close()
                    return
                if behavior == "mode":
                    response["rlt_switch"] = not request["rlt_switch"]
                if behavior == "shape":
                    response["actions"] = np.zeros((20, 16))
                if behavior == "nan":
                    response["actions"][0, 0] = np.nan
                ws.send(pack(response))
        except Exception:
            pass

    with serve(handler, "127.0.0.1", 0, compression=None) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        p = PolicyClient(
            {
                "uri": f"ws://127.0.0.1:{server.socket.getsockname()[1]}",
                "connect_timeout_s": 1,
                "timeout_s": 0.05 if behavior == "delay" else 1,
            }
        )
        try:
            yield p
        finally:
            p.close()
            server.shutdown()
            thread.join(2)


@pytest.mark.parametrize(
    "behavior", ["ok", "delay", "disconnect", "mode", "shape", "nan"]
)
def test_websocket_faults(behavior):
    with policy_server(behavior) as policy:
        if behavior == "ok":
            for actor in (False, True, False):
                assert policy.infer({"rlt_switch": actor}).shape == (10, 16)
        else:
            with pytest.raises(Exception):
                policy.infer({"rlt_switch": False})


def test_close_interrupts_inference():
    with policy_server("delay") as policy:
        policy.config["timeout_s"] = 30
        policy.connect()
        errors = []

        def request():
            try:
                policy.infer({"rlt_switch": False})
            except Exception as e:
                errors.append(e)

        worker = threading.Thread(target=request)
        worker.start()
        time.sleep(0.02)
        policy.close()
        worker.join(1)
        assert not worker.is_alive() and errors
        with pytest.raises(RuntimeError, match="closed"):
            policy.connect()


def test_terminal_restoration(monkeypatch):
    master, slave = pty.openpty()
    try:
        old = termios.tcgetattr(slave)
        with os.fdopen(os.dup(slave)) as stdin:
            monkeypatch.setattr(sys, "stdin", stdin)
            with Keyboard() as keyboard:
                os.write(master, b"\n q")
                time.sleep(0.01)
                assert keyboard.poll() == ["enter", "space", "quit"]
            assert termios.tcgetattr(slave) == old
    finally:
        os.close(master)
        os.close(slave)


def test_dds_start_observed_targets_gains_limits_and_release(monkeypatch):
    c = copy.deepcopy(load_config(CONFIG)["robot"])
    c["network_interface"] = "lo"
    q = np.array(load_config(CONFIG)["reset"]["state"])
    q[14:] = [12, 87]
    sdk = install_dds(monkeypatch, q)
    robot = G1Robot(c, gravity=lambda _: np.ones(14))
    try:
        robot.connect()
        assert not any(p.messages for p in sdk.publishers.values())
        np.testing.assert_allclose(robot.state(), q)
        robot.start()
        deadline = time.monotonic() + 1
        while (
            not all(p.messages for p in sdk.publishers.values())
            and time.monotonic() < deadline
        ):
            time.sleep(0.005)
        first = sdk.publishers["rt/arm_sdk"].messages[0]
        np.testing.assert_allclose([first.motor_cmd[i].q for i in ARM_IDS], q[:14])
        assert all(first.motor_cmd[i].mode == 0 for i in range(15))
        assert first.motor_cmd[15].kp == 80 and first.motor_cmd[19].kp == 40
        for i, side in enumerate(("left", "right")):
            np.testing.assert_allclose(
                sdk.publishers[f"rt/dex1/{side}/cmd"].messages[0].cmds[0].q,
                q[14 + i] * 5.4 / 100,
            )
        target = q.copy()
        target[0] += 0.3
        robot.command(target)
        time.sleep(0.03)
        last = sdk.publishers["rt/arm_sdk"].messages[-1]
        assert 0 < last.motor_cmd[15].q - q[0] <= 1 / 250 + 1e-9
        assert sdk.factories == [(0, "lo")]
    finally:
        robot.close()
    assert sdk.publishers["rt/arm_sdk"].messages[-1].motor_cmd[29].q == 0
    assert len(sdk.closed) == 6
    assert all(not thread.is_alive() for thread in robot.threads)


def test_camera_rgb_order_staleness_and_cleanup(monkeypatch):
    c = load_config(CONFIG)["cameras"]
    pipelines = install_cameras(monkeypatch, c["serials"])
    cameras = Cameras(c)
    cameras.connect()
    images = cameras.snapshot()
    for channel, name in enumerate(c["serials"]):
        assert np.all(images[name][..., channel] == 255)
        assert pipelines[channel].config.stream[3] == "rgb8"
    cameras.close()
    for name, (image, stamp) in list(cameras.frames.items()):
        cameras.frames[name] = (image, stamp - 5)
    with pytest.raises(RuntimeError, match="过期"):
        cameras.snapshot()
    assert all(not t.is_alive() for t in cameras.threads)
