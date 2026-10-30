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

"""OpenPI-compatible MessagePack transport without a training dependency."""

import threading
import time

import msgpack
import numpy as np

ACTION_NAMES = [
    f"k{side}{joint}.pos"
    for side in ("Left", "Right")
    for joint in (
        "ShoulderPitch",
        "ShoulderRoll",
        "ShoulderYaw",
        "Elbow",
        "WristRoll",
        "WristPitch",
        "WristYaw",
    )
] + ["left_gripper.pos", "right_gripper.pos"]


def pack_array(value):
    """Encode the byte-key ndarray schema used by OpenPI msgpack_numpy."""
    if isinstance(value, (np.ndarray, np.generic)):
        if value.dtype.kind in "VOc":
            raise ValueError(f"Unsupported ndarray dtype: {value.dtype}")
        if isinstance(value, np.ndarray):
            return {
                b"__ndarray__": True,
                b"data": value.tobytes(),
                b"dtype": value.dtype.str,
                b"shape": value.shape,
            }
        return {
            b"__npgeneric__": True,
            b"data": value.item(),
            b"dtype": value.dtype.str,
        }
    raise TypeError(f"Unsupported MessagePack object: {type(value)}")


def unpack_array(value):
    if b"__ndarray__" in value or b"__npgeneric__" in value:
        dtype = np.dtype(value[b"dtype"])
        if dtype.kind in "VOc":
            raise ValueError(f"Unsupported ndarray dtype: {dtype}")
        if b"__ndarray__" in value:
            return np.frombuffer(value[b"data"], dtype=dtype).reshape(value[b"shape"])
        return dtype.type(value[b"data"])
    return value


def pack(value: dict) -> bytes:
    return msgpack.packb(value, default=pack_array)


def unpack(value: bytes | str) -> dict:
    if isinstance(value, str):
        raise RuntimeError(f"Policy server error: {value}")
    return msgpack.unpackb(value, object_hook=unpack_array)


class PolicyClient:
    """One inference worker owns the socket; close may interrupt that worker."""

    def __init__(self, config):
        self.config = config
        self.ws = None
        self.metadata = {}
        self.last_latency_s = 0.0
        self.closed = threading.Event()
        self.lock = threading.Lock()

    def connect(self) -> None:
        from websockets.sync.client import connect

        if self.ws is not None:
            return
        if self.closed.is_set():
            raise RuntimeError("Policy client is closed")
        ws = connect(
            self.config["uri"],
            compression=None,
            max_size=16 * 1024**2,
            open_timeout=self.config["connect_timeout_s"],
            close_timeout=0.2,
        )
        try:
            with self.lock:
                if self.closed.is_set():
                    raise RuntimeError("Policy client is closed")
                self.ws = ws
            metadata = unpack(ws.recv(timeout=self.config["timeout_s"]))
            if (
                metadata.get("action_horizon") != 10
                or metadata.get("action_dim") != 16
                or metadata.get("action_names") != ACTION_NAMES
                or metadata.get("default_rlt_switch") is not False
                or metadata.get("client_async_supported") is not True
            ):
                raise ValueError(
                    "Server must expose the G1 H10/16D actor-switch protocol"
                )
            self.metadata = metadata
        except BaseException:
            self._disconnect()
            ws.close()
            raise

    def infer(self, observation: dict) -> np.ndarray:
        self.connect()
        ws = self.ws
        started = time.monotonic()
        try:
            ws.send(pack(observation))
            response = unpack(ws.recv(timeout=self.config["timeout_s"]))
            actions = np.asarray(response["actions"], dtype=np.float64)
            q = np.asarray(response["q_values"])
            if actions.shape != (10, 16) or not np.isfinite(actions).all():
                raise ValueError("Policy actions must be finite [10,16]")
            if q.shape != (2,) or not np.isfinite(q).all():
                raise ValueError("Policy twin Q must be finite [2]")
            if (
                not isinstance(response.get("rlt_switch"), (bool, np.bool_))
                or response["rlt_switch"] != observation["rlt_switch"]
            ):
                raise ValueError("Actor mode acknowledgement does not match request")
            self.last_latency_s = time.monotonic() - started
            return actions.copy()
        except BaseException:
            self._disconnect()
            raise

    def _disconnect(self):
        with self.lock:
            ws, self.ws = self.ws, None
        if ws is not None:
            ws.close()

    def close(self) -> None:
        self.closed.set()
        self._disconnect()
