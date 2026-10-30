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

"""Single-request asynchronous H10 queue, adapted from OpenPI deplay scheduling."""

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np


class ChunkScheduler:
    """Call tick at the action clock; a None result means hold, never extrapolate."""

    def __init__(self, policy, observe, prepare, config, notify, executor=None):
        self.policy, self.observe, self.prepare = policy, observe, prepare
        self.notify = notify
        self.enabled = config["enabled"]
        self.prefetch = config["prefetch_remaining_steps"]
        self.blend = config["blend_steps"]
        self.max_age = config["max_response_age_s"]
        if not 1 <= self.prefetch < 10 or not 0 <= self.blend <= self.prefetch:
            raise ValueError(
                "RTC requires 1 <= prefetch < 10 and 0 <= blend <= prefetch"
            )
        self.executor = executor or ThreadPoolExecutor(1, "g1-inference")
        self.pending = None
        self.epoch = 0
        self.steps = 0
        self.actor = False
        self.active = False
        self.queue = np.empty((0, 16))
        self.mode_announced = False
        self.stats = {"requests": 0, "discarded": 0, "underruns": 0, "skipped": 0}

    def clear(self) -> None:
        self.epoch += 1
        self.queue = np.empty((0, 16))
        self.mode_announced = False

    def start(self) -> None:
        self.clear()
        self.actor = False
        self.active = True

    def toggle(self) -> None:
        self.clear()
        self.actor = not self.actor

    def stop(self) -> None:
        self.clear()
        self.active = False
        self.actor = False

    def _request(self):
        observation = self.observe()
        observation["rlt_switch"] = self.actor
        self.pending = (
            self.executor.submit(self.policy.infer, observation),
            self.epoch,
            self.steps,
            time.monotonic(),
        )
        self.stats["requests"] += 1

    def tick(self) -> np.ndarray | None:
        if (
            self.pending is not None
            and self.active
            and self.pending[1] == self.epoch
            and time.monotonic() - self.pending[3] > self.max_age
        ):
            raise TimeoutError("推理观测已过期")
        if self.pending is not None and self.pending[0].done():
            future, epoch, requested_at, started = self.pending
            self.pending = None
            if epoch != self.epoch or not self.active:
                self.stats["discarded"] += 1
            else:
                chunk = future.result()
                if time.monotonic() - started > self.max_age:
                    raise TimeoutError("推理观测已过期")
                chunk = self.prepare(chunk)
                skipped = self.steps - requested_at
                if skipped >= len(chunk):
                    self.clear()
                    self.stats["discarded"] += 1
                    self.notify("推理结果已过期，保持当前位置并重新获取观测")
                else:
                    chunk = chunk[skipped:].copy()
                    overlap = min(self.blend, len(self.queue), len(chunk))
                    for i in range(overlap):
                        weight = (i + 1) / (overlap + 1)
                        chunk[i, :14] = (1 - weight) * self.queue[
                            i, :14
                        ] + weight * chunk[i, :14]
                    self.queue = chunk
                    self.stats["skipped"] += skipped
                    if not self.mode_announced:
                        self.notify(
                            f"新模式已生效：{'actor 开启' if self.actor else 'VLA / actor 关闭'}"
                        )
                        self.mode_announced = True
        if not self.active:
            return None
        if len(self.queue) == 0:
            if self.pending is not None and self.pending[1] == self.epoch:
                if self.mode_announced:
                    self.clear()
                    self.stats["underruns"] += 1
                    self.notify(
                        "推理延迟超过动作队列，保持当前位置；丢弃迟到响应后重新推理"
                    )
            if self.pending is None:
                self._request()
            return None
        action = self.queue[0].copy()
        self.queue = self.queue[1:]
        self.steps += 1
        return action

    def after_action(self) -> None:
        """Capture the boundary observation after installing this tick's target."""
        if (
            self.active
            and self.enabled
            and self.pending is None
            and len(self.queue) <= self.prefetch
        ):
            self._request()

    def close(self) -> None:
        self.stop()
        self.policy.close()
        self.executor.shutdown(wait=True, cancel_futures=True)
