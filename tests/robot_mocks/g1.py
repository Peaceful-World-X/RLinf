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

"""Small DDS and RealSense process-edge fakes for the standalone G1 client."""

import copy
import sys
import threading
from types import ModuleType
from types import SimpleNamespace as NS

import numpy as np


def install_dds(monkeypatch, state):
    publishers = {}
    callbacks = {}
    closed = []
    factories = []

    def motor(q=0.0):
        return NS(q=float(q), dq=0.0, tau=0.0, kp=0.0, kd=0.0, mode=0)

    class Publisher:
        def __init__(self, topic, _):
            self.topic, self.messages = topic, []
            publishers[topic] = self

        def Init(self):
            pass

        def Write(self, message):
            self.messages.append(copy.deepcopy(message))
            return True

        def Close(self):
            closed.append(self.topic)

    class Subscriber:
        def __init__(self, topic, _):
            self.topic = topic

        def Init(self, callback, _):
            callbacks[self.topic] = callback
            if self.topic == "rt/lowstate":
                values = [motor() for _ in range(35)]
                for i, q in enumerate(state[:14], 15):
                    values[i].q = q
                callback(NS(motor_state=values, mode_machine=3))
            else:
                i = 14 if "left" in self.topic else 15
                callback(NS(states=[motor(state[i] * 5.4 / 100)]))

        def Close(self):
            closed.append(self.topic)

    definitions = {
        "unitree_sdk2py.core.channel": {
            "ChannelPublisher": Publisher,
            "ChannelSubscriber": Subscriber,
            "ChannelFactoryInitialize": lambda *args: factories.append(args),
        },
        "unitree_sdk2py.idl.default": {
            "unitree_hg_msg_dds__LowCmd_": lambda: NS(
                motor_cmd=[motor() for _ in range(35)]
            ),
            "unitree_go_msg_dds__MotorCmd_": motor,
        },
        "unitree_sdk2py.idl.unitree_go.msg.dds_": {
            "MotorCmds_": NS,
            "MotorStates_": NS,
        },
        "unitree_sdk2py.idl.unitree_hg.msg.dds_": {"LowCmd_": NS, "LowState_": NS},
        "unitree_sdk2py.utils.crc": {"CRC": lambda: NS(Crc=lambda _: 123)},
    }
    for name, values in definitions.items():
        parts = name.split(".")
        for end in range(1, len(parts) + 1):
            module_name = ".".join(parts[:end])
            module = ModuleType(module_name)
            module.__path__ = []
            if end == len(parts):
                module.__dict__.update(values)
            monkeypatch.setitem(sys.modules, module_name, module)
    return NS(
        publishers=publishers, callbacks=callbacks, closed=closed, factories=factories
    )


def install_cameras(monkeypatch, serials):
    pipelines = []

    class Config:
        def enable_device(self, serial):
            self.serial = serial

        def enable_stream(self, *args):
            self.stream = args

    class Pipeline:
        def __init__(self):
            self.stop_event = threading.Event()
            pipelines.append(self)

        def start(self, config):
            self.config = config

        def wait_for_frames(self, **_):
            self.stop_event.wait(0.005)
            color = np.zeros((4, 6, 3), np.uint8)
            color[..., list(serials.values()).index(self.config.serial)] = 255
            return NS(get_color_frame=lambda: NS(get_data=lambda: color))

        def stop(self):
            self.stop_event.set()

    monkeypatch.setitem(
        sys.modules,
        "pyrealsense2",
        NS(
            pipeline=Pipeline,
            config=Config,
            stream=NS(color="color"),
            format=NS(rgb8="rgb8"),
        ),
    )
    return pipelines
