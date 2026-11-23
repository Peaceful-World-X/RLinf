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

"""Create two small G1 episodes directly on the worker."""

import tempfile
from pathlib import Path

import numpy as np
from assets_io import save_json
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def generate(dest, cameras):
    dest = Path(dest)
    if (dest / "datasets.json").exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
        stage = Path(tmp) / "input"
        keys = ["observation.images." + camera for camera in cameras]
        features = {
            key: {
                "dtype": "image",
                "shape": (64, 64, 3),
                "names": ["height", "width", "channels"],
            }
            for key in keys
        }
        for key in ("observation.state", "action"):
            features[key] = {"dtype": "float32", "shape": (16,), "names": ["joints"]}
        ds = LeRobotDataset.create(
            "g1/smoke",
            40,
            features,
            root=stage / "smoke",
            robot_type="g1",
            use_videos=False,
        )
        for ep in range(2):
            for frame in range(64):
                state = np.sin(np.arange(16) * 0.1 + frame / 64 + ep).astype("float32")
                item = {
                    key: np.full((64, 64, 3), (frame + i * 40) % 255, dtype="uint8")
                    for i, key in enumerate(keys)
                }
                ds.add_frame(
                    dict(item, **{"observation.state": state, "action": state + 0.01}),
                    task="Thread the cable tie.",
                )
            ds.save_episode()
        stats = {
            key: {
                k: [v] * 16
                for k, v in (("mean", 0.0), ("std", 1.0), ("q01", -1.0), ("q99", 1.0))
            }
            for key in ("state", "actions")
        }
        save_json(stage / "norm_stats.json", {"norm_stats": stats})
        save_json(
            stage / "datasets.json", [{"path": "smoke", "type": "sft", "weight": 1}]
        )
        stage.rename(dest)
