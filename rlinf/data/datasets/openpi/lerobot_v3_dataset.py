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

"""Read local LeRobot v3 data without converting videos or updating LeRobot."""

from __future__ import annotations

import json
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np


class LeRobotV3Dataset:
    """Supply OpenPI samples with episode-clamped actions and offset video frames.

    Numeric columns are loaded lazily once per worker. Video decoder handles use
    a bounded cache; decoded images are never retained between samples.
    """

    def __init__(
        self,
        root: str | Path,
        action_horizon: int,
        action_sequence_keys: Sequence[str] = ("action",),
        episode_indices: Sequence[int] | None = None,
        video_cache_size: int = 3,
    ) -> None:
        import pyarrow.parquet as pq

        self.root = Path(root)
        self.info = json.loads((self.root / "meta/info.json").read_text())
        if self.info["codebase_version"] != "v3.0":
            raise ValueError("LeRobotV3Dataset requires codebase_version v3.0")
        if action_horizon < 1 or video_cache_size < 1:
            raise ValueError("action_horizon and video_cache_size must be positive")
        self.action_horizon = action_horizon
        self.action_sequence_keys = tuple(action_sequence_keys)
        self.video_cache_size = video_cache_size
        self.camera_keys = [
            key
            for key, spec in self.info["features"].items()
            if spec["dtype"] == "video"
        ]
        self.tasks = {
            int(row["task_index"]): row["task"]
            for row in pq.read_table(self.root / "meta/tasks.parquet").to_pylist()
        }
        episodes = []
        for path in sorted((self.root / "meta/episodes").rglob("*.parquet")):
            columns = [
                key
                for key in pq.read_schema(path).names
                if not key.startswith("stats/")
            ]
            episodes.extend(pq.read_table(path, columns=columns).to_pylist())
        episodes.sort(key=lambda row: row["episode_index"])
        available = {row["episode_index"] for row in episodes}
        if episode_indices is not None:
            selected = [int(index) for index in episode_indices]
            if not selected or len(selected) != len(set(selected)):
                raise ValueError("episode_indices must be nonempty and unique")
            if missing := set(selected) - available:
                raise ValueError(f"Unknown episode indices: {sorted(missing)}")
            episodes = [row for row in episodes if row["episode_index"] in selected]
        if not episodes:
            raise ValueError(f"No episode metadata found under {self.root}")
        self.episodes = episodes
        self._offsets = np.cumsum([0] + [row["length"] for row in episodes])
        self._arrays: dict[str, np.ndarray] | None = None
        self._decoders: OrderedDict[str, Any] = OrderedDict()

    def __len__(self) -> int:
        return int(self._offsets[-1])

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_arrays"] = None
        state["_decoders"] = OrderedDict()
        return state

    def _load_arrays(self) -> None:
        import pyarrow.parquet as pq

        selected = [row["episode_index"] for row in self.episodes]
        paths = sorted(
            {
                str(
                    self.root
                    / self.info["data_path"].format(
                        chunk_index=row["data/chunk_index"],
                        file_index=row["data/file_index"],
                    )
                )
                for row in self.episodes
            }
        )
        columns = list(
            dict.fromkeys(
                [
                    "observation.state",
                    *self.action_sequence_keys,
                    "index",
                    "episode_index",
                    "frame_index",
                    "task_index",
                    "timestamp",
                ]
            )
        )
        table = pq.read_table(
            paths,
            columns=columns,
            filters=[("episode_index", "in", selected)],
        )
        if len(table) != len(self):
            raise ValueError(f"Expected {len(self)} selected rows, found {len(table)}")
        arrays = {}
        for key in columns:
            column = table[key].combine_chunks()
            if hasattr(column, "values"):
                arrays[key] = column.values.to_numpy().reshape(
                    len(table), *self.info["features"][key]["shape"]
                )
            else:
                arrays[key] = column.to_numpy()
        for position, episode in enumerate(self.episodes):
            start, end = self._offsets[position : position + 2]
            if not (
                np.all(arrays["episode_index"][start:end] == episode["episode_index"])
                and np.array_equal(
                    arrays["frame_index"][start:end], np.arange(end - start)
                )
                and np.array_equal(
                    arrays["index"][start:end],
                    np.arange(
                        episode["dataset_from_index"], episode["dataset_to_index"]
                    ),
                )
            ):
                raise ValueError(
                    f"Noncontiguous or mismatched rows for episode {episode['episode_index']}"
                )
        self._arrays = arrays

    def _image(self, episode: dict, key: str, frame_index: int) -> np.ndarray:
        from torchcodec.decoders import VideoDecoder

        prefix = f"videos/{key}/"
        relative = self.info["video_path"].format(
            video_key=key,
            chunk_index=episode[prefix + "chunk_index"],
            file_index=episode[prefix + "file_index"],
        )
        if relative not in self._decoders:
            if len(self._decoders) >= self.video_cache_size:
                self._decoders.popitem(last=False)
            self._decoders[relative] = VideoDecoder(
                str(self.root / relative),
                device="cpu",
                num_ffmpeg_threads=1,
                # Timestamp lookup is checked below; avoid a full-file index scan.
                seek_mode="approximate",
            )
        self._decoders.move_to_end(relative)
        timestamp = episode[prefix + "from_timestamp"] + frame_index / self.info["fps"]
        # Match by PTS, allowing only rounding error in the v3 timestamp offsets.
        frame = self._decoders[relative].get_frame_played_at(timestamp + 1e-6)
        if abs(frame.pts_seconds - timestamp) > 1e-3:
            raise ValueError(
                f"Video frame mismatch at {relative}: {frame.pts_seconds} != {timestamp}"
            )
        return frame.data.numpy().astype(np.float32) / 255.0

    def __getitem__(self, index: int) -> dict[str, Any]:
        index = int(index)
        if index < 0 or index >= len(self):
            raise IndexError(f"Sample index {index} outside [0, {len(self)})")
        if self._arrays is None:
            self._load_arrays()
        position = int(np.searchsorted(self._offsets, index, side="right") - 1)
        episode = self.episodes[position]
        end = int(self._offsets[position + 1])
        sample = {
            key: values[index].copy()
            for key, values in self._arrays.items()
            if key not in self.action_sequence_keys
        }
        indices = np.minimum(np.arange(index, index + self.action_horizon), end - 1)
        for key in self.action_sequence_keys:
            sample[key] = self._arrays[key][indices].copy()
            sample[f"{key}_is_pad"] = (
                np.arange(index, index + self.action_horizon) >= end
            )
        for key in self.camera_keys:
            sample[key] = self._image(episode, key, int(sample["frame_index"]))
        sample["task"] = self.tasks[int(sample["task_index"])]
        return sample
