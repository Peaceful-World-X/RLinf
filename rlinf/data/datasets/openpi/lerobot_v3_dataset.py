# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Read local LeRobot v3 episodes without converting their shared video files."""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np


class LeRobotV3Dataset:
    """Provide raw LeRobot samples for the official OpenPI transforms.

    Args:
        root: Existing, read-only LeRobot v3 dataset directory.
        action_horizon: Number of actions, padded with the episode's last action.
        episode_ids: Original episode IDs to include; None selects every episode.
        sample_indices: Original global frame indices in the selected episodes.
            Their supplied order is preserved; None selects all selected frames.
    """

    def __init__(
        self,
        root: str | Path,
        action_horizon: int,
        episode_ids: list[int] | None = None,
        sample_indices: list[int] | None = None,
    ):
        import pyarrow.parquet as pq

        self.root = Path(root)
        self.info = json.loads((self.root / "meta/info.json").read_text())
        if self.info["codebase_version"] != "v3.0":
            raise ValueError("LeRobotV3Dataset requires codebase_version=v3.0")
        self.action_horizon = int(action_horizon)
        self.fps = float(self.info["fps"])
        if self.action_horizon < 1 or self.fps <= 0:
            raise ValueError("action_horizon and dataset fps must be positive")
        self.video_keys = [
            key
            for key, feature in self.info["features"].items()
            if feature["dtype"] == "video"
        ]
        paths = sorted((self.root / "meta/episodes").glob("chunk-*/file-*.parquet"))
        if not paths:
            raise FileNotFoundError("No v3 episode metadata parquet files found")
        records = [
            record for path in paths for record in pq.read_table(path).to_pylist()
        ]
        self.episodes = {int(record["episode_index"]): record for record in records}
        if len(self.episodes) != len(records):
            raise ValueError("Episode metadata contains duplicate episode IDs")
        selected = sorted(self.episodes if episode_ids is None else episode_ids)
        if not selected or len(set(selected)) != len(selected):
            raise ValueError("episode_ids must be nonempty and contain no duplicates")
        unknown = set(selected) - self.episodes.keys()
        if unknown:
            raise ValueError(f"Unknown episode IDs: {sorted(unknown)}")
        self.episode_ids = selected
        selected_records = [self.episodes[ep] for ep in selected]
        ranges = [
            (int(ep["dataset_from_index"]), int(ep["dataset_to_index"]))
            for ep in selected_records
        ]
        for ep, (start, end) in zip(selected_records, ranges, strict=True):
            if start < 0 or end <= start or end - start != int(ep["length"]):
                raise ValueError(
                    f"Invalid frame range for episode {ep['episode_index']}"
                )
        all_indices = np.concatenate([np.arange(start, end) for start, end in ranges])
        if len(np.unique(all_indices)) != len(all_indices):
            raise ValueError("Selected episodes have overlapping global frame ranges")
        self.frame_indices = all_indices
        if sample_indices is not None:
            indices = np.asarray(sample_indices, dtype=np.int64)
            if (
                indices.ndim != 1
                or len(indices) == 0
                or len(np.unique(indices)) != len(indices)
            ):
                raise ValueError(
                    "sample_indices must be nonempty unique global frame indices"
                )
            if not np.isin(indices, all_indices).all():
                raise ValueError(
                    "sample_indices includes frames outside the selected episodes"
                )
            self.frame_indices = indices
        self._starts = np.array([start for start, _ in ranges])
        self._ends = np.array([end for _, end in ranges])
        if np.any(self._starts[1:] < self._ends[:-1]):
            raise ValueError("Episode IDs must follow increasing global frame ranges")
        self.tasks = {}
        for task in pq.read_table(self.root / "meta/tasks.parquet").to_pylist():
            text = task.get("task", task.get("__index_level_0__"))
            if not isinstance(text, str):
                raise ValueError("Task metadata must contain a task string")
            self.tasks[int(task["task_index"])] = text
        self._tables: OrderedDict[str, Any] = OrderedDict()
        self._decoders: OrderedDict[str, Any] = OrderedDict()

    def __len__(self) -> int:
        return len(self.frame_indices)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_tables"] = OrderedDict()
        state["_decoders"] = OrderedDict()
        return state

    def _table(self, episode: dict[str, Any]) -> Any:
        import pyarrow.parquet as pq

        path = self.info["data_path"].format(
            chunk_index=episode["data/chunk_index"],
            file_index=episode["data/file_index"],
        )
        if path not in self._tables:
            self._tables[path] = pq.read_table(self.root / path, memory_map=True)
            if len(self._tables) > 2:
                self._tables.popitem(last=False)
        self._tables.move_to_end(path)
        return self._tables[path]

    def _image(self, episode: dict[str, Any], key: str, timestamp: float) -> np.ndarray:
        from torchcodec.decoders import VideoDecoder

        path = self.info["video_path"].format(
            video_key=key,
            chunk_index=episode[f"videos/{key}/chunk_index"],
            file_index=episode[f"videos/{key}/file_index"],
        )
        if path not in self._decoders:
            self._decoders[path] = VideoDecoder(
                str(self.root / path),
                device="cpu",
                seek_mode="approximate",
                num_ffmpeg_threads=1,
            )
            if len(self._decoders) > 6:
                self._decoders.popitem(last=False)
        self._decoders.move_to_end(path)
        decoder = self._decoders[path]
        query = float(episode[f"videos/{key}/from_timestamp"]) + timestamp
        fps = decoder.metadata.average_fps
        if fps is None or abs(fps - self.fps) > 1e-3:
            raise ValueError(
                f"Video fps {fps} differs from dataset fps {self.fps}: {path}"
            )
        frame = decoder.get_frame_at(round(query * fps))
        if abs(float(frame.pts_seconds) - query) > 1e-3:
            raise ValueError(f"Video timestamp mismatch at {query:.6f}s: {path}")
        return frame.data.numpy().astype(np.float32) / 255.0

    def __getitem__(self, index: int) -> dict[str, Any]:
        absolute = int(self.frame_indices[index])
        position = int(np.searchsorted(self._starts, absolute, side="right") - 1)
        episode = self.episodes[self.episode_ids[position]]
        table = self._table(episode)
        offset = int(table["index"][0].as_py())
        local = absolute - offset
        row = table.slice(local, 1).to_pylist()[0]
        if (
            int(row["index"]) != absolute
            or int(row["episode_index"]) != self.episode_ids[position]
        ):
            raise ValueError(
                f"Parquet frame {absolute} disagrees with episode metadata"
            )
        end = min(absolute + self.action_horizon, int(episode["dataset_to_index"]))
        chunk = table.slice(local, end - absolute)
        if len(chunk) != end - absolute:
            raise ValueError(
                "An episode's action chunk extends beyond its parquet file"
            )
        if int(chunk["episode_index"][-1].as_py()) != self.episode_ids[position]:
            raise ValueError("Action chunk crosses an episode boundary")
        actions = np.asarray(chunk["action"].to_pylist(), dtype=np.float32)
        padding = self.action_horizon - len(actions)
        if padding:
            actions = np.concatenate(
                [actions, np.repeat(actions[-1:], padding, axis=0)]
            )
        item = {
            key: np.asarray(
                value, dtype=self.info["features"].get(key, {}).get("dtype")
            )
            for key, value in row.items()
        }
        item["action"] = actions
        item["action_is_pad"] = np.arange(self.action_horizon) >= end - absolute
        item["prompt"] = item["task"] = self.tasks[int(row["task_index"])]
        for key in self.video_keys:
            item[key] = self._image(episode, key, float(row["timestamp"]))
        return item
