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

"""Score bounded shards with RLinf's RECAP implementation. A complete parquet is the restart unit."""

import argparse
import bisect
import fcntl
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from omegaconf import OmegaConf
from torchcodec.decoders import VideoDecoder

sys.path.insert(
    0,
    str(
        Path(os.environ["REPO_PATH"])
        / "examples/offline_rl/advantage_labeling/recap/process"
    ),
)
import compute_advantages as recap

from rlinf.models.embodiment.value_model.recap.modeling_critic import ValueCriticModel


class VideoRows:
    """Fetch consecutive frames together while preserving original LeRobot row keys."""

    cameras = (
        "observation.images.cam_left_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
    )

    def __init__(self, root, block):
        self.root = root
        self.info = json.loads((root / "meta/info.json").read_text())
        episodes = list(
            map(json.loads, (root / "meta/episodes.jsonl").read_text().splitlines())
        )
        assert [x["episode_index"] for x in episodes] == list(range(len(episodes)))
        self.ends = np.cumsum([x["length"] for x in episodes]).tolist()
        self.episode_data_index = {"from": [0] + self.ends[:-1], "to": self.ends}
        self.tasks = {
            t["task_index"]: t["task"]
            for t in map(
                json.loads, (root / "meta/tasks.jsonl").read_text().splitlines()
            )
        }
        self.block = block
        self.episode = None
        self.batch = None

    def __len__(self):
        return self.ends[-1]

    def __getitem__(self, row):
        ep = bisect.bisect_right(self.ends, row)
        i = row - self.episode_data_index["from"][ep]
        if self.episode != ep:
            args = {
                "episode_chunk": ep // self.info["chunks_size"],
                "episode_index": ep,
            }
            tab = pq.read_table(
                self.root / self.info["data_path"].format(**args),
                columns=[
                    "observation.state",
                    "timestamp",
                    "frame_index",
                    "task_index",
                    "episode_index",
                ],
            ).to_pydict()
            self.states = np.asarray(tab["observation.state"], dtype=np.float32)
            self.timestamps = np.asarray(tab["timestamp"]).reshape(-1)
            self.task_ids = np.asarray(tab["task_index"]).reshape(-1)
            assert np.all(np.asarray(tab["episode_index"]) == ep)
            assert np.array_equal(
                np.asarray(tab["frame_index"]).reshape(-1), np.arange(len(self.states))
            )
            self.decoders = {
                c: VideoDecoder(
                    str(
                        self.root / self.info["video_path"].format(video_key=c, **args)
                    ),
                    dimension_order="NCHW",
                    num_ffmpeg_threads=1,
                )
                for c in self.cameras
            }
            self.episode, self.batch = ep, None
        group = i // self.block
        if self.batch != group:
            start = group * self.block
            stop = min(start + self.block, len(self.states))
            self.frames = {}
            for c, decoder in self.decoders.items():
                chunk = decoder.get_frames_in_range(start, stop)
                np.testing.assert_allclose(
                    chunk.pts_seconds.numpy(),
                    self.timestamps[start:stop],
                    atol=0.0005,
                    rtol=0,
                )
                self.frames[c] = chunk.data
            self.batch = group
        sample = {c: data[i % self.block] for c, data in self.frames.items()}
        sample.update(
            {
                "episode_index": ep,
                "frame_index": i,
                "observation.state": self.states[i],
                "task": self.tasks[int(self.task_ids[i])],
            }
        )
        return sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    ap.add_argument("output", type=Path)
    args = ap.parse_args()
    cfg = OmegaConf.load(args.config)
    model = ValueCriticModel.from_checkpoint(
        **recap._parse_value_model_kwargs(cfg), device="cuda"
    )
    for spec in sorted(
        cfg.data.train_data_paths, key=lambda x: (x.type == "sft", x.dataset_path)
    ):
        root = Path(spec.dataset_path)
        folder = args.output / root.name
        folder.mkdir(parents=True, exist_ok=True)
        dataset = VideoRows(root, cfg.advantage.batch_size)
        side = recap.load_returns_sidecar(root, returns_tag=cfg.advantage.returns_tag)
        n = math.ceil(len(dataset) / 4096)
        for shard in range(n):
            dest = folder / f"{shard:04d}.parquet"
            with (folder / f"{shard:04d}.lock").open("a") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                if dest.exists():
                    continue
                print("scoring", root.name, shard, n, flush=True)
                with (
                    torch.inference_mode(),
                    torch.autocast("cuda", dtype=torch.bfloat16),
                ):
                    df = recap.compute_advantages_for_dataset(
                        model,
                        dataset,
                        dataset.tasks,
                        cfg,
                        dict(spec),
                        None,
                        rank=shard,
                        world_size=n,
                        global_return_min=cfg.data.return_min,
                        global_return_max=cfg.data.return_max,
                        returns_sidecar=side,
                    )
                start, stop = recap.get_shard_indices(len(dataset), shard, n)
                assert (
                    len(df) == stop - start
                    and np.isfinite(df.select_dtypes("number")).all().all()
                )
                assert np.array_equal(
                    np.take(dataset.episode_data_index["from"], df.episode_index)
                    + df.frame_index,
                    np.arange(start, stop),
                )
                temp = dest.with_suffix(".partial")
                df.to_parquet(temp, index=False)
                os.replace(temp, dest)
                print("saved", root.name, shard, len(df), flush=True)


if __name__ == "__main__":
    main()
