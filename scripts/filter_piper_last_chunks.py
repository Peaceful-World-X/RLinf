#!/usr/bin/env python3
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

"""Keep the last N high-level chunks from each Piper trajectory."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch


def slice_tree(value: Any, start: int, end: int, length: int):
    if torch.is_tensor(value):
        if value.shape and int(value.shape[0]) == length:
            return value[start:end].clone()
        return value
    if isinstance(value, dict):
        return {k: slice_tree(v, start, end, length) for k, v in value.items()}
    if isinstance(value, list):
        if len(value) == length:
            return value[start:end]
        return [slice_tree(v, start, end, length) for v in value]
    return value


def reward_sum(trajectory: dict) -> float:
    rewards = trajectory.get("rewards", None)
    if torch.is_tensor(rewards):
        return float(rewards.float().sum().item())
    return 0.0


def filter_file(src: Path, dst: Path, keep_chunks: int) -> dict:
    trajectory = torch.load(src, map_location="cpu", weights_only=False)
    length = int(trajectory["actions"].shape[0])
    start = max(0, length - int(keep_chunks))
    end = length
    filtered = slice_tree(trajectory, start, end, length)
    filtered["max_episode_length"] = int(end - start)
    filtered["last_chunk_segment"] = {
        "source_file": str(src),
        "source_length": length,
        "start_chunk": int(start),
        "end_chunk": int(end),
        "kept_chunks": int(end - start),
    }
    torch.save(filtered, dst)
    return {
        "file": src.name,
        "source_length": length,
        "start_chunk": int(start),
        "end_chunk": int(end),
        "kept_chunks": int(end - start),
        "source_reward_sum": reward_sum(trajectory),
        "kept_reward_sum": reward_sum(filtered),
    }


def write_trajectory_index(input_dir: Path, output_dir: Path, summaries: list[dict]):
    index_path = input_dir / "trajectory_index.json"
    if not index_path.exists():
        return

    index_data = json.loads(index_path.read_text(encoding="utf-8"))
    source_index = {
        int(k): v for k, v in index_data.get("trajectory_index", {}).items()
    }
    trajectory_index = {}
    trajectory_id_list = []
    summary_by_file = {row["file"]: row for row in summaries}

    for src in sorted(output_dir.glob("trajectory_*.pt")):
        parts = src.stem.split("_")
        if len(parts) < 3:
            continue
        trajectory_id = int(parts[1])
        model_weights_id = "_".join(parts[2:])
        row = summary_by_file.get(src.name, {})
        kept_chunks = int(row.get("kept_chunks", 0))

        info = dict(source_index.get(trajectory_id, {}))
        info.update(
            {
                "num_samples": kept_chunks,
                "trajectory_id": trajectory_id,
                "max_episode_length": kept_chunks,
                "shape": [kept_chunks, 1, 1],
                "model_weights_id": model_weights_id,
            }
        )
        trajectory_index[str(trajectory_id)] = info
        trajectory_id_list.append(trajectory_id)

    output = {
        "trajectory_index": trajectory_index,
        "trajectory_id_list": trajectory_id_list,
    }
    (output_dir / "trajectory_index.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--keep-chunks", type=int, default=20)
    parser.add_argument("--gamma", type=float, default=0.89)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.keep_chunks <= 0:
        raise ValueError("--keep-chunks must be positive")
    if args.output.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output} exists; pass --overwrite")
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True, exist_ok=True)

    summaries = [
        filter_file(src, args.output / src.name, args.keep_chunks)
        for src in sorted(args.input.glob("trajectory_*.pt"))
    ]

    for extra in (
        "metadata.json",
        "conversion_summary.json",
        "reward_fix_summary.json",
    ):
        src = args.input / extra
        if src.exists():
            shutil.copy2(src, args.output / extra)
    write_trajectory_index(args.input, args.output, summaries)

    kept = np.asarray([row["kept_chunks"] for row in summaries], dtype=np.float64)
    starts = np.asarray([row["start_chunk"] for row in summaries], dtype=np.float64)
    rewards = np.asarray(
        [row["kept_reward_sum"] for row in summaries], dtype=np.float64
    )
    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "keep_chunks": int(args.keep_chunks),
        "gamma_for_experiments": float(args.gamma),
        "num_trajectories": len(summaries),
        "start_chunk_mean": float(starts.mean()) if len(starts) else 0.0,
        "start_chunk_min": int(starts.min()) if len(starts) else 0,
        "start_chunk_max": int(starts.max()) if len(starts) else 0,
        "kept_chunks_mean": float(kept.mean()) if len(kept) else 0.0,
        "kept_chunks_min": int(kept.min()) if len(kept) else 0,
        "kept_chunks_max": int(kept.max()) if len(kept) else 0,
        "kept_reward_sum_min": float(rewards.min()) if len(rewards) else 0.0,
        "kept_reward_sum_max": float(rewards.max()) if len(rewards) else 0.0,
        "kept_reward_sum_mean": float(rewards.mean()) if len(rewards) else 0.0,
        "trajectories": summaries,
    }
    (args.output / "last_chunk_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(
        json.dumps({k: v for k, v in summary.items() if k != "trajectories"}, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
