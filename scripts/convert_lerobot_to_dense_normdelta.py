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

"""Convert LeRobot Piper episodes into dense normalized-delta replay files.

The saved Piper RLToken/TD3 replay buffer expects high-level chunk samples:

    curr_obs at anchor frame t
    actions  = 10 absolute low-level actions starting at t, converted to
               OpenPI pi0.5 normalized-delta action space
    next_obs = observation after the chunk, approximated by frame t + horizon

This converter uses every frame in the selected tail window as an anchor. For a
last-20-chunk policy with horizon=10 this yields up to 200 training samples per
LeRobot episode, instead of one sample per 10 executed low-level steps.
"""

from __future__ import annotations

import argparse
import json
import shutil
import uuid
from pathlib import Path

import av
import cv2
import numpy as np
import pyarrow.parquet as pq
import torch

DELTA_MASK = np.array([True] * 6 + [False] + [True] * 6 + [False], dtype=bool)
DEFAULT_MODEL_WEIGHTS_ID = str(
    uuid.uuid5(uuid.NAMESPACE_DNS, "lerobot-cube-insert-dense-normdelta")
)


def load_action_stats(
    norm_stats_path: Path, std_floor: float
) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(norm_stats_path.read_text())
    stats = payload.get("norm_stats", payload)["actions"]
    mean = np.asarray(stats["mean"][:14], dtype=np.float32)
    std = np.asarray(stats["std"][:14], dtype=np.float32)
    std = np.where(np.abs(std) < std_floor, 1.0, std).astype(np.float32)
    return mean, std


def resize_rgb(frame_rgb: np.ndarray, image_size: int) -> np.ndarray:
    return cv2.resize(frame_rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)


def read_video_tail(
    video_path: Path, start_frame: int, length: int, image_size: int
) -> np.ndarray:
    frames: list[np.ndarray] = []
    try:
        container = av.open(str(video_path))
    except Exception as exc:
        raise FileNotFoundError(f"Cannot open video: {video_path}") from exc

    try:
        for idx, frame in enumerate(container.decode(video=0)):
            if idx < start_frame:
                continue
            if idx >= start_frame + length:
                break
            frame_rgb = frame.to_ndarray(format="rgb24")
            frames.append(resize_rgb(frame_rgb, image_size))
    finally:
        container.close()

    if not frames:
        raise RuntimeError(
            f"No frames read from {video_path} starting at {start_frame}"
        )
    while len(frames) < length:
        frames.append(frames[-1].copy())
    return np.stack(frames, axis=0).astype(np.uint8)


def get_column_matrix(table: pq.Table, key: str, width: int) -> np.ndarray:
    values = table[key].to_pylist()
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < width:
        raise ValueError(f"{key} has shape {arr.shape}, expected [T, >= {width}]")
    return arr[:, :width].astype(np.float32, copy=False)


def action_chunk_for_anchor(
    actions: np.ndarray, anchor: int, horizon: int
) -> np.ndarray:
    end = min(anchor + horizon, actions.shape[0])
    chunk = actions[anchor:end]
    if chunk.shape[0] < horizon:
        pad = np.repeat(actions[-1:], horizon - chunk.shape[0], axis=0)
        chunk = np.concatenate([chunk, pad], axis=0)
    return chunk.astype(np.float32, copy=False)


def absolute_to_normdelta_chunk(
    absolute_chunk: np.ndarray,
    state: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    delta = absolute_chunk.astype(np.float32).copy()
    delta[:, DELTA_MASK] -= state[None, DELTA_MASK]
    norm = (delta - mean[None, :]) / (std[None, :] + 1e-6)
    return norm.astype(np.float32), delta.astype(np.float32)


def build_episode_trajectory(
    *,
    lerobot_dir: Path,
    episode_index: int,
    length: int,
    out_path: Path,
    action_mean: np.ndarray,
    action_std: np.ndarray,
    action_horizon: int,
    tail_chunks: int,
    image_size: int,
    gamma: float,
    model_weights_id: str,
) -> dict:
    chunk_id = episode_index // 1000
    parquet_path = (
        lerobot_dir
        / "data"
        / f"chunk-{chunk_id:03d}"
        / f"episode_{episode_index:06d}.parquet"
    )
    table = pq.read_table(parquet_path)
    states_all = get_column_matrix(table, "observation.state", 14)
    actions_all = get_column_matrix(table, "action", 14)
    if length != states_all.shape[0]:
        length = int(states_all.shape[0])

    tail_frames = max(1, int(tail_chunks) * int(action_horizon))
    start = max(0, length - tail_frames)
    anchor_indices = np.arange(start, length, dtype=np.int64)
    num_samples = int(anchor_indices.shape[0])
    tail_len = length - start

    views = {
        "cam_high": read_video_tail(
            lerobot_dir
            / "videos"
            / f"chunk-{chunk_id:03d}"
            / "observation.images.cam_high"
            / f"episode_{episode_index:06d}.mp4",
            start,
            tail_len,
            image_size,
        ),
        "cam_left_wrist": read_video_tail(
            lerobot_dir
            / "videos"
            / f"chunk-{chunk_id:03d}"
            / "observation.images.cam_left_wrist"
            / f"episode_{episode_index:06d}.mp4",
            start,
            tail_len,
            image_size,
        ),
        "cam_right_wrist": read_video_tail(
            lerobot_dir
            / "videos"
            / f"chunk-{chunk_id:03d}"
            / "observation.images.cam_right_wrist"
            / f"episode_{episode_index:06d}.mp4",
            start,
            tail_len,
            image_size,
        ),
    }

    curr_states = states_all[anchor_indices]
    next_indices = np.minimum(anchor_indices + action_horizon, length - 1)
    next_states = states_all[next_indices]
    curr_offsets = anchor_indices - start
    next_offsets = next_indices - start

    norm_chunks = np.empty((num_samples, action_horizon, 14), dtype=np.float32)
    delta_chunks = np.empty_like(norm_chunks)
    abs_chunks = np.empty_like(norm_chunks)
    for out_i, anchor in enumerate(anchor_indices):
        absolute_chunk = action_chunk_for_anchor(
            actions_all, int(anchor), action_horizon
        )
        norm_chunk, delta_chunk = absolute_to_normdelta_chunk(
            absolute_chunk, curr_states[out_i], action_mean, action_std
        )
        norm_chunks[out_i] = norm_chunk
        delta_chunks[out_i] = delta_chunk
        abs_chunks[out_i] = absolute_chunk

    rewards = np.zeros((num_samples, 1, 1), dtype=np.float32)
    dones = np.zeros((num_samples, 1, 1), dtype=np.bool_)
    terminal_mask = (anchor_indices + action_horizon) >= (length - 1)
    rewards[terminal_mask, 0, 0] = 1.0
    dones[terminal_mask, 0, 0] = True

    mc_return = np.zeros_like(rewards, dtype=np.float32)
    running = np.zeros((1, 1), dtype=np.float32)
    for i in range(num_samples - 1, -1, -1):
        future = np.zeros_like(running) if dones[i] else running
        running = rewards[i] + float(gamma) * future
        mc_return[i] = running

    curr_main = views["cam_high"][curr_offsets][:, None]
    curr_extra = np.stack(
        [views["cam_left_wrist"][curr_offsets], views["cam_right_wrist"][curr_offsets]],
        axis=1,
    )[:, None]
    next_main = views["cam_high"][next_offsets][:, None]
    next_extra = np.stack(
        [views["cam_left_wrist"][next_offsets], views["cam_right_wrist"][next_offsets]],
        axis=1,
    )[:, None]

    trajectory = {
        "max_episode_length": int(length),
        "model_weights_id": model_weights_id,
        "actions": torch.from_numpy(norm_chunks.reshape(num_samples, 1, -1)),
        "intervene_flags": torch.zeros(
            (num_samples, 1, action_horizon * 14), dtype=torch.bool
        ),
        "rewards": torch.from_numpy(rewards),
        "terminations": torch.from_numpy(dones.copy()),
        "truncations": torch.zeros((num_samples, 1, 1), dtype=torch.bool),
        "dones": torch.from_numpy(dones.copy()),
        "forward_inputs": {
            "action": torch.from_numpy(norm_chunks.reshape(num_samples, 1, -1)),
            "model_action": torch.from_numpy(norm_chunks.reshape(num_samples, 1, -1)),
            "ref_action": torch.from_numpy(norm_chunks.reshape(num_samples, 1, -1)),
            "action_delta": torch.from_numpy(delta_chunks.reshape(num_samples, 1, -1)),
            "env_action_absolute": torch.from_numpy(
                abs_chunks.reshape(num_samples, 1, -1)
            ),
            "mc_return": torch.from_numpy(mc_return),
            "lerobot_episode_index": torch.full(
                (num_samples, 1, 1), episode_index, dtype=torch.int64
            ),
            "lerobot_frame_index": torch.from_numpy(
                anchor_indices.reshape(num_samples, 1, 1)
            ),
        },
        "curr_obs": {
            "states": torch.from_numpy(curr_states[:, None].astype(np.float32)),
            "main_images": torch.from_numpy(curr_main),
            "extra_view_images": torch.from_numpy(curr_extra),
            "ref_action": torch.from_numpy(norm_chunks.reshape(num_samples, 1, -1)),
        },
        "next_obs": {
            "states": torch.from_numpy(next_states[:, None].astype(np.float32)),
            "main_images": torch.from_numpy(next_main),
            "extra_view_images": torch.from_numpy(next_extra),
            "ref_action": torch.from_numpy(
                np.concatenate([norm_chunks[1:], norm_chunks[-1:]], axis=0).reshape(
                    num_samples, 1, -1
                )
            ),
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(trajectory, out_path)
    roundtrip_abs = delta_chunks.copy()
    roundtrip_abs[:, :, DELTA_MASK] += curr_states[:, None, DELTA_MASK]
    roundtrip_err = np.abs(roundtrip_abs - abs_chunks)
    return {
        "episode_index": int(episode_index),
        "length": int(length),
        "start_frame": int(start),
        "num_samples": int(num_samples),
        "reward_sum": float(rewards.sum()),
        "norm_min": float(norm_chunks.min()),
        "norm_max": float(norm_chunks.max()),
        "norm_mean": float(norm_chunks.mean()),
        "norm_std": float(norm_chunks.std()),
        "roundtrip_abs_max_err": float(roundtrip_err.max()),
        "file": out_path.name,
    }


def load_episodes(lerobot_dir: Path) -> list[dict]:
    episodes = []
    with (lerobot_dir / "meta" / "episodes.jsonl").open() as f:
        for line in f:
            if line.strip():
                episodes.append(json.loads(line))
    return episodes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lerobot-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--norm-stats", required=True)
    parser.add_argument("--tail-chunks", type=int, default=20)
    parser.add_argument("--action-horizon", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--gamma", type=float, default=0.89)
    parser.add_argument("--std-floor", type=float, default=1.0)
    parser.add_argument("--limit-episodes", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--model-weights-id", default=DEFAULT_MODEL_WEIGHTS_ID)
    args = parser.parse_args()

    lerobot_dir = Path(args.lerobot_dir)
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_dir} exists; pass --overwrite")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    action_mean, action_std = load_action_stats(Path(args.norm_stats), args.std_floor)
    episodes = load_episodes(lerobot_dir)
    if args.limit_episodes > 0:
        episodes = episodes[: args.limit_episodes]

    records = []
    trajectory_index = {}
    total_samples = 0
    for trajectory_id, episode in enumerate(episodes):
        episode_index = int(episode["episode_index"])
        out_path = output_dir / f"trajectory_{trajectory_id}_{args.model_weights_id}.pt"
        record = build_episode_trajectory(
            lerobot_dir=lerobot_dir,
            episode_index=episode_index,
            length=int(episode["length"]),
            out_path=out_path,
            action_mean=action_mean,
            action_std=action_std,
            action_horizon=args.action_horizon,
            tail_chunks=args.tail_chunks,
            image_size=args.image_size,
            gamma=args.gamma,
            model_weights_id=args.model_weights_id,
        )
        records.append(record)
        total_samples += record["num_samples"]
        trajectory_index[str(trajectory_id)] = {
            "num_samples": record["num_samples"],
            "trajectory_id": trajectory_id,
            "max_episode_length": record["length"],
            "shape": [record["num_samples"], 1, 1],
            "model_weights_id": args.model_weights_id,
        }
        print(
            f"[{trajectory_id + 1}/{len(episodes)}] episode={episode_index} "
            f"samples={record['num_samples']} norm=[{record['norm_min']:.3f}, "
            f"{record['norm_max']:.3f}] roundtrip={record['roundtrip_abs_max_err']:.3g}",
            flush=True,
        )

    metadata = {
        "trajectory_format": "pt",
        "size": len(records),
        "total_samples": int(total_samples),
        "trajectory_counter": len(records),
        "seed": 1234,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (output_dir / "trajectory_index.json").write_text(
        json.dumps(
            {
                "trajectory_index": trajectory_index,
                "trajectory_id_list": list(range(len(records))),
            },
            indent=2,
        )
    )
    summary = {
        "lerobot_dir": str(lerobot_dir),
        "output_dir": str(output_dir),
        "norm_stats": str(args.norm_stats),
        "tail_chunks": int(args.tail_chunks),
        "action_horizon": int(args.action_horizon),
        "tail_frames": int(args.tail_chunks * args.action_horizon),
        "image_size": int(args.image_size),
        "gamma": float(args.gamma),
        "std_floor": float(args.std_floor),
        "num_trajectories": len(records),
        "total_samples": int(total_samples),
        "norm_min": float(min(r["norm_min"] for r in records)),
        "norm_max": float(max(r["norm_max"] for r in records)),
        "roundtrip_abs_max_err": float(
            max(r["roundtrip_abs_max_err"] for r in records)
        ),
        "records": records,
    }
    (output_dir / "conversion_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
