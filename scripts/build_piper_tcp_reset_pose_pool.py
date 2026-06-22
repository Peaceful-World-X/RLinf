#!/usr/bin/env python3
# Copyright 2026 The GIGA Authors.
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

"""Build a Piper reset pose pool from offline absolute action chunks."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rlinf.utils.offline_td3_visualization import SimpleUrdfKinematics  # noqa: E402


def _natural_key(path: Path) -> list[object]:
    parts = re.split(r"(\d+)", path.name)
    return [int(part) if part.isdigit() else part for part in parts]


def _to_numpy(value) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _load_absolute_action_chunks(traj: dict, action_dim: int) -> np.ndarray:
    forward_inputs = traj.get("forward_inputs", {})
    candidates = (
        forward_inputs.get("env_action_absolute", None),
        forward_inputs.get("executed_action", None),
        traj.get("actions", None),
    )
    for candidate in candidates:
        if candidate is None:
            continue
        arr = _to_numpy(candidate).astype(np.float64)
        if arr.ndim >= 3 and arr.shape[1] == 1:
            arr = arr[:, 0]
        arr = arr.reshape(arr.shape[0], -1)
        if arr.shape[-1] % action_dim == 0:
            horizon = arr.shape[-1] // action_dim
            return arr.reshape(arr.shape[0], horizon, action_dim)
    raise KeyError(
        "Cannot find an absolute action chunk field. Expected "
        "forward_inputs.env_action_absolute, forward_inputs.executed_action, or actions."
    )


def _right_tcp(
    kinematics: SimpleUrdfKinematics,
    right_q: np.ndarray,
    base_offset: np.ndarray,
    tip_link: str,
) -> np.ndarray:
    links, _ = kinematics.fk_positions(right_q, tip_link=tip_link)
    return np.asarray(links[-1], dtype=np.float64) + base_offset


def build_pose_pool(args: argparse.Namespace) -> dict:
    data_dir = Path(args.data_dir)
    paths = sorted(data_dir.glob("*.pt"), key=_natural_key)
    if args.limit > 0:
        paths = paths[: args.limit]
    if not paths:
        raise FileNotFoundError(f"No .pt trajectories found under {data_dir}")

    kinematics = SimpleUrdfKinematics(args.urdf, base_link=args.base_link)
    tcp_target = np.asarray(args.tcp_target, dtype=np.float64)
    right_base_offset = np.asarray(args.right_base_offset, dtype=np.float64)

    poses: list[list[float]] = []
    metadata: list[dict] = []
    skipped: list[str] = []
    distances: list[float] = []
    within_count = 0

    for path in paths:
        try:
            traj = torch.load(path, map_location="cpu")
            chunks = _load_absolute_action_chunks(traj, args.action_dim)
        except Exception as exc:
            skipped.append(f"{path.name}: {exc}")
            continue

        best = None
        first_within = None
        for chunk_idx in range(chunks.shape[0]):
            for substep_idx in range(chunks.shape[1]):
                q = np.asarray(chunks[chunk_idx, substep_idx], dtype=np.float64)
                tcp = _right_tcp(
                    kinematics,
                    q[7:14],
                    right_base_offset,
                    args.tip_link,
                )
                dist = float(np.linalg.norm(tcp - tcp_target))
                item = (dist, chunk_idx, substep_idx, q, tcp)
                if best is None or dist < best[0]:
                    best = item
                if dist <= float(args.radius):
                    first_within = item
                    break
            if first_within is not None:
                break

        selected = first_within
        within_radius = True
        if selected is None:
            if not args.fallback_nearest:
                skipped.append(f"{path.name}: no TCP point within radius")
                continue
            selected = best
            within_radius = False
        if selected is None:
            skipped.append(f"{path.name}: empty action chunk")
            continue

        dist, chunk_idx, substep_idx, pose, tcp = selected
        reward_sum = float(_to_numpy(traj.get("rewards", torch.zeros(1))).sum())
        pose = np.asarray(pose, dtype=np.float64).reshape(args.action_dim)
        if pose.shape[0] < 14:
            skipped.append(f"{path.name}: action_dim={pose.shape[0]} is less than 14")
            continue
        pose = pose[:14].copy()
        if args.right_gripper_override is not None:
            pose[13] = float(args.right_gripper_override)

        poses.append([float(v) for v in pose])
        distances.append(float(dist))
        within_count += int(within_radius)
        metadata.append(
            {
                "trajectory": path.name,
                "trajectory_path": str(path),
                "reward_sum": reward_sum,
                "success": bool(reward_sum > 0.0),
                "chunk_index": int(chunk_idx),
                "substep_index": int(substep_idx),
                "tcp_distance": float(dist),
                "within_radius": bool(within_radius),
                "right_tcp": [float(v) for v in tcp],
                "left_gripper": float(pose[6]),
                "right_gripper": float(pose[13]),
            }
        )

    if not poses:
        raise RuntimeError(f"No usable reset poses built from {data_dir}")

    distances_np = np.asarray(distances, dtype=np.float64)
    right_grippers = np.asarray([pose[13] for pose in poses], dtype=np.float64)
    return {
        "source_dir": str(data_dir),
        "urdf": str(args.urdf),
        "base_link": args.base_link,
        "tip_link": args.tip_link,
        "tcp_target": [float(v) for v in tcp_target],
        "radius": float(args.radius),
        "right_base_offset": [float(v) for v in right_base_offset],
        "num_poses": len(poses),
        "num_within_radius": int(within_count),
        "distance_stats": {
            "min": float(distances_np.min()),
            "mean": float(distances_np.mean()),
            "max": float(distances_np.max()),
        },
        "right_gripper_stats": {
            "min": float(right_grippers.min()),
            "mean": float(right_grippers.mean()),
            "max": float(right_grippers.max()),
        },
        "poses": poses,
        "metadata": metadata,
        "skipped": skipped,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--urdf", required=True)
    parser.add_argument("--base-link", default="base_link")
    parser.add_argument("--tip-link", default="gripper")
    parser.add_argument("--tcp-target", nargs=3, type=float, required=True)
    parser.add_argument("--radius", type=float, default=0.10)
    parser.add_argument(
        "--right-base-offset", nargs=3, type=float, default=[0.0, -0.22, 0.0]
    )
    parser.add_argument("--action-dim", type=int, default=14)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--fallback-nearest", action="store_true")
    parser.add_argument("--right-gripper-override", type=float, default=None)
    args = parser.parse_args()

    result = build_pose_pool(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(result, f, indent=2)

    print(
        "Saved reset pose pool: "
        f"{output} poses={result['num_poses']} "
        f"within_radius={result['num_within_radius']} "
        f"distance_stats={result['distance_stats']} "
        f"right_gripper_stats={result['right_gripper_stats']} "
        f"skipped={len(result['skipped'])}",
        flush=True,
    )
    if result["skipped"]:
        print("Skipped examples:", result["skipped"][:5], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
