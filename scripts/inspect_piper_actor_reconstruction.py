#!/usr/bin/env python3
# Copyright 2025 The RLinf Authors.
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

"""Interactively inspect PT/GT/actor absolute joint chunks.

This is a dry-run numeric viewer for validation ``metrics.json`` files. It does
not create a robot environment and never sends commands to hardware.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

JOINT_NAMES = [
    "L_j1",
    "L_j2",
    "L_j3",
    "L_j4",
    "L_j5",
    "L_j6",
    "L_grip",
    "R_j1",
    "R_j2",
    "R_j3",
    "R_j4",
    "R_j5",
    "R_j6",
    "R_grip",
]


def _parse_indices(text: str, max_len: int, *, default_all: bool = False) -> list[int]:
    text = str(text).strip().lower()
    if text in {"all", "*"}:
        return list(range(max_len))
    if text == "" and default_all:
        return list(range(max_len))

    indices: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            start = int(left)
            end = int(right)
            if end < start:
                raise ValueError(f"Bad descending range {part!r}")
            indices.extend(range(start, end + 1))
        else:
            indices.append(int(part))

    bad = [idx for idx in indices if idx < 0 or idx >= max_len]
    if bad:
        raise ValueError(f"indices out of range [0, {max_len - 1}]: {bad}")
    return sorted(dict.fromkeys(indices))


def _array(payload: dict[str, Any], key: str) -> np.ndarray:
    if key not in payload:
        raise KeyError(f"metrics.json does not contain {key!r}")
    arr = np.asarray(payload[key], dtype=np.float64)
    if arr.ndim != 3 or arr.shape[-1] != 14:
        raise ValueError(
            f"{key} must have shape [chunks, substeps, 14], got {arr.shape}"
        )
    return arr


def _format_vec(vec: np.ndarray, precision: int) -> str:
    return np.array2string(
        np.asarray(vec, dtype=np.float64),
        precision=precision,
        suppress_small=True,
        max_line_width=220,
    )


def _summarize(name: str, delta: np.ndarray) -> str:
    delta = np.asarray(delta, dtype=np.float64).reshape(14)
    return (
        f"{name}: max_abs={np.max(np.abs(delta)):.6f}, "
        f"l2={np.linalg.norm(delta):.6f}, "
        f"left_max={np.max(np.abs(delta[:7])):.6f}, "
        f"right_max={np.max(np.abs(delta[7:])):.6f}"
    )


def _print_table(
    *,
    pt: np.ndarray,
    gt: np.ndarray,
    actor: np.ndarray,
    precision: int,
) -> None:
    print(
        "dim      joint       PT_absolute     GT_rebuilt      Actor_rebuilt   "
        "Actor-GT      PT-GT"
    )
    print("-" * 88)
    for dim, name in enumerate(JOINT_NAMES):
        print(
            f"{dim:02d}       {name:<7} "
            f"{pt[dim]:>13.{precision}f} "
            f"{gt[dim]:>13.{precision}f} "
            f"{actor[dim]:>14.{precision}f} "
            f"{(actor[dim] - gt[dim]):>11.{precision}f} "
            f"{(pt[dim] - gt[dim]):>11.{precision}f}"
        )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay-style numeric inspection of PT absolute, GT reconstructed, "
            "and actor reconstructed joint chunks from validation metrics.json."
        )
    )
    parser.add_argument(
        "--metrics", required=True, help="Path to validation metrics.json"
    )
    parser.add_argument(
        "--chunks",
        default="all",
        help="Chunk positions to inspect, e.g. 0, 10-15, all. These are positions in metrics.json.",
    )
    parser.add_argument(
        "--substeps",
        default="all",
        help="Substeps inside each chunk, e.g. 0, all, 0-9.",
    )
    parser.add_argument(
        "--top-errors",
        type=int,
        default=0,
        help="If >0, inspect the N chunks with largest Actor-GT max error.",
    )
    parser.add_argument(
        "--table",
        action="store_true",
        help="Print one row per joint instead of compact vectors.",
    )
    parser.add_argument("--precision", type=int, default=5)
    parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="Print selected entries without waiting for Enter.",
    )
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    metrics_path = Path(args.metrics).expanduser()
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    pt = _array(metrics, "full_pt_absolute_action_chunks")
    gt = _array(metrics, "gt_reconstructed_absolute_action_chunks")
    actor = _array(metrics, "actor_absolute_action_chunks")
    if not (pt.shape == gt.shape == actor.shape):
        raise ValueError(
            f"PT/GT/actor shapes differ: {pt.shape}, {gt.shape}, {actor.shape}"
        )

    chunk_indices = metrics.get("chunk_indices", list(range(pt.shape[0])))
    rewards = metrics.get("rewards", None)
    pt_gt = pt - gt
    actor_gt = actor - gt
    actor_pt = actor - pt

    if args.top_errors > 0:
        per_chunk_max = np.max(np.abs(actor_gt), axis=(1, 2))
        chunks = np.argsort(per_chunk_max)[::-1][: int(args.top_errors)].tolist()
    else:
        chunks = _parse_indices(args.chunks, pt.shape[0])
    substeps = _parse_indices(args.substeps, pt.shape[1], default_all=True)

    print("Loaded metrics:", metrics_path)
    print("trajectory:", metrics.get("trajectory_file", "<unknown>"))
    print("shape [chunks, substeps, joints]:", pt.shape)
    print(
        _summarize(
            "overall PT-GT",
            pt_gt.reshape(-1, 14)[
                np.argmax(np.max(np.abs(pt_gt.reshape(-1, 14)), axis=1))
            ],
        )
    )
    print(
        f"overall actor-GT: mse={np.mean(actor_gt**2):.9g}, "
        f"mean_abs={np.mean(np.abs(actor_gt)):.9g}, "
        f"max_abs={np.max(np.abs(actor_gt)):.9g}"
    )
    print("selected chunk positions:", chunks)
    print("selected substeps:", substeps)
    print(
        "joint order:", ", ".join(f"{i}:{name}" for i, name in enumerate(JOINT_NAMES))
    )

    for chunk_pos in chunks:
        original_chunk = (
            chunk_indices[chunk_pos] if chunk_pos < len(chunk_indices) else chunk_pos
        )
        chunk_err = np.abs(actor_gt[chunk_pos])
        max_substep, max_dim = np.unravel_index(
            int(np.argmax(chunk_err)), chunk_err.shape
        )
        print("\n" + "=" * 110)
        print(
            f"chunk_pos={chunk_pos}, original_chunk_index={original_chunk}, "
            f"chunk_actor_gt_mse={np.mean(actor_gt[chunk_pos] ** 2):.9g}, "
            f"chunk_actor_gt_mean_abs={np.mean(np.abs(actor_gt[chunk_pos])):.9g}, "
            f"chunk_actor_gt_max_abs={chunk_err[max_substep, max_dim]:.9g} "
            f"at substep={max_substep}, dim={max_dim}/{JOINT_NAMES[max_dim]}"
        )
        if rewards is not None and chunk_pos < len(rewards):
            print("reward:", rewards[chunk_pos])

        for substep in substeps:
            print("\n" + "-" * 110)
            print(
                f"chunk_pos={chunk_pos}, original_chunk_index={original_chunk}, substep={substep}"
            )
            print(_summarize("PT-GT", pt_gt[chunk_pos, substep]))
            print(_summarize("Actor-GT", actor_gt[chunk_pos, substep]))
            print(_summarize("Actor-PT", actor_pt[chunk_pos, substep]))
            if args.table:
                _print_table(
                    pt=pt[chunk_pos, substep],
                    gt=gt[chunk_pos, substep],
                    actor=actor[chunk_pos, substep],
                    precision=args.precision,
                )
            else:
                print(
                    "PT absolute     :",
                    _format_vec(pt[chunk_pos, substep], args.precision),
                )
                print(
                    "GT reconstructed:",
                    _format_vec(gt[chunk_pos, substep], args.precision),
                )
                print(
                    "Actor rebuilt   :",
                    _format_vec(actor[chunk_pos, substep], args.precision),
                )
                print(
                    "Actor - GT      :",
                    _format_vec(actor_gt[chunk_pos, substep], args.precision),
                )
                print(
                    "PT - GT         :",
                    _format_vec(pt_gt[chunk_pos, substep], args.precision),
                )

            if not args.no_prompt:
                response = (
                    input("Enter=next, p=table for this step, q=quit > ")
                    .strip()
                    .lower()
                )
                if response in {"q", "quit"}:
                    return 0
                if response == "p" and not args.table:
                    _print_table(
                        pt=pt[chunk_pos, substep],
                        gt=gt[chunk_pos, substep],
                        actor=actor[chunk_pos, substep],
                        precision=args.precision,
                    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
