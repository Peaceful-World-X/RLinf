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

"""Replay PT/GT/actor absolute chunks saved in validation metrics.

This script reuses the conservative stepping logic from
``replay_piper_pt_trajectory.py``:

- dry-run by default;
- ``--execute`` requires typing ``EXECUTE`` before any hardware command;
- every target requires Enter/confirmation;
- large live/target mismatches require ``FORCE`` unless explicitly disabled.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.replay_piper_pt_trajectory import (  # noqa: E402
    _format_vec,
    _get_live_qpos,
    _go_to_recorded_start,
    _make_env,
    _parse_indices,
    _prompt,
    _step_env_with_report,
    _summarize_delta,
    _trajectory_1d,
    _write_log,
)

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


def _as_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().float().numpy()
    return np.asarray(value)


def _load_metrics(path: str) -> dict[str, Any]:
    metrics_path = Path(path).expanduser()
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_trajectory(path: str) -> dict[str, Any]:
    trajectory_path = Path(path).expanduser()
    if not trajectory_path.is_file():
        raise FileNotFoundError(trajectory_path)
    trajectory = torch.load(trajectory_path, map_location="cpu", weights_only=False)
    if "curr_obs" not in trajectory or "states" not in trajectory["curr_obs"]:
        raise KeyError("Trajectory must contain curr_obs.states")
    return trajectory


def _metrics_array(metrics: dict[str, Any], key: str) -> np.ndarray:
    if key not in metrics:
        raise KeyError(f"metrics.json does not contain {key!r}")
    arr = np.asarray(metrics[key], dtype=np.float64)
    if arr.ndim != 3 or arr.shape[-1] != 14:
        raise ValueError(
            f"{key} must have shape [chunks, substeps, 14], got {arr.shape}"
        )
    return arr


def _select_metric_positions(
    *,
    chunks_text: str,
    chunk_index_mode: str,
    metric_chunk_indices: list[int],
) -> list[int]:
    if chunk_index_mode == "position":
        return _parse_indices(chunks_text, len(metric_chunk_indices), default_all=True)

    if str(chunks_text).strip().lower() in {"all", "*", ""}:
        return list(range(len(metric_chunk_indices)))

    max_original = max(metric_chunk_indices) + 1 if metric_chunk_indices else 0
    requested = _parse_indices(chunks_text, max_original, default_all=False)
    original_to_pos = {
        int(chunk_idx): pos for pos, chunk_idx in enumerate(metric_chunk_indices)
    }
    missing = [idx for idx in requested if idx not in original_to_pos]
    if missing:
        raise ValueError(
            f"Requested original chunk indices not present in metrics: {missing}. "
            f"Available range/list starts with {metric_chunk_indices[:20]}"
        )
    return [original_to_pos[idx] for idx in requested]


def _print_comparison(
    *,
    pt: np.ndarray,
    gt: np.ndarray,
    actor: np.ndarray,
    target_source: str,
    precision: int,
) -> None:
    print("PT absolute     :", _format_vec(pt, precision=precision))
    print("GT reconstructed:", _format_vec(gt, precision=precision))
    print("Actor rebuilt   :", _format_vec(actor, precision=precision))
    print(_summarize_delta("GT-PT", gt - pt))
    print(_summarize_delta("Actor-GT", actor - gt))
    print(_summarize_delta("Actor-PT", actor - pt))
    if target_source in {"pt", "gt", "actor"}:
        selected = {"pt": pt, "gt": gt, "actor": actor}[target_source]
        print(
            f"selected target ({target_source}):",
            _format_vec(selected, precision=precision),
        )


def _print_joint_table(
    pt: np.ndarray, gt: np.ndarray, actor: np.ndarray, precision: int
) -> None:
    print(
        "dim joint       PT_absolute     GT_rebuilt      Actor_rebuilt   "
        "Actor-GT      Actor-PT      GT-PT"
    )
    print("-" * 102)
    for dim, name in enumerate(JOINT_NAMES):
        print(
            f"{dim:02d}  {name:<7} "
            f"{pt[dim]:>13.{precision}f} "
            f"{gt[dim]:>13.{precision}f} "
            f"{actor[dim]:>14.{precision}f} "
            f"{(actor[dim] - gt[dim]):>11.{precision}f} "
            f"{(actor[dim] - pt[dim]):>11.{precision}f} "
            f"{(gt[dim] - pt[dim]):>11.{precision}f}"
        )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Safely replay PT/GT/actor absolute action chunks from validation metrics."
    )
    parser.add_argument(
        "--metrics", required=True, help="Path to validation metrics.json"
    )
    parser.add_argument(
        "--trajectory",
        default=None,
        help="Optional trajectory .pt override. Default uses metrics['trajectory_file'].",
    )
    parser.add_argument(
        "--target-source",
        choices=["pt", "gt", "actor", "state"],
        default="actor",
        help="Which absolute target to dry-run/replay.",
    )
    parser.add_argument(
        "--chunks",
        default="0",
        help="Chunk indices, e.g. 0, 10-15, all. Default: 0.",
    )
    parser.add_argument(
        "--chunk-index-mode",
        choices=["original", "position"],
        default="original",
        help=(
            "Interpret --chunks as original trajectory chunk indices or positions "
            "inside the metrics arrays."
        ),
    )
    parser.add_argument(
        "--substeps",
        default="all",
        help="Substeps inside each action chunk, e.g. 0, all, 0-9. Ignored for state target.",
    )
    parser.add_argument("--batch-index", type=int, default=0)
    parser.add_argument("--precision", type=int, default=4)
    parser.add_argument(
        "--table", action="store_true", help="Print per-joint comparison table."
    )
    parser.add_argument(
        "--execute", action="store_true", help="Actually command the real robot."
    )
    parser.add_argument(
        "--dummy", action="store_true", help="Use dummy env instead of hardware."
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Call env.reset() before replay. Default: do not move robot to reset pose.",
    )
    parser.add_argument("--max-env-steps", type=int, default=10000)
    parser.add_argument("--step-frequency", type=float, default=10.0)
    parser.add_argument("--joint-speed-pct", type=int, default=20)
    parser.add_argument("--delta-action-scale", type=float, default=0.01)
    parser.add_argument("--max-live-state-diff", type=float, default=0.25)
    parser.add_argument("--max-action-jump", type=float, default=0.35)
    parser.add_argument("--allow-large-diff", action="store_true")
    parser.add_argument(
        "--go-to-start",
        action="store_true",
        help="Stepwise confirmed interpolation to the first selected recorded state.",
    )
    parser.add_argument("--go-to-start-min-steps", type=int, default=1)
    parser.add_argument(
        "--start-at-nearest-state",
        action="store_true",
        help="After env creation, start from the recorded state nearest to live qpos.",
    )
    parser.add_argument("--sleep-after-step", type=float, default=0.0)
    parser.add_argument("--log-jsonl", default=None)
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    np.set_printoptions(precision=args.precision, suppress=True, linewidth=220)
    os.environ.setdefault("RLINF_SKIP_ROS_CLEANUP", "1")

    metrics = _load_metrics(args.metrics)
    trajectory_path = args.trajectory or metrics.get("trajectory_file", None)
    if not trajectory_path:
        raise ValueError(
            "--trajectory is required when metrics.json has no trajectory_file"
        )
    trajectory = _load_trajectory(trajectory_path)

    pt_chunks = _metrics_array(metrics, "full_pt_absolute_action_chunks")
    gt_chunks = _metrics_array(metrics, "gt_reconstructed_absolute_action_chunks")
    actor_chunks = _metrics_array(metrics, "actor_absolute_action_chunks")
    if not (pt_chunks.shape == gt_chunks.shape == actor_chunks.shape):
        raise ValueError(
            f"PT/GT/actor shapes differ: {pt_chunks.shape}, {gt_chunks.shape}, {actor_chunks.shape}"
        )

    metric_chunk_indices = [
        int(x) for x in metrics.get("chunk_indices", list(range(pt_chunks.shape[0])))
    ]
    selected_positions = _select_metric_positions(
        chunks_text=args.chunks,
        chunk_index_mode=args.chunk_index_mode,
        metric_chunk_indices=metric_chunk_indices,
    )
    substep_indices = _parse_indices(
        args.substeps, pt_chunks.shape[1], default_all=True
    )

    curr_states_all = _as_numpy(trajectory["curr_obs"]["states"]).astype(np.float64)
    if curr_states_all.ndim == 2:
        curr_states_all = curr_states_all[:, None, :]
    if curr_states_all.shape[1] <= args.batch_index:
        raise IndexError(
            f"batch_index={args.batch_index} but curr_obs.states batch dim is {curr_states_all.shape[1]}"
        )
    curr_states = curr_states_all[:, args.batch_index, :14]
    rewards = _trajectory_1d(trajectory, "rewards")
    dones = _trajectory_1d(trajectory, "dones")
    terminations = _trajectory_1d(trajectory, "terminations")
    truncations = _trajectory_1d(trajectory, "truncations")

    print("Loaded metrics:", Path(args.metrics))
    print("Loaded trajectory:", Path(trajectory_path))
    print(f"metrics shape={pt_chunks.shape}, target_source={args.target_source}")
    print(
        f"overall PT-GT max_abs={np.max(np.abs(pt_chunks - gt_chunks)):.9g}, "
        f"Actor-GT max_abs={np.max(np.abs(actor_chunks - gt_chunks)):.9g}, "
        f"Actor-GT mean_abs={np.mean(np.abs(actor_chunks - gt_chunks)):.9g}"
    )
    print("selected metric positions:", selected_positions)
    print(
        "selected original chunks:",
        [metric_chunk_indices[pos] for pos in selected_positions],
    )
    if args.target_source != "state":
        print("selected substeps:", substep_indices)

    env = None
    if args.execute:
        if not args.dummy:
            text = input(
                "\nThis will command the REAL Piper robot. Type EXECUTE to continue > "
            ).strip()
            if text != "EXECUTE":
                print("Aborted.")
                return 1
        env = _make_env(args)
        if args.reset:
            print("Calling env.reset() because --reset was provided...")
            env.reset()
    else:
        print("\nDRY-RUN only. Add --execute to command the robot.")

    if args.start_at_nearest_state:
        if env is None:
            print("--start-at-nearest-state ignored in dry-run mode.")
        else:
            live = _get_live_qpos(env)
            available_states = np.stack(
                [curr_states[metric_chunk_indices[pos]] for pos in selected_positions]
            )
            dists = np.linalg.norm(available_states - live.reshape(1, 14), axis=1)
            nearest_pos_idx = int(np.argmin(dists))
            selected_positions = selected_positions[nearest_pos_idx:]
            print(
                f"Nearest selected recorded state: original_chunk="
                f"{metric_chunk_indices[selected_positions[0]]}, l2={float(dists[nearest_pos_idx]):.5f}"
            )
            print(
                "updated original chunks:",
                [metric_chunk_indices[pos] for pos in selected_positions],
            )

    if args.go_to_start:
        if env is None:
            print("--go-to-start ignored in dry-run mode.")
        elif not selected_positions:
            raise ValueError("No chunks selected; cannot --go-to-start.")
        else:
            first_original = metric_chunk_indices[selected_positions[0]]
            _go_to_recorded_start(
                env=env,
                target_start=curr_states[first_original],
                chunk_idx=first_original,
                args=args,
            )

    try:
        for metric_pos in selected_positions:
            original_chunk = metric_chunk_indices[metric_pos]
            recorded_state = curr_states[original_chunk]
            reward = rewards[original_chunk] if original_chunk < len(rewards) else 0.0
            done = dones[original_chunk] if original_chunk < len(dones) else 0.0
            term = (
                terminations[original_chunk]
                if original_chunk < len(terminations)
                else 0.0
            )
            trunc = (
                truncations[original_chunk]
                if original_chunk < len(truncations)
                else 0.0
            )

            print("\n" + "=" * 110)
            print(
                f"metric_pos={metric_pos}/{len(metric_chunk_indices) - 1}, "
                f"original_chunk={original_chunk}, reward={float(reward):.4g}, "
                f"done={int(done)}, term={int(term)}, trunc={int(trunc)}"
            )
            print(
                "recorded curr_state:",
                _format_vec(recorded_state, precision=args.precision),
            )
            print(
                "PT first/last     :",
                _format_vec(pt_chunks[metric_pos, 0], precision=args.precision),
            )
            print(
                "                   ",
                _format_vec(pt_chunks[metric_pos, -1], precision=args.precision),
            )
            print(
                "Actor first/last  :",
                _format_vec(actor_chunks[metric_pos, 0], precision=args.precision),
            )
            print(
                "                   ",
                _format_vec(actor_chunks[metric_pos, -1], precision=args.precision),
            )
            print(
                _summarize_delta(
                    "PT first-recorded_state", pt_chunks[metric_pos, 0] - recorded_state
                )
            )
            print(
                _summarize_delta(
                    "Actor first-recorded_state",
                    actor_chunks[metric_pos, 0] - recorded_state,
                )
            )
            print(
                _summarize_delta(
                    "Actor-GT chunk",
                    actor_chunks[metric_pos].reshape(-1, 14)[
                        np.argmax(
                            np.max(
                                np.abs(
                                    actor_chunks[metric_pos] - gt_chunks[metric_pos]
                                ),
                                axis=1,
                            )
                        )
                    ]
                    - gt_chunks[metric_pos].reshape(-1, 14)[
                        np.argmax(
                            np.max(
                                np.abs(
                                    actor_chunks[metric_pos] - gt_chunks[metric_pos]
                                ),
                                axis=1,
                            )
                        )
                    ],
                )
            )

            if args.target_source == "state":
                steps = [(None, recorded_state)]
            else:
                source_chunks = {
                    "pt": pt_chunks,
                    "gt": gt_chunks,
                    "actor": actor_chunks,
                }[args.target_source]
                steps = [
                    (substep, source_chunks[metric_pos, substep])
                    for substep in substep_indices
                ]

            executed_targets = 0
            skipped_targets = 0
            for substep_idx, target in steps:
                if substep_idx is not None:
                    print("\n" + "-" * 110)
                    print(
                        f"[original chunk {original_chunk}, metric_pos {metric_pos}, "
                        f"substep {substep_idx}] comparison before target prompt"
                    )
                    _print_comparison(
                        pt=pt_chunks[metric_pos, substep_idx],
                        gt=gt_chunks[metric_pos, substep_idx],
                        actor=actor_chunks[metric_pos, substep_idx],
                        target_source=args.target_source,
                        precision=args.precision,
                    )
                    if args.table:
                        _print_joint_table(
                            pt_chunks[metric_pos, substep_idx],
                            gt_chunks[metric_pos, substep_idx],
                            actor_chunks[metric_pos, substep_idx],
                            args.precision,
                        )

                live_before = _get_live_qpos(env) if env is not None else None
                decision = _prompt(
                    execute=bool(args.execute),
                    chunk_idx=original_chunk,
                    substep_idx=substep_idx,
                    target=target,
                    recorded_state=recorded_state,
                    live_qpos=live_before,
                    max_live_state_diff=float(args.max_live_state_diff),
                    max_action_jump=float(args.max_action_jump),
                    allow_large_diff=bool(args.allow_large_diff),
                )
                if decision == "quit":
                    print("Quit requested.")
                    return 0
                if decision == "skip":
                    skipped_targets += 1
                    _write_log(
                        args.log_jsonl,
                        {
                            "time": time.time(),
                            "original_chunk": original_chunk,
                            "metric_pos": metric_pos,
                            "substep": substep_idx,
                            "target_source": args.target_source,
                            "decision": "skip",
                            "recorded_state": recorded_state,
                            "target": target,
                            "live_before": live_before,
                        },
                    )
                    continue

                live_after = None
                if args.execute:
                    live_after = _step_env_with_report(
                        env, target, float(args.sleep_after_step)
                    )
                executed_targets += 1
                _write_log(
                    args.log_jsonl,
                    {
                        "time": time.time(),
                        "original_chunk": original_chunk,
                        "metric_pos": metric_pos,
                        "substep": substep_idx,
                        "target_source": args.target_source,
                        "decision": "execute" if args.execute else "dry_run",
                        "recorded_state": recorded_state,
                        "pt_target": (
                            pt_chunks[metric_pos, substep_idx]
                            if substep_idx is not None
                            else None
                        ),
                        "gt_target": (
                            gt_chunks[metric_pos, substep_idx]
                            if substep_idx is not None
                            else None
                        ),
                        "actor_target": (
                            actor_chunks[metric_pos, substep_idx]
                            if substep_idx is not None
                            else None
                        ),
                        "target": target,
                        "live_before": live_before,
                        "live_after": live_after,
                    },
                )

            print(
                f"\nRecorded reward for original chunk {original_chunk}: "
                f"reward={float(reward):.4g}, done={int(done)}, term={int(term)}, trunc={int(trunc)}, "
                f"executed_targets={executed_targets}, skipped_targets={skipped_targets}"
            )
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        return 130
    finally:
        if env is not None and hasattr(env, "close"):
            env.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
