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

"""Step-confirmed replay for saved Piper `.pt` trajectories.

The script is intentionally conservative:

- dry-run by default; add `--execute` to command the real robot;
- requires Enter before every replayed target;
- uses absolute joint targets, even for norm-delta converted datasets;
- never resets the robot unless `--reset` is explicitly passed.
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

DELTA_MASK = np.array([True] * 6 + [False] + [True] * 6 + [False], dtype=bool)
DEFAULT_NORM_STATS_PATH = (
    "/home/focal/shared_disk/users/kwj/weight/openpi/pi05_cube_insert/norm_stats.json"
)


def _as_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().float().numpy()
    return np.asarray(value)


def _get_nested(payload: dict[str, Any], dotted_key: str) -> Any:
    current: Any = payload
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(f"Cannot find key {dotted_key!r}; missing {part!r}")
        current = current[part]
    return current


def _trajectory_1d(
    trajectory: dict[str, Any], key: str, default: float = 0.0
) -> np.ndarray:
    if key not in trajectory:
        length = int(trajectory["actions"].shape[0])
        return np.full((length,), default, dtype=np.float64)
    return _as_numpy(trajectory[key]).reshape(-1).astype(np.float64)


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


def _format_vec(vec: np.ndarray, precision: int = 4) -> str:
    return np.array2string(
        np.asarray(vec, dtype=np.float64),
        precision=precision,
        suppress_small=True,
        max_line_width=180,
    )


def _summarize_delta(name: str, delta: np.ndarray) -> str:
    delta = np.asarray(delta, dtype=np.float64).reshape(14)
    left = float(np.max(np.abs(delta[:7])))
    right = float(np.max(np.abs(delta[7:14])))
    all_max = float(np.max(np.abs(delta)))
    l2 = float(np.linalg.norm(delta))
    return f"{name}: max_abs={all_max:.5f}, l2={l2:.5f}, left_max={left:.5f}, right_max={right:.5f}"


def _load_norm_stats(
    path: str | None, std_floor: float
) -> tuple[np.ndarray, np.ndarray] | None:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    stats = payload.get("norm_stats", payload)["actions"]
    mean = np.asarray(stats["mean"][:14], dtype=np.float64)
    std = np.asarray(stats["std"][:14], dtype=np.float64)
    std = np.where(np.abs(std) < float(std_floor), 1.0, std)
    return mean, std


def _normalized_delta_to_absolute(
    actions: np.ndarray,
    states: np.ndarray,
    norm_stats: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    mean, std = norm_stats
    shape = actions.shape
    actions = actions.reshape(shape[0], shape[1], -1, 14).astype(np.float64)
    states = states.reshape(shape[0], shape[1], 14).astype(np.float64)
    delta = actions * (std.reshape(1, 1, 1, 14) + 1e-6) + mean.reshape(1, 1, 1, 14)
    absolute = delta.copy()
    absolute[..., DELTA_MASK] += states[:, :, None, :][..., DELTA_MASK]
    return absolute.reshape(shape)


def _choose_absolute_actions(
    trajectory: dict[str, Any],
    action_key: str | None,
    action_space: str,
    norm_stats: tuple[np.ndarray, np.ndarray] | None,
    batch_index: int,
) -> tuple[np.ndarray, str]:
    curr_states = _as_numpy(trajectory["curr_obs"]["states"]).astype(np.float64)

    if action_key:
        raw = _as_numpy(_get_nested(trajectory, action_key)).astype(np.float64)
        source_name = action_key
    else:
        forward_inputs = trajectory.get("forward_inputs", {})
        if isinstance(forward_inputs, dict) and "env_action_absolute" in forward_inputs:
            raw = _as_numpy(forward_inputs["env_action_absolute"]).astype(np.float64)
            source_name = "forward_inputs.env_action_absolute"
            action_space = "absolute"
        else:
            raw = _as_numpy(trajectory["actions"]).astype(np.float64)
            source_name = "actions"

    if raw.ndim == 2:
        raw = raw[:, None, :]
    if curr_states.ndim == 2:
        curr_states = curr_states[:, None, :]
    if raw.shape[1] <= batch_index:
        raise IndexError(
            f"batch_index={batch_index} but action batch dim is {raw.shape[1]}"
        )

    if action_space == "auto":
        # Converted datasets keep absolute env targets under forward_inputs.env_action_absolute.
        # If the selected source is raw actions and looks normalized, require explicit stats.
        looks_normalized = float(np.nanmax(np.abs(raw))) <= 2.0
        action_space = (
            "normalized_delta"
            if looks_normalized and norm_stats is not None
            else "absolute"
        )

    if action_space == "normalized_delta":
        if norm_stats is None:
            raise ValueError(
                "--norm-stats is required when --action-space normalized_delta"
            )
        raw = _normalized_delta_to_absolute(raw, curr_states, norm_stats)
        source_name += " -> normalized_delta_to_absolute"
    elif action_space != "absolute":
        raise ValueError(f"Unsupported action_space={action_space!r}")

    absolute = raw[:, batch_index]
    horizon = absolute.shape[-1] // 14
    if absolute.shape[-1] % 14 != 0:
        raise ValueError(f"Action last dim must be H*14, got shape={absolute.shape}")
    return absolute.reshape(absolute.shape[0], horizon, 14), source_name


def _source_label_for_chunk(
    *,
    chunk_idx: int,
    main_source_name: str,
    hybrid_action_source: str | None,
    hybrid_switch_chunk: int | None,
    hybrid_switch_from_end: int | None,
    length: int,
) -> str:
    if hybrid_action_source is None:
        return main_source_name
    switch = (
        length - int(hybrid_switch_from_end)
        if hybrid_switch_from_end is not None
        else int(hybrid_switch_chunk)
    )
    return hybrid_action_source if chunk_idx >= switch else main_source_name


def _load_trajectory(args: argparse.Namespace) -> dict[str, Any]:
    path = Path(args.trajectory).expanduser()
    if not path.is_file():
        raise FileNotFoundError(path)
    trajectory = torch.load(path, map_location="cpu")
    if "curr_obs" not in trajectory or "states" not in trajectory["curr_obs"]:
        raise KeyError("Trajectory must contain curr_obs.states")
    if "actions" not in trajectory:
        raise KeyError("Trajectory must contain actions")
    return trajectory


def _make_env(args: argparse.Namespace):
    os.environ.setdefault("RLINF_SKIP_ROS_CLEANUP", "1")
    from rlinf.envs.realworld.piper.task.peg_insertion_env import PiperPegInsertionEnv

    override_cfg = {
        "is_dummy": bool(args.dummy),
        "joint_action_mode": "absolute",
        "delta_action_scale": float(args.delta_action_scale),
        "enable_human_intervention": False,
        "wait_teleop_release_before_reset": False,
        "wait_enter_after_reset": False,
        "step_frequency": float(args.step_frequency),
        "joint_speed_pct": int(args.joint_speed_pct),
        "camera_names": [],
        "img_topics": [],
        "obs_img_resolution": [16, 16],
        "gripper_action_threshold": None,
        "sliding_window_action_buffer": False,
        "smooth_action_chunk": False,
        "max_num_steps": int(args.max_env_steps),
    }
    return PiperPegInsertionEnv(override_cfg=override_cfg)


def _get_live_qpos(env: Any) -> np.ndarray:
    if getattr(env, "config", None) is not None and env.config.is_dummy:
        return np.zeros(14, dtype=np.float64)
    if getattr(env, "_controller", None) is None:
        return np.zeros(14, dtype=np.float64)
    return np.asarray(env._controller.get_qpos(), dtype=np.float64).reshape(14)


def _prompt(
    *,
    execute: bool,
    chunk_idx: int,
    substep_idx: int | None,
    target: np.ndarray,
    recorded_state: np.ndarray,
    live_qpos: np.ndarray | None,
    max_live_state_diff: float,
    max_action_jump: float,
    allow_large_diff: bool,
) -> str:
    print("")
    if substep_idx is None:
        print(f"[state step {chunk_idx}] target recorded state")
    else:
        print(f"[chunk {chunk_idx}, substep {substep_idx}] target action")
    print("recorded_state:", _format_vec(recorded_state))
    print("target        :", _format_vec(target))
    print(_summarize_delta("target-recorded_state", target - recorded_state))

    needs_force = False
    if live_qpos is not None:
        print("live_qpos     :", _format_vec(live_qpos))
        state_delta = live_qpos - recorded_state
        action_delta = target - live_qpos
        print(_summarize_delta("live-recorded_state", state_delta))
        print(_summarize_delta("target-live", action_delta))
        if np.max(np.abs(state_delta)) > max_live_state_diff:
            print(
                f"WARNING: live state differs from recorded state by more than "
                f"{max_live_state_diff:.3f} rad/gripper-unit."
            )
            needs_force = True
        if np.max(np.abs(action_delta)) > max_action_jump:
            print(
                f"WARNING: target jump from live state is more than "
                f"{max_action_jump:.3f} rad/gripper-unit."
            )
            needs_force = True

    if not execute:
        prompt = "DRY-RUN: Enter=next, s=skip, p=print full target, q=quit > "
    elif needs_force and not allow_large_diff:
        prompt = "Type FORCE to execute this large move, s=skip, p=print full target, q=quit > "
    else:
        prompt = "EXECUTE: Enter=send this step, s=skip, p=print full target, q=quit > "
    response = input(prompt).strip()
    if execute and needs_force and not allow_large_diff and response != "FORCE":
        if response in {"q", "quit"}:
            return "quit"
        if response == "p":
            print("target full:", _format_vec(target, precision=7))
            return _prompt(
                execute=execute,
                chunk_idx=chunk_idx,
                substep_idx=substep_idx,
                target=target,
                recorded_state=recorded_state,
                live_qpos=live_qpos,
                max_live_state_diff=max_live_state_diff,
                max_action_jump=max_action_jump,
                allow_large_diff=allow_large_diff,
            )
        print("Skipped because FORCE was not typed.")
        return "skip"
    if response in {"q", "quit"}:
        return "quit"
    if response == "s":
        return "skip"
    if response == "p":
        print("target full:", _format_vec(target, precision=7))
        return _prompt(
            execute=execute,
            chunk_idx=chunk_idx,
            substep_idx=substep_idx,
            target=target,
            recorded_state=recorded_state,
            live_qpos=live_qpos,
            max_live_state_diff=max_live_state_diff,
            max_action_jump=max_action_jump,
            allow_large_diff=allow_large_diff,
        )
    return "execute"


def _write_log(log_path: str | None, record: dict[str, Any]) -> None:
    if not log_path:
        return

    def convert(value: Any):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        return value

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(
            json.dumps({k: convert(v) for k, v in record.items()}, ensure_ascii=False)
            + "\n"
        )


def _step_env_with_report(
    env: Any, target: np.ndarray, sleep_after_step: float
) -> np.ndarray:
    obs, reward, terminated, truncated, _info = env.step(target)
    live_after = np.asarray(obs["state"]["qpos"], dtype=np.float64).reshape(14)
    print("live_after   :", _format_vec(live_after))
    print(_summarize_delta("live_after-target", live_after - target))
    print(f"env reward={reward}, terminated={terminated}, truncated={truncated}")
    if sleep_after_step > 0:
        time.sleep(float(sleep_after_step))
    return live_after


def _go_to_recorded_start(
    *,
    env: Any,
    target_start: np.ndarray,
    chunk_idx: int,
    args: argparse.Namespace,
) -> None:
    live = _get_live_qpos(env)
    max_delta = float(np.max(np.abs(target_start - live)))
    step_limit = max(float(args.max_action_jump), 1e-6)
    num_steps = max(1, int(np.ceil(max_delta / step_limit)))
    num_steps = max(num_steps, int(args.go_to_start_min_steps))
    print("\n" + "=" * 100)
    print(
        f"Going to recorded start for chunk {chunk_idx}: "
        f"max_abs_delta={max_delta:.5f}, interpolation_steps={num_steps}"
    )
    print("recorded start:", _format_vec(target_start))

    for align_step in range(num_steps):
        live_now = _get_live_qpos(env)
        remaining_steps = num_steps - align_step
        target = live_now + (target_start - live_now) / float(remaining_steps)
        decision = _prompt(
            execute=True,
            chunk_idx=chunk_idx,
            substep_idx=None,
            target=target,
            # This alignment step is relative to the current live pose; print the
            # final recorded start separately above.
            recorded_state=live_now,
            live_qpos=live_now,
            max_live_state_diff=float(args.max_live_state_diff),
            max_action_jump=float(args.max_action_jump),
            allow_large_diff=bool(args.allow_large_diff),
        )
        if decision == "quit":
            raise KeyboardInterrupt("Quit requested during go-to-start.")
        if decision == "skip":
            print("Skipped this go-to-start interpolation step.")
            continue
        live_after = _step_env_with_report(env, target, float(args.sleep_after_step))
        _write_log(
            args.log_jsonl,
            {
                "time": time.time(),
                "chunk": chunk_idx,
                "substep": None,
                "decision": "go_to_start",
                "recorded_state": target_start,
                "target": target,
                "live_before": live_now,
                "live_after": live_after,
            },
        )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Safely replay one saved Piper .pt trajectory with per-step confirmation."
    )
    parser.add_argument("--trajectory", required=True, help="Path to trajectory_*.pt")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually command the real robot. Omit for dry-run inspection.",
    )
    parser.add_argument(
        "--dummy", action="store_true", help="Use dummy env instead of hardware."
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Call env.reset() before replay. Default: do not move robot to reset pose.",
    )
    parser.add_argument(
        "--target-source",
        choices=["action", "state", "ref_action", "train_action"],
        default="action",
        help=(
            "Replay action chunks or recorded curr_obs.states. "
            "ref_action replays curr_obs.ref_action as normalized-delta; "
            "train_action replays top-level actions as normalized-delta."
        ),
    )
    parser.add_argument(
        "--action-key",
        default=None,
        help=(
            "Nested action key. Default auto-prefers forward_inputs.env_action_absolute, "
            "else actions. Example: forward_inputs.model_action_absolute"
        ),
    )
    parser.add_argument(
        "--hybrid-action-key",
        default=None,
        help=(
            "Optional second nested action key used after the hybrid switch point. "
            "Example: forward_inputs.actor_replay_env_action_absolute."
        ),
    )
    parser.add_argument(
        "--hybrid-action-space",
        choices=["auto", "absolute", "normalized_delta"],
        default="auto",
        help="How to interpret --hybrid-action-key.",
    )
    parser.add_argument(
        "--hybrid-switch-chunk",
        type=int,
        default=None,
        help="When --hybrid-action-key is set, start using it at this chunk index.",
    )
    parser.add_argument(
        "--hybrid-switch-from-end",
        type=int,
        default=None,
        help=(
            "When --hybrid-action-key is set, start using it this many chunks from "
            "the end. Example: 20 means first length-20 chunks use the main source, "
            "the final 20 chunks use --hybrid-action-key."
        ),
    )
    parser.add_argument(
        "--action-space",
        choices=["auto", "absolute", "normalized_delta"],
        default="auto",
        help="How to interpret the selected action tensor.",
    )
    parser.add_argument(
        "--norm-stats",
        default=None,
        help="OpenPI norm_stats.json for normalized_delta.",
    )
    parser.add_argument("--norm-std-floor", type=float, default=1.0)
    parser.add_argument("--batch-index", type=int, default=0)
    parser.add_argument(
        "--chunks",
        default="0",
        help="Chunk indices to replay, e.g. 0, 10-15, all. Default: 0.",
    )
    parser.add_argument(
        "--substeps",
        default="all",
        help="Substeps inside each action chunk, e.g. 0, all, 0-9. Ignored for --target-source state.",
    )
    parser.add_argument("--max-env-steps", type=int, default=10000)
    parser.add_argument("--step-frequency", type=float, default=10.0)
    parser.add_argument("--joint-speed-pct", type=int, default=20)
    parser.add_argument("--delta-action-scale", type=float, default=0.01)
    parser.add_argument(
        "--max-live-state-diff",
        type=float,
        default=0.25,
        help="Require FORCE if live qpos differs from recorded state by more than this.",
    )
    parser.add_argument(
        "--max-action-jump",
        type=float,
        default=0.35,
        help="Require FORCE if target differs from live qpos by more than this.",
    )
    parser.add_argument(
        "--allow-large-diff",
        action="store_true",
        help="Do not require FORCE for large live/target differences.",
    )
    parser.add_argument(
        "--go-to-start",
        action="store_true",
        help=(
            "Before replay, move from current live qpos to the first selected recorded "
            "state using stepwise confirmed interpolation."
        ),
    )
    parser.add_argument(
        "--go-to-start-min-steps",
        type=int,
        default=1,
        help="Minimum number of confirmed interpolation steps used by --go-to-start.",
    )
    parser.add_argument(
        "--start-at-nearest-state",
        action="store_true",
        help=(
            "After creating the real env, find the recorded state nearest to current "
            "live qpos and start replay from there."
        ),
    )
    parser.add_argument("--sleep-after-step", type=float, default=0.0)
    parser.add_argument("--log-jsonl", default=None, help="Optional replay log path.")
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    np.set_printoptions(precision=4, suppress=True, linewidth=180)

    if args.hybrid_action_key:
        if (args.hybrid_switch_chunk is None) == (args.hybrid_switch_from_end is None):
            raise ValueError(
                "--hybrid-action-key requires exactly one of "
                "--hybrid-switch-chunk or --hybrid-switch-from-end."
            )
    elif (
        args.hybrid_switch_chunk is not None or args.hybrid_switch_from_end is not None
    ):
        raise ValueError(
            "--hybrid-switch-chunk/--hybrid-switch-from-end require --hybrid-action-key."
        )

    trajectory = _load_trajectory(args)
    curr_states_all = _as_numpy(trajectory["curr_obs"]["states"]).astype(np.float64)
    if curr_states_all.ndim == 2:
        curr_states_all = curr_states_all[:, None, :]
    if curr_states_all.shape[1] <= args.batch_index:
        raise IndexError(
            f"batch_index={args.batch_index} but curr_obs.states batch dim is {curr_states_all.shape[1]}"
        )
    curr_states = curr_states_all[:, args.batch_index, :14]
    length = int(curr_states.shape[0])
    rewards = _trajectory_1d(trajectory, "rewards")
    dones = _trajectory_1d(trajectory, "dones")
    terminations = _trajectory_1d(trajectory, "terminations")
    truncations = _trajectory_1d(trajectory, "truncations")

    if args.target_source == "ref_action":
        if args.action_key is not None and args.action_key not in {
            "curr_obs.ref_action",
            "forward_inputs.ref_action",
        }:
            raise ValueError(
                "--target-source ref_action uses curr_obs.ref_action or forward_inputs.ref_action; "
                "do not pass a different --action-key."
            )
        curr_obs = trajectory.get("curr_obs", {})
        forward_inputs = trajectory.get("forward_inputs", {})
        if isinstance(curr_obs, dict) and "ref_action" in curr_obs:
            args.action_key = "curr_obs.ref_action"
        elif isinstance(forward_inputs, dict) and "ref_action" in forward_inputs:
            args.action_key = "forward_inputs.ref_action"
        else:
            raise KeyError(
                "--target-source ref_action requires curr_obs.ref_action or forward_inputs.ref_action"
            )
        args.action_space = "normalized_delta"
    elif args.target_source == "train_action":
        if args.action_key is not None and args.action_key != "actions":
            raise ValueError(
                "--target-source train_action uses top-level actions; do not pass a different --action-key."
            )
        args.action_key = "actions"
        args.action_space = "normalized_delta"

    if args.action_space == "normalized_delta" and not args.norm_stats:
        if Path(DEFAULT_NORM_STATS_PATH).is_file():
            args.norm_stats = DEFAULT_NORM_STATS_PATH
        else:
            raise ValueError(
                "--norm-stats is required for normalized_delta replay; "
                f"default not found: {DEFAULT_NORM_STATS_PATH}"
            )

    norm_stats = _load_norm_stats(args.norm_stats, args.norm_std_floor)
    action_chunks, action_source = _choose_absolute_actions(
        trajectory,
        args.action_key,
        args.action_space,
        norm_stats,
        args.batch_index,
    )
    hybrid_action_chunks = None
    hybrid_action_source = None
    if args.hybrid_action_key:
        hybrid_norm_stats = norm_stats
        if args.hybrid_action_space == "normalized_delta" and hybrid_norm_stats is None:
            raise ValueError(
                "--norm-stats is required when --hybrid-action-space normalized_delta"
            )
        hybrid_action_chunks, hybrid_action_source = _choose_absolute_actions(
            trajectory,
            args.hybrid_action_key,
            args.hybrid_action_space,
            hybrid_norm_stats,
            args.batch_index,
        )
        if hybrid_action_chunks.shape != action_chunks.shape:
            raise ValueError(
                "Hybrid action chunks must match main action shape: "
                f"{hybrid_action_chunks.shape} vs {action_chunks.shape}"
            )
    horizon = int(action_chunks.shape[1])
    chunk_indices = _parse_indices(args.chunks, length)
    substep_indices = _parse_indices(args.substeps, horizon, default_all=True)

    print("Loaded trajectory:", Path(args.trajectory))
    print(f"length={length}, action_horizon={horizon}, action_source={action_source}")
    if hybrid_action_chunks is not None:
        if args.hybrid_switch_from_end is not None:
            switch_chunk = length - int(args.hybrid_switch_from_end)
            switch_desc = (
                f"chunk >= {switch_chunk} "
                f"(last {int(args.hybrid_switch_from_end)} chunks)"
            )
        else:
            switch_chunk = int(args.hybrid_switch_chunk)
            switch_desc = f"chunk >= {switch_chunk}"
        print(
            "hybrid action replay: "
            f"main source before {switch_desc}; "
            f"{hybrid_action_source} from {switch_desc}"
        )
    print("selected chunks:", chunk_indices)
    if args.target_source != "state":
        print("selected substeps:", substep_indices)
    if args.action_space == "normalized_delta":
        print("norm_stats:", args.norm_stats)
    print(
        f"reward stats: min={float(np.min(rewards)):.4g}, max={float(np.max(rewards)):.4g}, "
        f"sum={float(np.sum(rewards)):.4g}"
    )

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
            dists = np.linalg.norm(curr_states - live.reshape(1, 14), axis=1)
            nearest = int(np.argmin(dists))
            print(
                f"Nearest recorded state to live qpos: chunk={nearest}, "
                f"l2={float(dists[nearest]):.5f}, "
                f"max_abs={float(np.max(np.abs(curr_states[nearest] - live))):.5f}"
            )
            chunk_indices = [idx for idx in chunk_indices if idx >= nearest]
            if not chunk_indices:
                chunk_indices = [nearest]
            print("updated selected chunks:", chunk_indices)

    if args.go_to_start:
        if env is None:
            print("--go-to-start ignored in dry-run mode.")
        elif not chunk_indices:
            raise ValueError("No chunk selected; cannot --go-to-start.")
        else:
            _go_to_recorded_start(
                env=env,
                target_start=curr_states[chunk_indices[0]],
                chunk_idx=chunk_indices[0],
                args=args,
            )

    try:
        for chunk_idx in chunk_indices:
            recorded_state = curr_states[chunk_idx]
            selected_action_chunks = action_chunks
            selected_source = action_source
            if hybrid_action_chunks is not None:
                selected_source = _source_label_for_chunk(
                    chunk_idx=chunk_idx,
                    main_source_name=action_source,
                    hybrid_action_source=hybrid_action_source,
                    hybrid_switch_chunk=args.hybrid_switch_chunk,
                    hybrid_switch_from_end=args.hybrid_switch_from_end,
                    length=length,
                )
                if selected_source == hybrid_action_source:
                    selected_action_chunks = hybrid_action_chunks
            print("\n" + "=" * 100)
            print(
                f"chunk={chunk_idx}/{length - 1}, reward={float(rewards[chunk_idx]):.4g}, "
                f"done={int(dones[chunk_idx])}, term={int(terminations[chunk_idx])}, "
                f"trunc={int(truncations[chunk_idx])}, selected_source={selected_source}"
            )
            print("recorded curr_state:", _format_vec(recorded_state))
            print(
                "action chunk first :",
                _format_vec(selected_action_chunks[chunk_idx, 0]),
            )
            print(
                "action chunk last  :",
                _format_vec(selected_action_chunks[chunk_idx, -1]),
            )
            if hybrid_action_chunks is not None:
                print("main first/last   :", _format_vec(action_chunks[chunk_idx, 0]))
                print("                  :", _format_vec(action_chunks[chunk_idx, -1]))
                print(
                    "hybrid first/last :",
                    _format_vec(hybrid_action_chunks[chunk_idx, 0]),
                )
                print(
                    "                  :",
                    _format_vec(hybrid_action_chunks[chunk_idx, -1]),
                )
                diff = hybrid_action_chunks[chunk_idx] - action_chunks[chunk_idx]
                print(
                    "hybrid-main chunk: "
                    f"mse={float(np.mean(diff**2)):.9g}, "
                    f"mae={float(np.mean(np.abs(diff))):.9g}, "
                    f"max_abs={float(np.max(np.abs(diff))):.9g}"
                )
            print(
                _summarize_delta(
                    "first-recorded_state",
                    selected_action_chunks[chunk_idx, 0] - recorded_state,
                )
            )
            print(
                _summarize_delta(
                    "last-recorded_state",
                    selected_action_chunks[chunk_idx, -1] - recorded_state,
                )
            )

            if args.target_source == "state":
                steps = [(None, recorded_state)]
            else:
                steps = [
                    (substep, selected_action_chunks[chunk_idx, substep])
                    for substep in substep_indices
                ]

            executed_targets = 0
            skipped_targets = 0
            for substep_idx, target in steps:
                live_before = _get_live_qpos(env) if env is not None else None
                decision = _prompt(
                    execute=bool(args.execute),
                    chunk_idx=chunk_idx,
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
                            "chunk": chunk_idx,
                            "substep": substep_idx,
                            "selected_source": selected_source,
                            "decision": "skip",
                            "recorded_state": recorded_state,
                            "target": target,
                            "live_before": live_before,
                        },
                    )
                    continue

                executed_targets += 1
                live_after = None
                reward = terminated = truncated = None
                if args.execute and env is not None:
                    obs, reward, terminated, truncated, _info = env.step(target)
                    live_after = np.asarray(
                        obs["state"]["qpos"], dtype=np.float64
                    ).reshape(14)
                    print("live_after   :", _format_vec(live_after))
                    print(_summarize_delta("live_after-target", live_after - target))
                    print(
                        f"env reward={reward}, terminated={terminated}, truncated={truncated}"
                    )
                    if args.sleep_after_step > 0:
                        time.sleep(float(args.sleep_after_step))

                _write_log(
                    args.log_jsonl,
                    {
                        "time": time.time(),
                        "chunk": chunk_idx,
                        "substep": substep_idx,
                        "selected_source": selected_source,
                        "decision": "execute" if args.execute else "dry_run_next",
                        "recorded_state": recorded_state,
                        "target": target,
                        "live_before": live_before,
                        "live_after": live_after,
                        "reward": reward,
                        "terminated": terminated,
                        "truncated": truncated,
                    },
                )
            print(
                f"\nRecorded reward for chunk {chunk_idx}: "
                f"reward={float(rewards[chunk_idx]):.4g}, "
                f"done={int(dones[chunk_idx])}, "
                f"term={int(terminations[chunk_idx])}, "
                f"trunc={int(truncations[chunk_idx])}, "
                f"executed_targets={executed_targets}, skipped_targets={skipped_targets}"
            )
    finally:
        if env is not None and hasattr(env, "close"):
            env.close()

    print("Replay finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
