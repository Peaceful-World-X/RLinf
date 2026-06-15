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

"""Visualize one Piper offline .pt trajectory without actor/critic inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rlinf.utils.offline_td3_visualization import (  # noqa: E402
    SimpleUrdfKinematics,
    compute_chunk_mc_returns,
    save_json,
    tcp_trajectories_from_joint_targets,
)


def tensor_1d(trajectory: dict, key: str, default: float = 0.0) -> np.ndarray:
    value = trajectory.get(key, None)
    if value is None:
        length = int(trajectory["actions"].shape[0])
        return np.full((length,), default, dtype=np.float64)
    if torch.is_tensor(value):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=np.float64).reshape(-1)


def tensor_array(value) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def set_equal_3d(ax, arrays: list[np.ndarray]) -> None:
    points = np.concatenate(arrays, axis=0)
    centers = points.mean(axis=0)
    ranges = points.max(axis=0) - points.min(axis=0)
    radius = max(float(ranges.max()) / 2.0, 1e-3)
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def plot_pt_trajectory(
    out_path: Path,
    state_tcp: dict[str, np.ndarray],
    action_tcp_chunks: list[dict[str, np.ndarray]],
) -> None:
    fig = plt.figure(figsize=(11, 5), dpi=150)
    axes = [
        fig.add_subplot(1, 2, 1, projection="3d"),
        fig.add_subplot(1, 2, 2, projection="3d"),
    ]
    colors = plt.cm.viridis(np.linspace(0.0, 1.0, max(len(action_tcp_chunks), 1)))
    for ax, side in zip(axes, ["left", "right"]):
        state = state_tcp[side]
        ax.plot(
            state[:, 0],
            state[:, 1],
            state[:, 2],
            color="#222222",
            linewidth=2.2,
            label="state TCP",
        )
        for idx, action_tcp in enumerate(action_tcp_chunks):
            action = action_tcp[side]
            ax.plot(
                action[:, 0],
                action[:, 1],
                action[:, 2],
                color=colors[idx],
                linewidth=0.8,
                alpha=0.7,
                label="action chunks" if idx == 0 else None,
            )
        ax.scatter(
            state[0, 0], state[0, 1], state[0, 2], color="#2f8f46", s=28, label="start"
        )
        ax.scatter(
            state[-1, 0], state[-1, 1], state[-1, 2], color="#b3472f", s=28, label="end"
        )
        ax.set_title(f"{side} TCP")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
        all_action_points = [chunk[side] for chunk in action_tcp_chunks]
        set_equal_3d(ax, [state, *all_action_points])
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_consistency(
    out_path: Path, first_err: np.ndarray, last_err: np.ndarray
) -> None:
    steps = np.arange(len(first_err))
    fig, ax = plt.subplots(1, 1, figsize=(9, 3.5), dpi=150)
    ax.plot(
        steps,
        first_err,
        marker="o",
        markersize=3,
        label="|action[0] - curr_state| mean",
    )
    ax.plot(
        steps,
        last_err,
        marker="o",
        markersize=3,
        label="|action[-1] - next_state| mean",
    )
    ax.set_xlabel("trajectory chunk index")
    ax.set_ylabel("mean abs joint error")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_reward_return(
    out_path: Path, rewards: np.ndarray, returns: np.ndarray
) -> None:
    steps = np.arange(len(rewards))
    fig, ax1 = plt.subplots(1, 1, figsize=(9, 3.5), dpi=150)
    ax1.bar(steps, rewards, color="#a6c8ff", label="reward")
    ax1.set_xlabel("trajectory chunk index")
    ax1.set_ylabel("reward")
    ax2 = ax1.twinx()
    ax2.plot(steps, returns, color="#2f8f46", linewidth=2.0, label="MC return")
    ax2.set_ylabel("MC return")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--urdf", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--base-link", default="base_link")
    parser.add_argument("--tip-link", default="gripper")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--action-horizon", type=int, default=10)
    parser.add_argument(
        "--base-offsets",
        type=float,
        nargs=6,
        default=[0.0, 0.22, 0.0, 0.0, -0.22, 0.0],
    )
    args = parser.parse_args()

    trajectory_path = Path(args.trajectory)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    trajectory = torch.load(trajectory_path, map_location="cpu")
    actions = tensor_array(trajectory["actions"][:, 0]).reshape(
        -1, args.action_horizon, 14
    )
    curr_states = tensor_array(trajectory["curr_obs"]["states"][:, 0]).reshape(-1, 14)
    next_states = tensor_array(trajectory["next_obs"]["states"][:, 0]).reshape(-1, 14)
    rewards = tensor_1d(trajectory, "rewards")
    terminations = tensor_1d(trajectory, "terminations")
    if "dones" in trajectory:
        terminations = np.maximum(terminations, tensor_1d(trajectory, "dones"))
    returns = compute_chunk_mc_returns(
        rewards, terminations, args.gamma, args.action_horizon
    )

    first_err = np.abs(actions[:, 0] - curr_states).mean(axis=1)
    last_err = np.abs(actions[:, -1] - next_states).mean(axis=1)
    next_step_errors = np.abs(actions - next_states[:, None, :]).mean(axis=2)
    best_next_steps = next_step_errors.argmin(axis=1)
    best_next_err = next_step_errors.min(axis=1)

    kinematics = SimpleUrdfKinematics(args.urdf, base_link=args.base_link)
    base_offsets = np.asarray(args.base_offsets, dtype=np.float64).reshape(2, 3)
    state_joints = np.concatenate([curr_states, next_states[-1:]], axis=0)
    state_tcp = tcp_trajectories_from_joint_targets(
        state_joints,
        kinematics,
        base_offsets,
        tip_link=args.tip_link,
    )
    action_tcp_chunks = [
        tcp_trajectories_from_joint_targets(
            actions[idx],
            kinematics,
            base_offsets,
            tip_link=args.tip_link,
        )
        for idx in range(actions.shape[0])
    ]
    executed_prefix_tcp_chunks = [
        tcp_trajectories_from_joint_targets(
            actions[idx, : best_next_steps[idx] + 1],
            kinematics,
            base_offsets,
            tip_link=args.tip_link,
        )
        for idx in range(actions.shape[0])
    ]

    plot_pt_trajectory(out_dir / "raw_pt_tcp_3d.png", state_tcp, action_tcp_chunks)
    plot_pt_trajectory(
        out_dir / "raw_pt_tcp_3d_executed_prefix.png",
        state_tcp,
        executed_prefix_tcp_chunks,
    )
    plot_consistency(out_dir / "state_action_consistency.png", first_err, last_err)
    plot_reward_return(out_dir / "reward_mc_return.png", rewards, returns)

    summary = {
        "trajectory_file": str(trajectory_path),
        "length": int(actions.shape[0]),
        "action_shape": list(actions.shape),
        "curr_state_shape": list(curr_states.shape),
        "action_min": float(actions.min()),
        "action_max": float(actions.max()),
        "state_min": float(curr_states.min()),
        "state_max": float(curr_states.max()),
        "frac_action_outside_pm1": float(np.mean((actions < -1.0) | (actions > 1.0))),
        "first_action_vs_curr_mean_abs": float(first_err.mean()),
        "first_action_vs_curr_max_abs": float(
            np.abs(actions[:, 0] - curr_states).max()
        ),
        "last_action_vs_next_mean_abs": float(last_err.mean()),
        "last_action_vs_next_max_abs": float(
            np.abs(actions[:, -1] - next_states).max()
        ),
        "best_action_substep_vs_next_mean": float(best_next_steps.mean()),
        "best_action_substep_hist": {
            str(idx): int(np.sum(best_next_steps == idx))
            for idx in range(args.action_horizon)
        },
        "best_action_substep_vs_next_mean_abs": float(best_next_err.mean()),
        "positive_reward_indices": np.flatnonzero(rewards > 0).astype(int).tolist(),
        "done_indices": np.flatnonzero(terminations > 0).astype(int).tolist(),
        "mc_return_first": float(returns[0]) if len(returns) else 0.0,
        "mc_return_last": float(returns[-1]) if len(returns) else 0.0,
        "mc_discount_per_chunk": float(args.gamma**args.action_horizon),
        "outputs": [
            str(out_dir / "raw_pt_tcp_3d.png"),
            str(out_dir / "raw_pt_tcp_3d_executed_prefix.png"),
            str(out_dir / "state_action_consistency.png"),
            str(out_dir / "reward_mc_return.png"),
            str(out_dir / "summary.json"),
        ],
    }
    save_json(str(out_dir / "summary.json"), summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
