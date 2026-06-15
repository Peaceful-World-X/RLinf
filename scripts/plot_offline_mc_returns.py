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

"""Plot Monte Carlo returns for offline Piper .pt trajectories."""

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
    compute_chunk_mc_returns,
    save_json,
)


def trajectory_id(path: Path) -> int:
    stem = path.stem
    parts = stem.split("_")
    for part in parts:
        if part.isdigit():
            return int(part)
    return -1


def tensor_1d(trajectory: dict, key: str, default: float = 0.0) -> np.ndarray:
    value = trajectory.get(key, None)
    if value is None:
        return np.full(
            (int(trajectory["actions"].shape[0]),), default, dtype=np.float64
        )
    if torch.is_tensor(value):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=np.float64).reshape(-1)


def interp_curve(values: np.ndarray, points: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(values) == 1:
        return np.full((points,), values[0], dtype=np.float64)
    x_old = np.linspace(0.0, 1.0, len(values))
    x_new = np.linspace(0.0, 1.0, points)
    return np.interp(x_new, x_old, values)


def plot_spaghetti(out_path: Path, curves: list[np.ndarray], title: str) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(9, 4.5), dpi=150)
    for curve in curves:
        ax.plot(
            np.arange(len(curve)), curve, color="#2a6fbb", alpha=0.12, linewidth=1.0
        )
    if curves:
        arr = np.stack(curves, axis=0)
        mean = arr.mean(axis=0)
        q25, q75 = np.quantile(arr, [0.25, 0.75], axis=0)
        ax.plot(
            np.arange(len(mean)), mean, color="#222222", linewidth=2.2, label="mean"
        )
        ax.fill_between(
            np.arange(len(mean)), q25, q75, color="#222222", alpha=0.15, label="25-75%"
        )
    ax.set_title(title)
    ax.set_xlabel("normalized trajectory progress")
    ax.set_ylabel("MC return")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_examples(out_path: Path, items: list[dict], title: str, count: int) -> None:
    selected = items[:count]
    fig, ax = plt.subplots(1, 1, figsize=(9, 4.5), dpi=150)
    for item in selected:
        returns = item["returns"]
        ax.plot(
            np.arange(len(returns)), returns, linewidth=1.6, label=f"traj {item['id']}"
        )
    ax.set_title(title)
    ax.set_xlabel("trajectory chunk index")
    ax.set_ylabel("MC return")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    if selected:
        ax.legend(loc="best", ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--action-horizon", type=int, default=10)
    parser.add_argument("--interp-points", type=int, default=100)
    parser.add_argument("--example-count", type=int, default=8)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    per_traj_dir = out_dir / "per_trajectory"
    per_traj_dir.mkdir(parents=True, exist_ok=True)

    items: list[dict] = []
    for path in sorted(data_dir.glob("trajectory_*.pt"), key=trajectory_id):
        trajectory = torch.load(path, map_location="cpu")
        rewards = tensor_1d(trajectory, "rewards")
        terminations = tensor_1d(trajectory, "terminations")
        if "dones" in trajectory:
            terminations = np.maximum(terminations, tensor_1d(trajectory, "dones"))
        if "truncations" in trajectory:
            terminations = np.maximum(
                terminations, tensor_1d(trajectory, "truncations")
            )
        returns = compute_chunk_mc_returns(
            rewards,
            terminations,
            args.gamma,
            args.action_horizon,
        )
        is_success = bool(np.max(rewards) > 0.0)
        item = {
            "id": trajectory_id(path),
            "file": str(path),
            "length": int(len(returns)),
            "success": is_success,
            "max_reward": float(np.max(rewards)) if len(rewards) else 0.0,
            "first_return": float(returns[0]) if len(returns) else 0.0,
            "last_return": float(returns[-1]) if len(returns) else 0.0,
            "returns": returns,
            "interp_returns": interp_curve(returns, args.interp_points),
        }
        items.append(item)

        fig, ax = plt.subplots(1, 1, figsize=(7, 3.2), dpi=140)
        ax.plot(np.arange(len(returns)), returns, color="#2f8f46", linewidth=2.0)
        ax.bar(np.arange(len(rewards)), rewards, color="#a6c8ff", alpha=0.65)
        ax.set_title(
            f"trajectory {item['id']} ({'success' if is_success else 'failure'})"
        )
        ax.set_xlabel("trajectory chunk index")
        ax.set_ylabel("MC return / reward")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(per_traj_dir / f"trajectory_{item['id']:03d}_mc_return.png")
        plt.close(fig)

    success_items = [item for item in items if item["success"]]
    failure_items = [item for item in items if not item["success"]]
    plot_spaghetti(
        out_dir / "success_mc_returns_normalized.png",
        [item["interp_returns"] for item in success_items],
        "Success trajectories: data-action MC return",
    )
    plot_spaghetti(
        out_dir / "failure_mc_returns_normalized.png",
        [item["interp_returns"] for item in failure_items],
        "Failure trajectories: data-action MC return",
    )
    plot_examples(
        out_dir / "success_examples_mc_return.png",
        success_items,
        "Example success trajectories: data-action MC return",
        args.example_count,
    )
    plot_examples(
        out_dir / "failure_examples_mc_return.png",
        failure_items,
        "Example failure trajectories: data-action MC return",
        args.example_count,
    )

    summary = {
        "data_dir": str(data_dir),
        "num_trajectories": len(items),
        "num_success": len(success_items),
        "num_failure": len(failure_items),
        "gamma": float(args.gamma),
        "action_horizon": int(args.action_horizon),
        "discount_per_chunk": float(args.gamma**args.action_horizon),
        "success_first_return_mean": float(
            np.mean([item["first_return"] for item in success_items])
        )
        if success_items
        else 0.0,
        "success_length_mean": float(
            np.mean([item["length"] for item in success_items])
        )
        if success_items
        else 0.0,
        "failure_first_return_mean": float(
            np.mean([item["first_return"] for item in failure_items])
        )
        if failure_items
        else 0.0,
        "outputs": [
            str(out_dir / "success_mc_returns_normalized.png"),
            str(out_dir / "failure_mc_returns_normalized.png"),
            str(out_dir / "success_examples_mc_return.png"),
            str(out_dir / "failure_examples_mc_return.png"),
            str(per_traj_dir),
            str(out_dir / "summary.json"),
        ],
    }
    save_json(str(out_dir / "summary.json"), summary)
    serializable = []
    for item in items:
        compact = dict(item)
        compact["returns"] = compact["returns"].tolist()
        compact["interp_returns"] = compact["interp_returns"].tolist()
        serializable.append(compact)
    save_json(str(out_dir / "trajectories.json"), {"trajectories": serializable})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
