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

"""Plot actual RLT scalars from one RLinf run's TensorBoard directory."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_scalars(root: Path) -> dict[str, dict[int, tuple[float, float]]]:
    """Merge event files, retaining the latest wall-time value at each step."""
    values = {}
    for directory in sorted({p.parent for p in root.rglob("events.out.tfevents.*")}):
        events = EventAccumulator(str(directory), size_guidance={"scalars": 0})
        events.Reload()
        for tag in events.Tags()["scalars"]:
            if tag not in {
                "train/rlt_loss",
                "train/vla_loss",
                "train/loss",
                "train/lr",
            }:
                continue
            target = values.setdefault(tag, {})
            for event in events.Scalars(tag):
                if not np.isfinite(event.value):
                    raise ValueError(f"Non-finite {tag} at logged step {event.step}")
                if event.wall_time >= target.get(event.step, (-np.inf, 0))[0]:
                    target[event.step] = (event.wall_time, event.value)
    if not values.get("train/rlt_loss"):
        raise ValueError(f"No train/rlt_loss TensorBoard scalars under {root}")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--window", type=int, default=100, help="Logged points per mean"
    )
    args = parser.parse_args()
    if args.window < 1:
        parser.error("--window must be positive")
    values = read_scalars(args.log_dir)
    tags = sorted(values)
    steps = sorted(values["train/rlt_loss"])
    # SFTRunner logs zero-based _step after completing the optimizer update.
    x = np.asarray(steps) + 1
    y = np.asarray([values["train/rlt_loss"][s][1] for s in steps])
    if np.any(y <= 0):
        raise ValueError("Log-scale reconstruction loss must be positive")
    window = min(args.window, len(y))
    means = np.convolve(y, np.ones(window) / window, mode="valid")
    mean_steps = x[window - 1 :]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "rlt_loss.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["optimizer_step", *tags])
        for step in sorted(set().union(*(set(v) for v in values.values()))):
            writer.writerow(
                [step + 1, *[values[t].get(step, (0, ""))[1] for t in tags]]
            )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")
    fig.suptitle("RLinf | RL Token Reconstruction Loss", weight="bold")
    for ax in axes:
        ax.plot(x, y, color="#7fa9c5", alpha=0.5, linewidth=0.7, label="Logged loss")
        ax.plot(
            mean_steps,
            means,
            color="#007f89",
            linewidth=2,
            label=f"{window}-point mean",
        )
        ax.set_xlabel("Completed optimizer step")
        ax.set_ylabel("RLT reconstruction MSE")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
    axes[0].set_yscale("log")
    axes[0].set_title("Full training (log scale)")
    start = max(0, len(x) // 2 - 1)
    axes[1].set_xlim(float(x[start]) - 0.5, float(x[-1]) + 0.5)
    tail = y[start:]
    margin = max(float(np.ptp(tail)) * 0.1, float(tail.mean()) * 0.01)
    axes[1].set_ylim(float(tail.min()) - margin, float(tail.max()) + margin)
    axes[1].set_title("Second half (linear scale)")
    for suffix in ["png", "pdf"]:
        fig.savefig(args.output_dir / f"rlt_loss.{suffix}", dpi=180)
    plt.close(fig)
    summary = {
        "source": str(args.log_dir.resolve()),
        "logged_points": len(y),
        "last_completed_step": int(x[-1]),
        "first_logged_rlt_loss": float(y[0]),
        "last_logged_rlt_loss": float(y[-1]),
        "last_window_mean": float(means[-1]),
        "window_points": window,
        "note": "Training loss only; this does not establish held-out or robot performance.",
    }
    if len(y) >= 2 * window:
        previous = float(y[-2 * window : -window].mean())
        summary["previous_window_mean"] = previous
        summary["last_window_relative_improvement"] = (
            previous - float(means[-1])
        ) / previous
    (args.output_dir / "rlt_loss.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
