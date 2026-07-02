#!/usr/bin/env python3
# Copyright 2026 The GIGA Authors.
#
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml


def _find_repo_root(start: Path) -> Path:
    for path in [start, *start.parents]:
        if (path / "rlinf").is_dir() and (path / "examples").is_dir():
            return path
    raise RuntimeError(f"Could not find RLinf repo root from {start}")


REPO_ROOT = _find_repo_root(Path(__file__).resolve().parent)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.embodiment.train_intervention_classifier import (  # noqa: E402
    ClassifierConfig,
    InterventionClassifier,
    binary_classification_metrics,
    build_classifier_rows,
)


def threshold_sweep(logits: torch.Tensor, labels: torch.Tensor) -> list[dict[str, Any]]:
    rows = []
    for threshold in np.linspace(0.05, 0.95, 19):
        metrics = binary_classification_metrics(
            logits, labels, threshold=float(threshold)
        )
        metrics["threshold"] = float(threshold)
        rows.append(metrics)
    return rows


def plot_timeline(
    rows: list[dict[str, Any]],
    probs: np.ndarray,
    labels: np.ndarray,
    output_dir: Path,
    max_trajectories: int = 20,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    by_path: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        by_path.setdefault(str(row["path"]), []).append(i)
    for plot_idx, (path, idxs) in enumerate(sorted(by_path.items())[:max_trajectories]):
        idxs = sorted(idxs, key=lambda i: rows[i]["rel"])
        rel = [rows[i]["rel"] for i in idxs]
        p = probs[idxs]
        y = labels[idxs]
        fig, ax = plt.subplots(figsize=(8, 2.8), dpi=150)
        ax.plot(rel, p, marker="o", label="P(intervene in 2 chunks)")
        ax.scatter(
            [r for r, yy in zip(rel, y) if yy == 1],
            [1.0 for yy in y if yy == 1],
            color="#c43c39",
            marker="x",
            s=60,
            label="positive label",
        )
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("trajectory chunk index")
        ax.set_ylabel("probability")
        ax.set_title(Path(path).name)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(
            output_dir / f"timeline_{plot_idx:03d}_{Path(path).stem}.png",
            bbox_inches="tight",
        )
        plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()

    raw_cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    cfg = ClassifierConfig(**raw_cfg)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ckpt_cfg = checkpoint.get("config", {})
    z_dim = int(ckpt_cfg.get("z_dim", cfg.z_dim))
    hidden_dim = int(ckpt_cfg.get("hidden_dim", cfg.hidden_dim))
    num_layers = int(ckpt_cfg.get("num_layers", cfg.num_layers))
    dropout = float(ckpt_cfg.get("dropout", cfg.dropout))
    threshold = float(
        args.threshold
        if args.threshold is not None
        else ckpt_cfg.get("threshold", cfg.threshold)
    )
    model = InterventionClassifier(z_dim, hidden_dim, num_layers, dropout)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    data = build_classifier_rows(cfg)
    with torch.no_grad():
        logits = model(data["z"].float()).cpu()
    labels = data["labels"].long().cpu()
    probs = torch.sigmoid(logits).numpy()
    metrics = binary_classification_metrics(logits, labels, threshold=threshold)
    sweep = threshold_sweep(logits, labels)
    best = max(sweep, key=lambda item: (float(item["f1"]), float(item["recall"])))

    output_dir = Path(args.output_dir or Path(cfg.output_dir) / "offline_eval")
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "metrics_at_threshold": metrics,
        "best_threshold_by_f1": best,
        "num_rows": int(labels.numel()),
    }
    (output_dir / "eval_metrics.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    with (output_dir / "threshold_sweep.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(sweep[0].keys()))
        writer.writeheader()
        writer.writerows(sweep)
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "path",
            "traj_index",
            "rel",
            "label",
            "probability",
            "prediction",
            "source_intervention_steps",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row, prob, label in zip(data["rows"], probs.tolist(), labels.tolist()):
            writer.writerow(
                {
                    "path": row["path"],
                    "traj_index": row["traj_index"],
                    "rel": row["rel"],
                    "label": int(label),
                    "probability": float(prob),
                    "prediction": int(prob >= threshold),
                    "source_intervention_steps": json.dumps(
                        row.get("source_intervention_steps", [])
                    ),
                }
            )
    plot_timeline(data["rows"], probs, labels.numpy(), output_dir / "timelines")
    print(json.dumps(payload, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
