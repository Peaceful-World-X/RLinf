#!/usr/bin/env python3
# Copyright 2026 The GIGA Authors.
#
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace
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


def resolve_decision_threshold(
    args_threshold: float | None, cfg_threshold: float
) -> float:
    return float(cfg_threshold if args_threshold is None else args_threshold)


def threshold_sweep(logits: torch.Tensor, labels: torch.Tensor) -> list[dict[str, Any]]:
    rows = []
    for threshold in np.linspace(0.05, 0.95, 19):
        metrics = binary_classification_metrics(
            logits, labels, threshold=float(threshold)
        )
        metrics["threshold"] = float(threshold)
        rows.append(metrics)
    return rows


def first_trigger_rel(
    rows: list[dict[str, Any]], probs: np.ndarray, threshold: float
) -> int | None:
    ordered = sorted(
        zip(rows, np.asarray(probs, dtype=np.float64).reshape(-1)),
        key=lambda item: int(item[0]["rel"]),
    )
    for row, prob in ordered:
        if float(prob) >= float(threshold):
            return int(row["rel"])
    return None


def latched_trigger_mask(
    rows: list[dict[str, Any]], probs: np.ndarray, threshold: float
) -> np.ndarray:
    probs_arr = np.asarray(probs, dtype=np.float64).reshape(-1)
    if len(rows) != int(probs_arr.shape[0]):
        raise ValueError(
            f"rows/probs length mismatch: {len(rows)} rows vs {probs_arr.shape[0]} probs"
        )

    out = np.zeros(len(rows), dtype=bool)
    active = False
    for original_idx, _row in sorted(
        enumerate(rows), key=lambda item: int(item[1]["rel"])
    ):
        if not active and float(probs_arr[original_idx]) >= float(threshold):
            active = True
        if active:
            out[original_idx] = True
    return out


def build_trigger_summary(
    rows: list[dict[str, Any]],
    probs: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> list[dict[str, Any]]:
    by_path: dict[str, list[int]] = {}
    for idx, row in enumerate(rows):
        by_path.setdefault(str(row["path"]), []).append(idx)

    out: list[dict[str, Any]] = []
    probs_arr = np.asarray(probs, dtype=np.float64).reshape(-1)
    labels_arr = np.asarray(labels, dtype=np.int64).reshape(-1)
    for path, idxs in sorted(by_path.items()):
        idxs = sorted(idxs, key=lambda i: int(rows[i]["rel"]))
        traj_rows = [rows[i] for i in idxs]
        traj_probs = probs_arr[idxs]
        traj_labels = labels_arr[idxs]
        trigger_rel = first_trigger_rel(traj_rows, traj_probs, threshold)
        actor_mask = latched_trigger_mask(traj_rows, traj_probs, threshold)
        trigger_prob = None
        if trigger_rel is not None:
            for row, prob in zip(traj_rows, traj_probs):
                if int(row["rel"]) == int(trigger_rel):
                    trigger_prob = float(prob)
                    break
        source_steps = sorted(
            {
                int(step)
                for row in traj_rows
                for step in row.get("source_intervention_steps", [])
            }
        )
        positive_rels = [
            int(row["rel"])
            for row, label in zip(traj_rows, traj_labels)
            if int(label) == 1
        ]
        out.append(
            {
                "path": path,
                "traj_index": int(traj_rows[0].get("traj_index", -1)),
                "threshold": float(threshold),
                "has_trigger": trigger_rel is not None,
                "first_trigger_rel": trigger_rel,
                "first_trigger_frame": trigger_rel,
                "first_trigger_probability": trigger_prob,
                "actor_intervention_start_rel": trigger_rel,
                "active_until_end": bool(actor_mask.any()),
                "label_positive_rels": positive_rels,
                "source_intervention_steps": source_steps,
            }
        )
    return out


def _resize_uint8_image(image: np.ndarray, size: int) -> np.ndarray:
    image = np.asarray(image)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    if image.shape[-1] > 3:
        image = image[..., :3]
    h, w = image.shape[:2]
    y_idx = np.linspace(0, max(h - 1, 0), int(size)).astype(np.int64)
    x_idx = np.linspace(0, max(w - 1, 0), int(size)).astype(np.int64)
    return image[np.ix_(y_idx, x_idx)]


def _three_views_from_trajectory(
    trajectory: dict[str, Any], rel: int, thumb_size: int
) -> np.ndarray | None:
    curr_obs = trajectory.get("curr_obs", {})
    if isinstance(curr_obs, dict):
        main = curr_obs.get("main_images")
        extra = curr_obs.get("extra_view_images")
        if torch.is_tensor(main) and torch.is_tensor(extra):
            if rel < main.shape[0] and rel < extra.shape[0] and extra.shape[2] >= 2:
                return np.asarray(
                    [
                        _resize_uint8_image(
                            main[rel, 0].detach().cpu().numpy(), thumb_size
                        ),
                        _resize_uint8_image(
                            extra[rel, 0, 0].detach().cpu().numpy(), thumb_size
                        ),
                        _resize_uint8_image(
                            extra[rel, 0, 1].detach().cpu().numpy(), thumb_size
                        ),
                    ],
                    dtype=np.uint8,
                )

    forward_inputs = trajectory.get("forward_inputs", {})
    if isinstance(forward_inputs, dict):
        main = forward_inputs.get("substep_main_images")
        extra = forward_inputs.get("substep_extra_view_images")
        if torch.is_tensor(main) and torch.is_tensor(extra):
            if rel < main.shape[0] and rel < extra.shape[0] and extra.shape[3] >= 2:
                return np.asarray(
                    [
                        _resize_uint8_image(
                            main[rel, 0, 0].detach().cpu().numpy(), thumb_size
                        ),
                        _resize_uint8_image(
                            extra[rel, 0, 0, 0].detach().cpu().numpy(), thumb_size
                        ),
                        _resize_uint8_image(
                            extra[rel, 0, 0, 1].detach().cpu().numpy(), thumb_size
                        ),
                    ],
                    dtype=np.uint8,
                )
    return None


def extract_three_view_images_for_rows(
    rows: list[dict[str, Any]], thumb_size: int = 64
) -> np.ndarray:
    loaded: dict[str, dict[str, Any]] = {}
    views = []
    blank = np.zeros((3, int(thumb_size), int(thumb_size), 3), dtype=np.uint8)
    for row in rows:
        path = str(row["path"])
        if path not in loaded:
            loaded[path] = torch.load(path, map_location="cpu", weights_only=False)
        image = _three_views_from_trajectory(loaded[path], int(row["rel"]), thumb_size)
        views.append(blank if image is None else image)
    return np.asarray(views, dtype=np.uint8)


def _shade_boolean_regions(ax, mask: np.ndarray, color: str, alpha: float) -> None:
    active = np.flatnonzero(np.asarray(mask, dtype=bool).reshape(-1))
    if active.size == 0:
        return
    start = int(active[0])
    prev = int(active[0])
    for raw_idx in active[1:]:
        idx = int(raw_idx)
        if idx != prev + 1:
            ax.axvspan(start - 0.5, prev + 0.5, color=color, alpha=alpha)
            start = idx
        prev = idx
    ax.axvspan(start - 0.5, prev + 0.5, color=color, alpha=alpha)


def select_image_timeline_groups(
    rows: list[dict[str, Any]],
    probs: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    max_trajectories: int = 20,
) -> list[dict[str, Any]]:
    by_path: dict[str, list[int]] = {}
    for idx, row in enumerate(rows):
        by_path.setdefault(str(row["path"]), []).append(idx)

    probs_arr = np.asarray(probs, dtype=np.float64).reshape(-1)
    labels_arr = np.asarray(labels, dtype=np.int64).reshape(-1)
    positive_groups: list[dict[str, Any]] = []
    negative_groups: list[dict[str, Any]] = []
    for path, idxs in sorted(by_path.items()):
        idxs = sorted(idxs, key=lambda i: int(rows[i]["rel"]))
        group_rows = [rows[i] for i in idxs]
        group_probs = probs_arr[idxs]
        group_labels = labels_arr[idxs]
        trigger_rel = first_trigger_rel(group_rows, group_probs, threshold)
        has_positive = bool(np.any(group_labels == 1))
        group = {
            "path": path,
            "indices": idxs,
            "category": "positive" if has_positive else "negative",
            "has_trigger": trigger_rel is not None,
            "first_trigger_rel": trigger_rel,
        }
        if has_positive:
            positive_groups.append(group)
        else:
            negative_groups.append(group)

    def sort_key(group: dict[str, Any]) -> tuple[int, int, str]:
        trigger_rel = group["first_trigger_rel"]
        return (
            0 if group["has_trigger"] else 1,
            int(trigger_rel) if trigger_rel is not None else 10**9,
            str(group["path"]),
        )

    positive_groups = sorted(positive_groups, key=sort_key)
    negative_groups = sorted(negative_groups, key=sort_key)
    max_trajectories = int(max_trajectories)
    if max_trajectories <= 0:
        return []
    if positive_groups and negative_groups:
        positive_budget = min(len(positive_groups), max_trajectories // 2)
        negative_budget = min(len(negative_groups), max_trajectories - positive_budget)
        spare = max_trajectories - positive_budget - negative_budget
        if spare > 0:
            positive_budget = min(len(positive_groups), positive_budget + spare)
        return positive_groups[:positive_budget] + negative_groups[:negative_budget]
    return (positive_groups or negative_groups)[:max_trajectories]


def plot_image_timeline(
    rows: list[dict[str, Any]],
    probs: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    out_path: Path,
    thumb_size: int = 64,
) -> None:
    if not rows:
        return
    ordered = sorted(
        zip(rows, np.asarray(probs).reshape(-1), np.asarray(labels).reshape(-1)),
        key=lambda item: int(item[0]["rel"]),
    )
    rows = [item[0] for item in ordered]
    probs_arr = np.asarray([item[1] for item in ordered], dtype=np.float64)
    labels_arr = np.asarray([item[2] for item in ordered], dtype=np.int64)
    rels = np.asarray([int(row["rel"]) for row in rows], dtype=np.int64)
    x = np.arange(len(rows), dtype=np.int64)
    raw_pred_mask = probs_arr >= float(threshold)
    actor_mask = latched_trigger_mask(rows, probs_arr, threshold)
    trigger_rel = first_trigger_rel(rows, probs_arr, threshold)
    trigger_pos = None
    if trigger_rel is not None:
        matches = np.flatnonzero(rels == int(trigger_rel))
        if matches.size:
            trigger_pos = int(matches[0])

    images = extract_three_view_images_for_rows(rows, thumb_size=thumb_size)
    height_ratios = [1.6, 0.7, 0.9, 0.9, 0.9]
    width = min(max(12.0, 0.42 * len(rows)), 44.0)
    fig, axes = plt.subplots(
        len(height_ratios),
        1,
        figsize=(width, 2.0 + sum(height_ratios) * 1.2),
        dpi=150,
        sharex=True,
        gridspec_kw={"height_ratios": height_ratios},
    )
    axes = np.asarray(axes).reshape(-1)

    ax_prob = axes[0]
    _shade_boolean_regions(ax_prob, actor_mask, "#f2c94c", 0.18)
    ax_prob.plot(
        x,
        probs_arr,
        color="#2a6fbb",
        marker="o",
        linewidth=1.6,
        label="P(actor intervention)",
    )
    ax_prob.axhline(
        float(threshold),
        color="#555555",
        linestyle="--",
        linewidth=1.1,
        label=f"threshold={threshold:.2f}",
    )
    positive_pos = x[labels_arr == 1]
    if positive_pos.size:
        ax_prob.scatter(
            positive_pos,
            probs_arr[labels_arr == 1],
            marker="x",
            s=65,
            color="#c43c39",
            label="label: pre-intervention",
        )
    if trigger_pos is not None:
        ax_prob.axvline(
            trigger_pos,
            color="#8e2f8f",
            linewidth=1.7,
            label=f"first trigger chunk={trigger_rel}",
        )
        ax_prob.scatter(
            [trigger_pos],
            [probs_arr[trigger_pos]],
            marker="*",
            s=130,
            color="#8e2f8f",
            zorder=5,
        )
    ax_prob.set_ylim(-0.05, 1.05)
    ax_prob.set_ylabel("probability")
    ax_prob.grid(True, alpha=0.25)
    ax_prob.legend(loc="best", ncol=3)
    ax_prob.set_title(
        f"{Path(str(rows[0]['path'])).name}, actor_start_chunk={trigger_rel if trigger_rel is not None else 'none'}"
    )

    ax_event = axes[1]
    _shade_boolean_regions(ax_event, actor_mask, "#f2c94c", 0.18)
    ax_event.step(
        x,
        actor_mask.astype(float),
        where="mid",
        color="#2a6fbb",
        label="actor active (latched)",
    )
    raw_trigger_pos = x[raw_pred_mask]
    if raw_trigger_pos.size:
        ax_event.scatter(
            raw_trigger_pos,
            np.ones_like(raw_trigger_pos) * 0.22,
            marker="|",
            color="#2a6fbb",
            s=70,
            label="raw classifier >= threshold",
        )
    ax_event.scatter(
        positive_pos,
        np.ones_like(positive_pos) * 0.55,
        marker="x",
        color="#c43c39",
        s=50,
        label="label chunk",
    )
    source_steps = sorted(
        {int(step) for row in rows for step in row.get("source_intervention_steps", [])}
    )
    rel_to_pos = {int(rel): pos for pos, rel in enumerate(rels.tolist())}
    onset_pos = [rel_to_pos[step] for step in source_steps if step in rel_to_pos]
    if onset_pos:
        ax_event.scatter(
            onset_pos,
            np.ones(len(onset_pos)) * 0.9,
            marker="v",
            color="#222222",
            s=45,
            label="recorded intervention onset",
        )
    if trigger_pos is not None:
        ax_event.axvline(trigger_pos, color="#8e2f8f", linewidth=1.7)
    ax_event.set_ylim(-0.1, 1.15)
    ax_event.set_yticks([0, 1])
    ax_event.set_ylabel("event")
    ax_event.grid(True, alpha=0.25)
    ax_event.legend(loc="best", ncol=3)

    view_names = ["main", "extra0", "extra1"]
    for view_idx in range(3):
        ax_img = axes[2 + view_idx]
        strip = np.concatenate([images[i, view_idx] for i in range(len(rows))], axis=1)
        ax_img.imshow(strip, aspect="auto", extent=(-0.5, len(rows) - 0.5, 0, 1))
        _shade_boolean_regions(ax_img, actor_mask, "#f2c94c", 0.16)
        if trigger_pos is not None:
            ax_img.axvline(trigger_pos, color="#8e2f8f", linewidth=1.7)
        for pos in range(len(rows) + 1):
            ax_img.axvline(pos - 0.5, color="white", linewidth=0.25, alpha=0.45)
        ax_img.set_yticks([])
        ax_img.set_ylabel(view_names[view_idx], rotation=0, labelpad=28, va="center")

    tick_stride = max(1, int(np.ceil(len(rows) / 18)))
    tick_positions = x[::tick_stride]
    tick_labels = [str(int(rel)) for rel in rels[::tick_stride]]
    axes[-1].set_xticks(tick_positions)
    axes[-1].set_xticklabels(tick_labels)
    axes[-1].set_xlabel("trajectory chunk/frame index")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_image_timelines(
    rows: list[dict[str, Any]],
    probs: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    output_dir: Path,
    max_trajectories: int = 20,
    thumb_size: int = 64,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_png in output_dir.glob("*.png"):
        old_png.unlink()
    groups = select_image_timeline_groups(
        rows,
        probs,
        labels,
        threshold,
        max_trajectories=max_trajectories,
    )
    for plot_idx, group in enumerate(groups):
        idxs = group["indices"]
        path = str(group["path"])
        category = str(group["category"])
        plot_image_timeline(
            [rows[i] for i in idxs],
            np.asarray([probs[i] for i in idxs]),
            np.asarray([labels[i] for i in idxs]),
            threshold,
            output_dir
            / f"{category}_image_timeline_{plot_idx:03d}_{Path(path).stem}.png",
            thumb_size=thumb_size,
        )


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
    parser.add_argument("--max-image-timelines", type=int, default=20)
    parser.add_argument("--timeline-thumb-size", type=int, default=64)
    args = parser.parse_args()

    raw_cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    cfg = ClassifierConfig(**raw_cfg)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ckpt_cfg = checkpoint.get("config", {})
    z_dim = int(ckpt_cfg.get("z_dim", cfg.z_dim))
    hidden_dim = int(ckpt_cfg.get("hidden_dim", cfg.hidden_dim))
    num_layers = int(ckpt_cfg.get("num_layers", cfg.num_layers))
    dropout = float(ckpt_cfg.get("dropout", cfg.dropout))
    model = InterventionClassifier(z_dim, hidden_dim, num_layers, dropout)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    output_dir = Path(args.output_dir or Path(cfg.output_dir) / "offline_eval")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Keep the original sampled eval metrics, but use full trajectories for
    # trigger localization so the first triggered frame is not hidden by
    # negative downsampling.
    data = build_classifier_rows(
        replace(cfg, output_dir=str(output_dir / "sampled_dataset"))
    )
    full_data = build_classifier_rows(
        replace(
            cfg, negative_sample_ratio=0.0, output_dir=str(output_dir / "full_dataset")
        )
    )
    with torch.no_grad():
        logits = model(data["z"].float()).cpu()
        full_logits = model(full_data["z"].float()).cpu()
    labels = data["labels"].long().cpu()
    full_labels = full_data["labels"].long().cpu()
    probs = torch.sigmoid(logits).numpy()
    full_probs = torch.sigmoid(full_logits).numpy()
    sweep = threshold_sweep(logits, labels)
    best = max(sweep, key=lambda item: (float(item["f1"]), float(item["recall"])))
    decision_threshold = resolve_decision_threshold(args.threshold, cfg.threshold)
    metrics = binary_classification_metrics(
        logits, labels, threshold=decision_threshold
    )
    full_metrics = binary_classification_metrics(
        full_logits, full_labels, threshold=decision_threshold
    )
    trigger_summary = build_trigger_summary(
        full_data["rows"], full_probs, full_labels.numpy(), decision_threshold
    )
    sampled_latched_actor_mask = np.zeros(len(data["rows"]), dtype=bool)
    full_latched_actor_mask = np.zeros(len(full_data["rows"]), dtype=bool)
    sampled_by_path: dict[str, list[int]] = {}
    full_by_path: dict[str, list[int]] = {}
    for idx, row in enumerate(data["rows"]):
        sampled_by_path.setdefault(str(row["path"]), []).append(idx)
    for idx, row in enumerate(full_data["rows"]):
        full_by_path.setdefault(str(row["path"]), []).append(idx)
    probs_arr = np.asarray(probs, dtype=np.float64).reshape(-1)
    full_probs_arr = np.asarray(full_probs, dtype=np.float64).reshape(-1)
    for idxs in sampled_by_path.values():
        ordered_idxs = sorted(idxs, key=lambda i: int(data["rows"][i]["rel"]))
        group_mask = latched_trigger_mask(
            [data["rows"][i] for i in ordered_idxs],
            probs_arr[ordered_idxs],
            decision_threshold,
        )
        sampled_latched_actor_mask[np.asarray(ordered_idxs, dtype=np.int64)] = (
            group_mask
        )
    for idxs in full_by_path.values():
        ordered_idxs = sorted(idxs, key=lambda i: int(full_data["rows"][i]["rel"]))
        group_mask = latched_trigger_mask(
            [full_data["rows"][i] for i in ordered_idxs],
            full_probs_arr[ordered_idxs],
            decision_threshold,
        )
        full_latched_actor_mask[np.asarray(ordered_idxs, dtype=np.int64)] = group_mask

    payload = {
        "metrics_at_threshold": metrics,
        "full_trajectory_metrics_at_threshold": full_metrics,
        "best_threshold_by_f1": best,
        "decision_threshold": decision_threshold,
        "num_rows": int(labels.numel()),
        "num_full_rows": int(full_labels.numel()),
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
            "raw_prediction",
            "latched_actor_intervention",
            "source_intervention_steps",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, (row, prob, label) in enumerate(
            zip(data["rows"], probs.tolist(), labels.tolist())
        ):
            writer.writerow(
                {
                    "path": row["path"],
                    "traj_index": row["traj_index"],
                    "rel": row["rel"],
                    "label": int(label),
                    "probability": float(prob),
                    "prediction": int(prob >= decision_threshold),
                    "raw_prediction": int(prob >= decision_threshold),
                    "latched_actor_intervention": int(sampled_latched_actor_mask[idx]),
                    "source_intervention_steps": json.dumps(
                        row.get("source_intervention_steps", [])
                    ),
                }
            )
    with (output_dir / "full_predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        fieldnames = [
            "path",
            "traj_index",
            "rel",
            "label",
            "probability",
            "prediction",
            "raw_prediction",
            "latched_actor_intervention",
            "source_intervention_steps",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, (row, prob, label) in enumerate(
            zip(full_data["rows"], full_probs.tolist(), full_labels.tolist())
        ):
            writer.writerow(
                {
                    "path": row["path"],
                    "traj_index": row["traj_index"],
                    "rel": row["rel"],
                    "label": int(label),
                    "probability": float(prob),
                    "prediction": int(prob >= decision_threshold),
                    "raw_prediction": int(prob >= decision_threshold),
                    "latched_actor_intervention": int(full_latched_actor_mask[idx]),
                    "source_intervention_steps": json.dumps(
                        row.get("source_intervention_steps", [])
                    ),
                }
            )
    with (output_dir / "trigger_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        fieldnames = [
            "path",
            "traj_index",
            "threshold",
            "has_trigger",
            "first_trigger_rel",
            "first_trigger_frame",
            "first_trigger_probability",
            "actor_intervention_start_rel",
            "active_until_end",
            "label_positive_rels",
            "source_intervention_steps",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in trigger_summary:
            encoded = dict(row)
            encoded["label_positive_rels"] = json.dumps(row["label_positive_rels"])
            encoded["source_intervention_steps"] = json.dumps(
                row["source_intervention_steps"]
            )
            writer.writerow(encoded)

    plot_timeline(data["rows"], probs, labels.numpy(), output_dir / "timelines")
    plot_image_timelines(
        full_data["rows"],
        full_probs,
        full_labels.numpy(),
        decision_threshold,
        output_dir / "image_timelines",
        max_trajectories=args.max_image_timelines,
        thumb_size=args.timeline_thumb_size,
    )
    print(json.dumps(payload, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
