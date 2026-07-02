#!/usr/bin/env python3
# Copyright 2026 The GIGA Authors.
#
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn


def _find_repo_root(start: Path) -> Path:
    for path in [start, *start.parents]:
        if (path / "rlinf").is_dir() and (path / "examples").is_dir():
            return path
    raise RuntimeError(f"Could not find RLinf repo root from {start}")


REPO_ROOT = _find_repo_root(Path(__file__).resolve().parent)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass
class ClassifierConfig:
    train_data_dirs: list[str]
    output_dir: str
    norm_stats_path: str
    urdf_path: str
    model_path: str | None = None
    rl_token_path: str | None = None
    feature_cache: str | None = None
    generate_feature_cache_if_missing: bool = True
    feature_cache_batch_size: int = 16
    feature_cache_task_description: str = "peg and insertion"
    feature_cache_model_config: str | None = None
    pre_intervention_chunks: int = 2
    include_positive_window: int = 1
    negative_sample_ratio: float = 3.0
    min_rel_chunk: int = 0
    max_steps: int = 3000
    batch_size: int = 256
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.1
    z_dim: int = 2048
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    seed: int = 1234
    gpu: int = 0
    log_interval: int = 20
    save_interval: int = 100
    patience: int = 500
    threshold: float = 0.5


def trajectory_number_from_path(path: Path) -> int:
    match = re.search(r"trajectory_(\d+)_", path.name)
    return int(match.group(1)) if match else -1


def compute_pre_intervention_labels(
    intervene_flags: torch.Tensor,
    pre_intervention_chunks: int = 2,
    include_positive_window: int = 1,
) -> tuple[torch.Tensor, dict[int, list[int]]]:
    if intervene_flags.ndim == 0:
        raise ValueError("intervene_flags must have at least one dimension")
    flags_by_step = (
        intervene_flags.detach()
        .cpu()
        .bool()
        .reshape(intervene_flags.shape[0], -1)
        .any(dim=1)
    )
    labels = torch.zeros(flags_by_step.shape[0], dtype=torch.long)
    source_steps: dict[int, list[int]] = {}
    all_intervention_steps = flags_by_step.nonzero(as_tuple=False).reshape(-1).tolist()
    # Only use onset steps: first step of each contiguous intervention block
    intervention_steps = [
        s
        for i, s in enumerate(all_intervention_steps)
        if i == 0 or s != all_intervention_steps[i - 1] + 1
    ]
    for intervention_step in intervention_steps:
        for offset in range(int(include_positive_window)):
            target_step = int(intervention_step) - int(pre_intervention_chunks) - offset
            if target_step < 0 or target_step >= labels.numel():
                continue
            labels[target_step] = 1
            source_steps.setdefault(target_step, []).append(int(intervention_step))
    return labels, source_steps


def grouped_split_trajectories(
    rows: list[dict[str, Any]], val_fraction: float, test_fraction: float, seed: int
) -> dict[str, list[int]]:
    by_traj: dict[str, list[int]] = {}
    for idx, row in enumerate(rows):
        by_traj.setdefault(str(row["traj_key"]), []).append(idx)
    traj_keys = sorted(by_traj)
    rng = random.Random(int(seed))
    rng.shuffle(traj_keys)
    n = len(traj_keys)
    n_test = (
        max(1, int(round(n * float(test_fraction))))
        if n >= 3 and test_fraction > 0
        else 0
    )
    n_val = (
        max(1, int(round(n * float(val_fraction))))
        if n >= 3 and val_fraction > 0
        else 0
    )
    if n_test + n_val >= n:
        n_test = min(n_test, max(0, n - 2))
        n_val = min(n_val, max(0, n - n_test - 1))
    test_keys = set(traj_keys[:n_test])
    val_keys = set(traj_keys[n_test : n_test + n_val])
    split = {"train": [], "val": [], "test": []}
    for key, indices in by_traj.items():
        if key in test_keys:
            split["test"].extend(indices)
        elif key in val_keys:
            split["val"].extend(indices)
        else:
            split["train"].extend(indices)
    return split


class InterventionClassifier(nn.Module):
    def __init__(self, z_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = int(z_dim)
        for _ in range(int(num_layers)):
            layers.append(nn.Linear(in_dim, int(hidden_dim)))
            layers.append(nn.LayerNorm(int(hidden_dim)))
            layers.append(nn.GELU())
            if float(dropout) > 0:
                layers.append(nn.Dropout(float(dropout)))
            in_dim = int(hidden_dim)
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z.float()).squeeze(-1)


def binary_classification_metrics(
    logits: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5
) -> dict[str, float | int]:
    probs = torch.sigmoid(logits.detach().float()).cpu()
    labels = labels.detach().long().cpu()
    preds = (probs >= float(threshold)).long()
    tp = int(((preds == 1) & (labels == 1)).sum().item())
    tn = int(((preds == 0) & (labels == 0)).sum().item())
    fp = int(((preds == 1) & (labels == 0)).sum().item())
    fn = int(((preds == 0) & (labels == 1)).sum().item())
    total = max(1, int(labels.numel()))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": (tp + tn) / total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "positive_rate": float(preds.float().mean().item()) if labels.numel() else 0.0,
        "label_positive_rate": float(labels.float().mean().item())
        if labels.numel()
        else 0.0,
    }


def resolve_feature_cache_path(cfg: ClassifierConfig) -> Path:
    if cfg.feature_cache:
        return Path(cfg.feature_cache)
    return Path(cfg.output_dir) / "intervention_classifier_rltoken_features.pt"


def select_obs_at_chunk(
    obs: dict[str, Any], rel: int, task_description: str
) -> dict[str, Any]:
    env_obs: dict[str, Any] = {}
    for src_key, out_key in (
        ("states", "states"),
        ("main_images", "main_images"),
        ("extra_view_images", "extra_view_images"),
        ("task_descriptions", "task_descriptions"),
    ):
        if src_key not in obs:
            continue
        value = obs[src_key]
        if torch.is_tensor(value):
            env_obs[out_key] = value[rel, 0].unsqueeze(0).contiguous()
        elif isinstance(value, list):
            env_obs[out_key] = value[rel] if rel < len(value) else value[-1]
        else:
            env_obs[out_key] = value
    if "task_descriptions" not in env_obs:
        env_obs["task_descriptions"] = [task_description]
    return env_obs


@torch.no_grad()
def extract_rl_token_from_model(model: Any, env_obs: dict[str, Any]) -> torch.Tensor:
    prefix_output, _, _ = model._build_prefix_cache_from_obs(env_obs)
    image_features = model._select_prefix_features(prefix_output)
    token = model.rl_token_autoencoder.encoder(image_features).detach().cpu().float()
    if token.dim() == 3:
        token = token.mean(dim=1)
    return token


def build_feature_model_cfg(cfg: ClassifierConfig):
    from omegaconf import OmegaConf

    if cfg.feature_cache_model_config:
        source_cfg = OmegaConf.load(cfg.feature_cache_model_config)
        model_cfg = OmegaConf.create(
            OmegaConf.to_container(source_cfg.actor.model, resolve=True)
        )
    else:
        model_cfg = OmegaConf.create({})
    if not model_cfg.get("openpi", None):
        raise ValueError(
            "feature_cache_model_config must point to an OpenPI data-collection yaml"
        )
    model_cfg.model_type = "openpi_rl_token"
    model_cfg.model_path = cfg.model_path
    model_cfg.rl_token_path = cfg.rl_token_path
    if not model_cfg.get("precision", None):
        model_cfg.precision = "bf16"
    model_cfg.is_lora = bool(model_cfg.get("is_lora", False))
    model_cfg.freeze_rl_token = True
    return model_cfg


def generate_feature_cache(cfg: ClassifierConfig) -> Path:
    if cfg.model_path is None or cfg.rl_token_path is None:
        raise ValueError(
            "model_path and rl_token_path are required to generate zrl features"
        )
    from rlinf.models import get_model

    out_path = resolve_feature_cache_path(cfg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")
    model = get_model(build_feature_model_cfg(cfg)).to(device).eval()
    tokens: list[torch.Tensor] = []
    metas: list[dict[str, Any]] = []
    for data_dir in cfg.train_data_dirs:
        root = Path(data_dir)
        traj_paths = sorted(
            root.glob("trajectory_*.pt"), key=trajectory_number_from_path
        )
        for traj_path in traj_paths:
            traj = torch.load(traj_path, map_location="cpu", weights_only=False)
            curr_obs = traj.get("curr_obs", {})
            if not isinstance(curr_obs, dict) or "states" not in curr_obs:
                continue
            num_chunks = int(curr_obs["states"].shape[0])
            rewards = traj.get("rewards")
            success = bool(
                rewards is not None
                and float(torch.as_tensor(rewards).float().sum().item()) > 0
            )
            for rel in range(num_chunks):
                env_obs = select_obs_at_chunk(
                    curr_obs, rel, cfg.feature_cache_task_description
                )
                token = extract_rl_token_from_model(model, env_obs)
                tokens.append(token.squeeze(0))
                metas.append(
                    {
                        "path": str(traj_path),
                        "data_dir": str(root),
                        "rel_chunk": int(rel),
                        "traj_index": int(trajectory_number_from_path(traj_path)),
                        "success": success,
                    }
                )
            print(f"[feature-cache] {traj_path.name} chunks={num_chunks}", flush=True)
    if not tokens:
        raise RuntimeError("No RL-token features were generated")
    payload = {"rltoken": torch.stack(tokens, dim=0).float().cpu(), "metas": metas}
    torch.save(payload, out_path)
    print(
        json.dumps(
            {
                "feature_cache": str(out_path),
                "num_rows": len(metas),
                "shape": list(payload["rltoken"].shape),
            },
            indent=2,
        ),
        flush=True,
    )
    return out_path


def ensure_feature_cache(cfg: ClassifierConfig) -> Path:
    path = resolve_feature_cache_path(cfg)
    if path.exists():
        return path
    if not cfg.generate_feature_cache_if_missing:
        raise FileNotFoundError(f"Feature cache not found: {path}")
    return generate_feature_cache(cfg)


def load_feature_cache(path: Path) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    z = payload["rltoken"].float()
    metas = list(payload["metas"])
    if z.shape[0] != len(metas):
        raise ValueError(f"Feature rows {z.shape[0]} do not match metas {len(metas)}")
    return z, metas


def build_classifier_rows(cfg: ClassifierConfig) -> dict[str, Any]:
    feature_cache = ensure_feature_cache(cfg)
    z_all, metas = load_feature_cache(feature_cache)
    meta_by_key = {
        (str(meta["path"]), int(meta["rel_chunk"])): idx
        for idx, meta in enumerate(metas)
    }
    rows: list[dict[str, Any]] = []
    loaded_paths = sorted(
        {Path(str(meta["path"])) for meta in metas},
        key=lambda p: (str(p.parent), trajectory_number_from_path(p)),
    )
    label_debug: list[dict[str, Any]] = []
    for traj_path in loaded_paths:
        traj = torch.load(traj_path, map_location="cpu", weights_only=False)
        flags = traj.get("intervene_flags")
        if flags is None:
            continue
        labels, sources = compute_pre_intervention_labels(
            flags, cfg.pre_intervention_chunks, cfg.include_positive_window
        )
        for rel in range(labels.numel()):
            if rel < int(cfg.min_rel_chunk):
                continue
            feature_idx = meta_by_key.get((str(traj_path), int(rel)))
            if feature_idx is None:
                continue
            rows.append(
                {
                    "feature_idx": int(feature_idx),
                    "path": str(traj_path),
                    "traj_key": str(traj_path),
                    "traj_index": trajectory_number_from_path(traj_path),
                    "rel": int(rel),
                    "label": int(labels[rel].item()),
                    "source_intervention_steps": sources.get(int(rel), []),
                }
            )
        positive_steps = [
            int(i) for i in labels.nonzero(as_tuple=False).reshape(-1).tolist()
        ]
        if positive_steps:
            label_debug.append(
                {
                    "path": str(traj_path),
                    "positive_label_steps": positive_steps[:20],
                    "num_positive_labels": len(positive_steps),
                }
            )
    if not rows:
        raise RuntimeError(
            "No classifier rows built; check feature cache and data paths"
        )
    positives = [r for r in rows if r["label"] == 1]
    negatives = [r for r in rows if r["label"] == 0]
    if cfg.negative_sample_ratio > 0 and positives:
        max_negatives = int(
            math.ceil(len(positives) * float(cfg.negative_sample_ratio))
        )
        if len(negatives) > max_negatives:
            rng = random.Random(int(cfg.seed))
            negatives = rng.sample(negatives, max_negatives)
            rows = sorted(positives + negatives, key=lambda r: (r["path"], r["rel"]))
    labels = torch.tensor([r["label"] for r in rows], dtype=torch.long)
    feature_indices = torch.tensor([r["feature_idx"] for r in rows], dtype=torch.long)
    data = {
        "z": z_all[feature_indices],
        "labels": labels,
        "rows": rows,
        "feature_cache": str(feature_cache),
        "label_debug": label_debug,
    }
    summary = {
        "num_rows": len(rows),
        "num_positive": int(labels.sum().item()),
        "num_negative": int((labels == 0).sum().item()),
        "positive_rate": float(labels.float().mean().item()),
        "num_trajectories": len({r["traj_key"] for r in rows}),
        "feature_cache": str(feature_cache),
        "pre_intervention_chunks": int(cfg.pre_intervention_chunks),
        "include_positive_window": int(cfg.include_positive_window),
        "negative_sample_ratio": float(cfg.negative_sample_ratio),
    }
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "classifier_dataset_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (out / "classifier_label_debug.json").write_text(
        json.dumps(label_debug[:200], indent=2), encoding="utf-8"
    )
    print(json.dumps({"classifier_dataset": summary}, ensure_ascii=False), flush=True)
    return data


def make_batch(
    data: dict[str, Any], indices: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    return data["z"][indices].to(device), data["labels"][indices].float().to(device)


def evaluate_model(
    model: nn.Module,
    data: dict[str, Any],
    indices: list[int],
    device: torch.device,
    threshold: float,
) -> dict[str, float | int]:
    if not indices:
        return {
            "loss": 0.0,
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }
    model.eval()
    idx = torch.tensor(indices, dtype=torch.long)
    logits_all = []
    labels_all = []
    losses = []
    with torch.no_grad():
        for start in range(0, len(indices), 1024):
            sub = idx[start : start + 1024]
            z, labels = make_batch(data, sub, device)
            logits = model(z)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            losses.append(float(loss.item()))
            logits_all.append(logits.cpu())
            labels_all.append(labels.cpu().long())
    logits = torch.cat(logits_all)
    labels = torch.cat(labels_all)
    metrics = binary_classification_metrics(logits, labels, threshold=threshold)
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    return metrics


def plot_training_history(history: list[dict[str, Any]], out_path: Path) -> None:
    if not history:
        return
    steps = [h["step"] for h in history]
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), dpi=150, sharex=True)
    axes[0].plot(
        steps, [h.get("train_loss", np.nan) for h in history], label="train loss"
    )
    axes[0].plot(steps, [h.get("val_loss", np.nan) for h in history], label="val loss")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(steps, [h.get("val_f1", np.nan) for h in history], label="val f1")
    axes[1].plot(
        steps, [h.get("val_recall", np.nan) for h in history], label="val recall"
    )
    axes[1].plot(
        steps, [h.get("val_precision", np.nan) for h in history], label="val precision"
    )
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlabel("step")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_checkpoint(
    path: Path,
    model: InterventionClassifier,
    cfg: ClassifierConfig,
    metrics: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "config": asdict(cfg), "metrics": metrics}, path
    )


def train(cfg: ClassifierConfig) -> dict[str, Any]:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = build_classifier_rows(cfg)
    splits = grouped_split_trajectories(
        data["rows"], cfg.val_fraction, cfg.test_fraction, cfg.seed
    )
    device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")
    model = InterventionClassifier(
        cfg.z_dim, cfg.hidden_dim, cfg.num_layers, cfg.dropout
    ).to(device)
    pos_count = (
        int(data["labels"][splits["train"]].sum().item()) if splits["train"] else 1
    )
    neg_count = max(1, len(splits["train"]) - pos_count)
    pos_weight = torch.tensor(
        [neg_count / max(1, pos_count)], dtype=torch.float32, device=device
    )
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    train_indices = list(splits["train"])
    history: list[dict[str, Any]] = []
    best_f1 = -1.0
    best_step = 0
    rng = random.Random(cfg.seed)
    for step in range(1, cfg.max_steps + 1):
        model.train()
        batch_indices = rng.choices(
            train_indices, k=min(cfg.batch_size, len(train_indices))
        )
        idx = torch.tensor(batch_indices, dtype=torch.long)
        z, labels = make_batch(data, idx, device)
        logits = model(z)
        loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % cfg.log_interval == 0 or step == 1:
            val_metrics = evaluate_model(
                model, data, splits["val"], device, cfg.threshold
            )
            record = {
                "step": step,
                "train_loss": float(loss.item()),
                "val_loss": float(val_metrics.get("loss", 0.0)),
                "val_f1": float(val_metrics.get("f1", 0.0)),
                "val_precision": float(val_metrics.get("precision", 0.0)),
                "val_recall": float(val_metrics.get("recall", 0.0)),
            }
            history.append(record)
            print(json.dumps(record), flush=True)
            if record["val_f1"] > best_f1:
                best_f1 = record["val_f1"]
                best_step = step
                save_checkpoint(
                    out / "best_intervention_classifier.pt",
                    model,
                    cfg,
                    {"val": val_metrics, "step": step},
                )
            if step - best_step >= cfg.patience:
                print(
                    f"early_stop step={step} best_step={best_step} best_f1={best_f1:.4f}",
                    flush=True,
                )
                break
        if step % cfg.save_interval == 0:
            save_checkpoint(
                out / f"checkpoint_step_{step}.pt", model, cfg, {"step": step}
            )
    best = torch.load(
        out / "best_intervention_classifier.pt", map_location="cpu", weights_only=False
    )
    model.load_state_dict(best["model"])
    model.to(device)
    final_metrics = {
        name: evaluate_model(model, data, idxs, device, cfg.threshold)
        for name, idxs in splits.items()
    }
    final_payload = {
        "splits": {k: len(v) for k, v in splits.items()},
        "metrics": final_metrics,
        "best_step": int(best["metrics"]["step"]),
    }
    (out / "classifier_metrics.json").write_text(
        json.dumps(final_payload, indent=2), encoding="utf-8"
    )
    (out / "training_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    plot_training_history(history, out / "training_history.png")
    save_checkpoint(out / "final_intervention_classifier.pt", model, cfg, final_payload)
    print(json.dumps(final_payload, indent=2), flush=True)
    return final_payload


def load_config_from_args() -> ClassifierConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    for field_name in ClassifierConfig.__dataclass_fields__:
        parser.add_argument("--" + field_name.replace("_", "-"), nargs="*")
    ns = parser.parse_args()
    cfg_dict: dict[str, Any] = {}
    if ns.config:
        with open(ns.config, encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
        if loaded:
            cfg_dict.update(loaded)
    for field_name, field_def in ClassifierConfig.__dataclass_fields__.items():
        raw = getattr(ns, field_name)
        if raw is None or len(raw) == 0:
            continue
        value: Any = raw if len(raw) > 1 else raw[0]
        current = cfg_dict.get(field_name, field_def.default)
        if isinstance(current, bool):
            value = str(value).lower() in {"1", "true", "yes", "on"}
        elif isinstance(current, int):
            value = int(value)
        elif isinstance(current, float):
            value = float(value)
        cfg_dict[field_name] = value
    return ClassifierConfig(**cfg_dict)


def main() -> int:
    cfg = load_config_from_args()
    train(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
