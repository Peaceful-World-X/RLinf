# Copyright 2026 The RLinf Authors.
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

"""Evaluate an RLinf G1 Stage 1 checkpoint on fixed held-out frames."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from rlinf.utils.logging import get_logger

logger = get_logger()


def load_config(path: Path):
    """Compose the same Hydra task configuration used by Stage 1 training."""
    path = path.expanduser().resolve()
    repo = Path(__file__).resolve().parents[1]
    os.environ.setdefault("EMBODIED_PATH", str(repo / "examples/embodiment"))
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    with initialize_config_dir(version_base=None, config_dir=str(path.parent)):
        return compose(config_name=path.stem)


def build_eval_dataset(cfg, dataset: Path, split: Path):
    """Reuse the official SFT transforms and the local LeRobot v3 adapter."""
    from rlinf.data.datasets.openpi.lerobot_v3_dataset import LeRobotV3Dataset
    from rlinf.data.datasets.openpi.official_sft_data_loader import (
        build_official_openpi_sft_dataloader,
    )

    indices = json.loads(split.read_text())
    train = indices["train_episode_indices"]
    val = indices["val_episode_indices"]
    if not train or not val or set(train) & set(val):
        raise ValueError("Train/validation episode lists must be nonempty and disjoint")
    if len(set(train)) != len(train) or len(set(val)) != len(val):
        raise ValueError("Episode split contains duplicate indices")
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    for key, value in {
        "data.episode_split_path": str(split),
        "data.num_workers": 0,
        "actor.eval_batch_size": 1,
    }.items():
        OmegaConf.update(cfg, key, value, force_add=True)
    loader, _ = build_official_openpi_sft_dataloader(
        cfg, world_size=1, rank=0, data_paths=str(dataset), eval_dataset=True
    )
    transformed = loader._data_loader.torch_loader.dataset
    raw = transformed
    while hasattr(raw, "_dataset"):
        raw = raw._dataset
    if not isinstance(raw, LeRobotV3Dataset):
        raise TypeError("This evaluator requires a local LeRobot v3 dataset")
    return transformed, raw


def select_samples(raw, count: int, seed: int) -> list[dict[str, int]]:
    """Select frames uniformly without replacement over all held-out frames."""
    if not 0 < count <= len(raw):
        raise ValueError(f"Expected 1 <= samples <= {len(raw)}, found {count}")
    offsets = np.cumsum([0] + [episode["length"] for episode in raw.episodes])
    records = []
    for index in sorted(random.Random(seed).sample(range(len(raw)), count)):
        position = int(np.searchsorted(offsets, index, side="right") - 1)
        episode = raw.episodes[position]
        frame = int(index - offsets[position])
        records.append(
            {
                "subset_index": index,
                "episode_index": int(episode["episode_index"]),
                "frame_index": frame,
                "dataset_index": int(episode["dataset_from_index"] + frame),
            }
        )
    return records


def load_checkpoint(cfg, checkpoint: Path):
    """Require complete Stage 1 weights before evaluating with mixed precision."""
    from rlinf.models.embodiment.openpi import get_model
    from rlinf.models.embodiment.openpi.checkpoint import (
        _normalize_state_dict,
        resolve_full_weights,
    )
    from rlinf.utils.ckpt_convertor.openpi._core import as_state_dict

    path = resolve_full_weights(checkpoint)
    if path is None:
        raise FileNotFoundError(f"No full_weights.pt found at {checkpoint}")
    state = _normalize_state_dict(
        as_state_dict(
            torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        )
    )
    rlt_keys = [key for key in state if key.startswith("rlt_module.")]
    if not rlt_keys:
        raise ValueError(
            "Checkpoint has no RLT tensors; base SFT weights are insufficient"
        )
    model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.actor.model, resolve=True))
    model_cfg.model_path = str(path)
    model_cfg.precision = "bf16"
    model_cfg.openpi.task = "sft"
    model_cfg.openpi.embed_dtype = "bf16"
    model_cfg.openpi.torch_compile = False
    if not model_cfg.openpi.use_rlt or not model_cfg.openpi.rlt_use_mask:
        raise ValueError("Evaluation requires use_rlt=True and rlt_use_mask=True")
    model = get_model(model_cfg)
    expected = model.state_dict()
    missing, unexpected = (
        sorted(set(expected) - set(state)),
        sorted(set(state) - set(expected)),
    )
    mismatched = [
        key
        for key in expected.keys() & state.keys()
        if expected[key].shape != state[key].shape
    ]
    if missing or unexpected or mismatched:
        raise ValueError(
            f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}, shapes={mismatched}"
        )
    del state, expected
    model.rlt_module.float()
    for name, parameter in model.named_parameters():
        if not torch.isfinite(parameter).all().item():
            raise ValueError(f"Nonfinite checkpoint tensor: {name}")
    model.eval().requires_grad_(False)
    return model, {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "mtime_ns": path.stat().st_mtime_ns,
        "rlt_tensor_count": len(rlt_keys),
        "strict_key_and_shape_match": True,
        "parameters_finite": True,
        "precision": "Official selective BF16 backbone, FP32 islands and RLT module",
    }


def masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    weights = mask.to(dtype=torch.float32).unsqueeze(-1)
    denominator = weights.sum() * target.shape[-1]
    if denominator.item() <= 0:
        raise ValueError("Observation has no valid prefix tokens")
    return (
        (prediction.float() - target.float()).square() * weights
    ).sum() / denominator


@torch.inference_mode()
def evaluate_sample(model, sample: dict) -> dict:
    from torch.utils.data import default_collate

    from rlinf.models.embodiment.openpi.modules.model import (
        Observation,
        _observation_to_dtype,
        preprocess_observation,
    )

    observation = Observation.from_dict(default_collate([sample]))
    observation = model._observation_to_device(observation)
    observation = preprocess_observation(observation, train=False)
    observation = _observation_to_dtype(observation, model.embed_dtype)
    prefix, mask, cache = model.build_prefix_cache(observation)
    del cache
    prefix, mask = model._select_rlt_prefix_embeddings(
        prefix, mask, observation.tokenized_prompt
    )
    if not torch.isfinite(prefix).all().item():
        raise ValueError("Nonfinite backbone prefix features")
    prefix = prefix.float()
    loss, extras = model._rlt_forward(prefix, mask)
    z = extras["z_rl"]
    if z.shape != (1, model.rlt_module.z_dim) or not torch.isfinite(z).all().item():
        raise ValueError(f"Invalid RLT output shape or values: {tuple(z.shape)}")
    if z.dtype != torch.float32:
        raise ValueError(f"Expected FP32 RLT output, found {z.dtype}")
    weights = mask.float().unsqueeze(-1)
    mean_prefix = (prefix * weights).sum(dim=1, keepdim=True) / weights.sum(
        dim=1, keepdim=True
    ).clamp_min(1)
    zero_prediction = model.rlt_module.decode(
        torch.zeros_like(z[:, None, :]), prefix, mask
    )
    values = {
        "rlt_mse": float(loss),
        "mean_prefix_mse": float(masked_mse(mean_prefix, prefix, mask)),
        "zero_token_mse": float(masked_mse(zero_prediction, prefix, mask)),
        "valid_prefix_tokens": int(mask.sum()),
        "prefix_dim": int(prefix.shape[-1]),
        "z_dim": int(z.shape[-1]),
        "z_l2_norm": float(z.norm()),
    }
    if not all(np.isfinite(value) for value in values.values()):
        raise ValueError(f"Nonfinite evaluation result: {values}")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output", type=Path, required=True, help="Evaluation JSON path"
    )
    args = parser.parse_args()
    start = time.monotonic()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    cfg = load_config(args.config)
    dataset, raw = build_eval_dataset(cfg, args.dataset, args.split)
    samples = select_samples(raw, args.samples, args.seed)
    manifest = {
        "seed": args.seed,
        "selection": "Python Random.sample without replacement over all held-out frames",
        "dataset": str(args.dataset.resolve()),
        "split": str(args.split.resolve()),
        "split_sha256": hashlib.sha256(args.split.read_bytes()).hexdigest(),
        "heldout_frame_count": len(raw),
        "heldout_episode_count": len(raw.episodes),
        "samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output.with_suffix(".samples.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    logger.info("Selected %d held-out frames; manifest=%s", len(samples), manifest_path)
    model, checkpoint_info = load_checkpoint(cfg, args.checkpoint)
    model.to(torch.device(args.device))
    rows = []
    for index, sample in enumerate(samples):
        metrics = evaluate_sample(model, dataset[sample["subset_index"]])
        rows.append({**sample, **metrics})
        if (index + 1) % 16 == 0 or index + 1 == len(samples):
            logger.info("Evaluated %d/%d frames", index + 1, len(samples))
    weights = np.array([row["valid_prefix_tokens"] * row["prefix_dim"] for row in rows])
    aggregate = {
        key: float(np.average([row[key] for row in rows], weights=weights))
        for key in ("rlt_mse", "mean_prefix_mse", "zero_token_mse")
    }
    aggregate["zero_token_mse_increase"] = (
        aggregate["zero_token_mse"] - aggregate["rlt_mse"]
    )
    aggregate["better_than_mean_prefix"] = (
        aggregate["rlt_mse"] < aggregate["mean_prefix_mse"]
    )
    report = {
        "checkpoint": checkpoint_info,
        "config": str(args.config.resolve()),
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "device": args.device,
        "preprocessing": "Official dataset transforms; preprocess_observation(train=False)",
        "sample_manifest": str(manifest_path.resolve()),
        "metrics": aggregate,
        "aggregation": "Sum of squared error divided by all valid prefix elements",
        "mean_prefix_baseline": "Per-observation valid-token mean, broadcast to every position",
        "zero_token_ablation": "Replace the single RL token with zeros; retain identical teacher forcing",
        "limitations": [
            "The official decoder uses shifted ground-truth prefix embeddings with causal teacher forcing.",
            "Reconstruction MSE is not an autoregressive generation or downstream policy-success metric.",
            "A zero-token loss increase indicates decoder use of z, not proof of control usefulness.",
            "Held-out episodes are excluded from RLT training; the base SFT checkpoint may have seen them.",
        ],
        "rlt_image_only": bool(cfg.actor.model.openpi.rlt_image_only),
        "tokens": {
            "shape": [len(rows), rows[0]["z_dim"]],
            "dtype": "float32",
            "finite": True,
        },
        "elapsed_seconds": time.monotonic() - start,
        "samples": rows,
    }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    logger.info("Evaluation complete: %s; metrics=%s", args.output, aggregate)


if __name__ == "__main__":
    main()
