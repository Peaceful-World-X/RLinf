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

"""Evaluate a saved RLT checkpoint on a fixed subset of held-out G1 episodes."""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from openpi.training import data_loader
from torch.utils.data import Subset

from rlinf.data.datasets.openpi.lerobot_v3_dataset import LeRobotV3Dataset
from rlinf.models.embodiment.openpi import get_model
from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
from rlinf.models.embodiment.openpi.modules import model as model_utils


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="g1_rlt_stage1_sft_openpi_pi05")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--device", default=None)
    parser.add_argument("--merge-only", action="store_true")
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    if args.merge_only:
        parts = sorted(args.output.parent.glob(args.output.stem + ".rank*.json"))
        if not parts and args.output.is_file():
            result = json.loads(args.output.read_text())
            if (
                result.get("world_size") == 1
                and Path(result["checkpoint"]).resolve() == args.checkpoint.resolve()
            ):
                return
        shards = [json.loads(path.read_text()) for path in parts]
        if not shards or len(shards) != shards[0]["world_size"]:
            raise ValueError("Missing evaluation ranks")
        identities = {(s["checkpoint"], s["split_path"]) for s in shards}
        if len(identities) != 1:
            raise ValueError("Mixed checkpoint/split evaluation shards")
        rows = sorted(
            [row for shard in shards for row in shard["per_sample"]],
            key=lambda r: r["validation_local_index"],
        )
        if len({r["validation_local_index"] for r in rows}) != len(rows):
            raise ValueError("Duplicate validation frames")
        counts = np.array([s["valid_elements"] for s in shards])
        result = {
            **shards[0],
            "world_size": len(shards),
            "samples": len(rows),
            "per_sample": rows,
            "valid_elements": int(counts.sum()),
        }
        for key in ("rlt_mse", "per_sample_mean_feature_mse", "zero_token_mse"):
            result[key] = float(np.average([s[key] for s in shards], weights=counts))
        result["vla_loss"] = float(np.mean([r["vla_loss"] for r in rows]))
        result["relative_improvement_over_mean"] = (
            1 - result["rlt_mse"] / result["per_sample_mean_feature_mse"]
        )
        result["zero_token_relative_degradation"] = (
            result["zero_token_mse"] / result["rlt_mse"] - 1
        )
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        return
    args.device = args.device or f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}"
    torch.set_num_threads(4)
    repo = Path(__file__).resolve().parents[1]
    with initialize_config_dir(
        config_dir=str(repo / "examples/sft/config"), version_base="1.1"
    ):
        cfg = compose(config_name=args.config)
    split = json.loads(Path(cfg.data.episode_split_path).read_text())
    assert not set(split["train"]) & set(split["validation"]), "Episode splits overlap"
    dataset = LeRobotV3Dataset(
        cfg.data.train_data_paths,
        cfg.actor.model.openpi.action_horizon,
        episode_ids=split["validation"],
    )
    indices = np.unique(
        np.linspace(0, len(dataset) - 1, min(args.samples, len(dataset)), dtype=int)
    )
    indices = indices[rank::world]
    config = get_openpi_config(
        cfg.actor.model.openpi.config_name,
        model_path=cfg.actor.model.model_path,
        data_kwargs=OmegaConf.to_container(cfg.actor.model.openpi_data, resolve=True),
    )
    data_config = config.data.create(config.assets_dirs, config.model)
    dataset = data_loader.transform_dataset(
        Subset(dataset, indices.tolist()), data_config
    )
    loader = data_loader.DataLoaderImpl(
        data_config,
        data_loader.TorchDataLoader(
            dataset,
            local_batch_size=1,
            num_batches=len(indices),
            num_workers=0,
            framework="pytorch",
        ),
    )
    cfg.actor.model.model_path = str(args.checkpoint.resolve())
    # Match the official FSDP forward precision; the RLT module stays FP32.
    cfg.actor.model.precision = "bf16"
    cfg.actor.model.openpi.torch_compile = False
    torch.manual_seed(42)
    network = get_model(cfg.actor.model).to(args.device).eval().requires_grad_(False)
    network.rlt_module.float()
    rows = []
    squared, baseline, ablated, count = 0.0, 0.0, 0.0, 0.0
    with torch.inference_mode():
        for index, (observation, actions) in zip(indices, loader, strict=True):
            torch.manual_seed(42 + int(index))
            np.random.seed(42 + int(index))
            random.seed(42 + int(index))
            vla = network.sft_forward((observation, actions))["vla_loss"]
            if not torch.isfinite(vla):
                raise ValueError("Nonfinite VLA validation loss")
            observation = network._observation_to_device(observation)
            observation = model_utils.preprocess_observation(observation, train=False)
            observation = model_utils._observation_to_dtype(
                observation, network.embed_dtype
            )
            prefix, mask, _ = network.build_prefix_cache(observation)
            prefix, mask = network._select_rlt_prefix_embeddings(
                prefix, mask, observation.tokenized_prompt
            )
            prefix = prefix.float()
            reconstructed, token = network.rlt_module.reconstruct(prefix, mask)
            zero_reconstruction = network.rlt_module.decode(
                torch.zeros_like(token), prefix, mask
            )
            assert token.shape == (1, 1, 2048) and torch.isfinite(token).all()
            valid = mask[..., None].float()
            denom = valid.sum() * prefix.shape[-1]
            mean = (prefix * valid).sum(dim=1, keepdim=True) / valid.sum(
                dim=1, keepdim=True
            )
            errors = [
                (((value.float() - prefix) ** 2) * valid).sum()
                for value in [reconstructed, mean, zero_reconstruction]
            ]
            assert all(torch.isfinite(value) for value in errors)
            squared += float(errors[0])
            baseline += float(errors[1])
            ablated += float(errors[2])
            count += float(denom)
            rows.append(
                {
                    "validation_local_index": int(index),
                    "mse": float(errors[0] / denom),
                    "vla_loss": float(vla),
                }
            )
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "samples": len(rows),
        "world_size": world,
        "valid_elements": int(count),
        "vla_loss": float(np.mean([r["vla_loss"] for r in rows])),
        "rlt_mse": squared / count,
        "per_sample_mean_feature_mse": baseline / count,
        "zero_token_mse": ablated / count,
        "relative_improvement_over_mean": 1.0 - squared / baseline,
        "zero_token_relative_degradation": ablated / squared - 1.0,
        "token_shape": [1, 2048],
        "finite": True,
        "preprocessing": "inference; no random image augmentation",
        "split_path": cfg.data.episode_split_path,
        "scope": "RLT holdout only; base VLA may have seen these episodes. No robot-success claim.",
        "per_sample": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output = (
        args.output
        if world == 1
        else args.output.with_name(args.output.stem + f".rank{rank}.json")
    )
    output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
