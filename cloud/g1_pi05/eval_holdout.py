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

"""Compare 20k and the final CFG checkpoint on fixed held-out G1 action chunks."""

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from assets_io import save_json
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from omegaconf import OmegaConf
from openpi import transforms
from torch.utils.data import default_collate

from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
from rlinf.models.embodiment.openpi_cfg import get_model
from rlinf.models.embodiment.openpi_cfg.openpi_cfg_action_model import Observation
from rlinf.workers.sft.fsdp_cfg_worker import FSDPCfgWorker


def held_out_windows(root, model_cfg, horizon):
    splits = json.loads((root / "episode_split.json").read_text())["sources"]
    specs = json.loads((root / "datasets.json").read_text())
    setup = get_openpi_config(
        model_cfg.openpi.config_name,
        model_path=model_cfg.model_path,
        repo_id=str(root / specs[0]["path"]),
    )
    data = setup.data.create(setup.assets_dirs, setup.model)
    encode = transforms.compose(
        [
            *data.repack_transforms.inputs,
            *data.data_transforms.inputs,
            transforms.Normalize(data.norm_stats, use_quantiles=data.use_quantile_norm),
            *FSDPCfgWorker._build_model_transforms(data),
        ]
    )
    windows = []
    for source, spec in enumerate(specs):
        path = root / spec["path"]
        split = splits[path.name]
        ids = sorted(split["val"])
        if (
            not ids
            or set(ids) & set(spec["episodes"])
            or spec["episodes"] != split["train"]
        ):
            raise ValueError(f"Invalid policy holdout for {path.name}")
        ds = LeRobotDataset(str(path), root=path, episodes=ids)
        metadata = {
            e["episode_index"]: e
            for e in map(
                json.loads, (path / "meta/episodes.jsonl").read_text().splitlines()
            )
        }
        info = json.loads((path / "meta/info.json").read_text())
        start = 0
        for ep in ids:
            length = metadata[ep]["length"]
            parquet = path / info["data_path"].format(
                episode_chunk=ep // info["chunks_size"], episode_index=ep
            )
            actions = np.asarray(
                pq.read_table(parquet, columns=["action"]).column(0).to_pylist(),
                dtype=np.float32,
            )
            if actions.shape != (length, 16):
                raise ValueError(f"Invalid actions in {parquet}")
            last = max(0, length - horizon)
            for frame in sorted({0, last // 2, last}):
                sample = ds[start + frame]
                if (int(sample["episode_index"]), int(sample["frame_index"])) != (
                    ep,
                    frame,
                ):
                    raise ValueError(
                        f"Holdout row misaligned at {path.name}/{ep}/{frame}"
                    )
                target = actions[frame : frame + horizon].copy()
                sample["prompt"] = sample["task"]
                sample["action"] = np.pad(
                    target, ((0, horizon - len(target)), (0, 0)), mode="edge"
                )
                sample = {
                    k: v.numpy() if isinstance(v, torch.Tensor) else v
                    for k, v in sample.items()
                }
                windows.append((path.name, ep, frame, source, encode(sample), target))
            start += length
        if start != len(ds):
            raise ValueError(f"Unexpected number of held-out rows in {path.name}")
    return windows


@torch.inference_mode()
def evaluate(model, windows, horizon):
    sums = {}
    for name, ep, frame, source, encoded, target in windows:
        observation = Observation.from_dict(
            model.precision_processor(default_collate([encoded]))
        )
        # Identical noise for this frame in the base and final checkpoint.
        rng = torch.Generator().manual_seed(4000 + (source << 40) + (ep << 20) + frame)
        noise = torch.randn((1, horizon, model.config.action_dim), generator=rng).to(
            "cuda"
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            predicted = model.sample_actions(observation, noise=noise)["actions"]
        decoded = model.output_transform(
            {"actions": predicted.float(), "state": observation.state.float()}
        )["actions"][0]
        if tuple(decoded.shape) != (horizon, 16):
            raise ValueError(f"Unexpected G1 action shape {tuple(decoded.shape)}")
        error = np.square(decoded.numpy()[: len(target)].astype(np.float64) - target)
        if not np.isfinite(error).all():
            raise ValueError(f"Nonfinite predicted actions at {name}/{ep}/{frame}")
        rec = sums.setdefault(name, {}).setdefault(ep, [0.0, 0.0, 0, 0])
        rec[0] += error[:, :14].sum()
        rec[1] += error[:, 14:].sum()
        rec[2] += len(target)
        rec[3] += 1
    sources = {}
    for name, episodes in sums.items():
        per_episode = [
            {
                "episode_index": ep,
                "windows": count,
                "joint_mse": joint / (steps * 14),
                "gripper_mse": gripper / (steps * 2),
            }
            for ep, (joint, gripper, steps, count) in episodes.items()
        ]
        sources[name] = {
            "episodes": len(per_episode),
            "windows": sum(e["windows"] for e in per_episode),
            "joint_mse": float(np.mean([e["joint_mse"] for e in per_episode])),
            "gripper_mse": float(np.mean([e["gripper_mse"] for e in per_episode])),
            "per_episode": per_episode,
        }
    return sources


def main(job, step):
    cfg = OmegaConf.load(job / "configs/policy.yaml").actor.model
    horizon = int(cfg.num_action_chunks)
    OmegaConf.update(cfg, "openpi.action_chunk", horizon, force_add=True)
    windows = held_out_windows(job / "input", cfg, horizon)
    model = get_model(cfg).eval().to("cuda")
    trained_config = model.config
    model.config = dataclasses.replace(trained_config, guidance_type="no_guide")
    result = {
        "checkpoint_step": step,
        "action_horizon": horizon,
        "denoising_steps": model.config.num_steps,
        "anchors": "first, middle and last complete action chunk; padded targets excluded",
        "aggregation": "mean per episode in recorded action coordinates",
        "base_20k": evaluate(model, windows, horizon),
    }
    checkpoint = job / "results/policy/checkpoints" / f"global_step_{step}"
    if (checkpoint / ".complete").read_text().strip() != str(step):
        raise ValueError(f"Incomplete policy checkpoint {step}")
    weights = torch.load(
        checkpoint / "actor/model_state_dict/full_weights.pt",
        weights_only=True,
        mmap=True,
        map_location="cpu",
    )
    model.load_state_dict(weights, strict=True)
    model.config = trained_config
    result["policy"] = evaluate(model, windows, horizon)
    save_json(job / "holdout-actions.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("job", type=Path)
    parser.add_argument("step", type=int)
    args = parser.parse_args()
    main(args.job.resolve(), args.step)
