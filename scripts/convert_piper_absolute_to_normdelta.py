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

"""Convert Piper absolute joint action chunks to OpenPI normalized-delta chunks."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch

DELTA_MASK = np.array([True] * 6 + [False] + [True] * 6 + [False], dtype=bool)


def load_action_stats(
    norm_stats_path: Path, std_floor: float
) -> tuple[torch.Tensor, torch.Tensor]:
    payload = json.loads(norm_stats_path.read_text())
    stats = payload.get("norm_stats", payload)["actions"]
    mean = torch.tensor(stats["mean"][:14], dtype=torch.float32)
    std = torch.tensor(stats["std"][:14], dtype=torch.float32)
    std = torch.where(std.abs() < std_floor, torch.ones_like(std), std)
    return mean, std


def absolute_to_delta(actions: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
    """OpenPI DeltaActions equivalent for [T, B, H*14] absolute chunks."""
    shape = actions.shape
    actions = actions.float().reshape(shape[0], shape[1], -1, 14).clone()
    states = states.float().reshape(shape[0], shape[1], 14)
    mask = torch.tensor(DELTA_MASK, dtype=torch.bool, device=actions.device)
    actions[..., mask] = actions[..., mask] - states[:, :, None, :][..., mask]
    return actions.reshape(shape)


def delta_to_normalized(
    delta_actions: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
    shape = delta_actions.shape
    actions = delta_actions.float().reshape(shape[0], shape[1], -1, 14)
    mean = mean.to(actions.device).reshape(1, 1, 1, 14)
    std = std.to(actions.device).reshape(1, 1, 1, 14)
    return ((actions - mean) / (std + 1e-6)).reshape(shape)


def normalized_to_absolute(
    norm_actions: torch.Tensor,
    states: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    shape = norm_actions.shape
    actions = norm_actions.float().reshape(shape[0], shape[1], -1, 14)
    mean = mean.to(actions.device).reshape(1, 1, 1, 14)
    std = std.to(actions.device).reshape(1, 1, 1, 14)
    delta = actions * (std + 1e-6) + mean
    states = states.float().reshape(shape[0], shape[1], 14)
    mask = torch.tensor(DELTA_MASK, dtype=torch.bool, device=actions.device)
    absolute = delta.clone()
    absolute[..., mask] = absolute[..., mask] + states[:, :, None, :][..., mask]
    return absolute.reshape(shape)


def convert_absolute_actions(
    actions: torch.Tensor,
    states: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    delta_actions = absolute_to_delta(actions.float(), states)
    norm_actions = delta_to_normalized(delta_actions, mean, std)
    return (
        norm_actions,
        delta_actions,
        normalized_to_absolute(norm_actions, states, mean, std),
    )


def convert_file(
    path: Path, out_path: Path, mean: torch.Tensor, std: torch.Tensor
) -> dict:
    traj = torch.load(path, map_location="cpu")
    absolute_actions = traj["actions"].float()
    states = traj["curr_obs"]["states"].float()
    norm_actions, delta_actions, recon_absolute = convert_absolute_actions(
        absolute_actions, states, mean, std
    )
    recon_err = (recon_absolute - absolute_actions).abs()

    traj["actions"] = norm_actions.to(dtype=traj["actions"].dtype)
    forward_inputs = traj.setdefault("forward_inputs", {})
    forward_inputs["env_action_absolute"] = absolute_actions
    forward_inputs["action_delta"] = delta_actions
    forward_inputs["action"] = norm_actions

    # Old real-world files only saved the behavior action. If future files also
    # contain the raw VLA action, keep it distinct and convert it too.
    original_model_action = forward_inputs.get("model_action", None)
    original_ref_action = forward_inputs.get("ref_action", None)
    if torch.is_tensor(original_model_action):
        model_norm, model_delta, model_abs = convert_absolute_actions(
            original_model_action.float().reshape_as(absolute_actions),
            states,
            mean,
            std,
        )
        forward_inputs["model_action"] = model_norm
        forward_inputs["model_action_delta"] = model_delta
        forward_inputs["model_action_absolute"] = model_abs
    else:
        forward_inputs["model_action"] = norm_actions

    if torch.is_tensor(original_ref_action):
        ref_norm, ref_delta, ref_abs = convert_absolute_actions(
            original_ref_action.float().reshape_as(absolute_actions), states, mean, std
        )
        forward_inputs["ref_action"] = ref_norm
        forward_inputs["ref_action_delta"] = ref_delta
        forward_inputs["ref_action_absolute"] = ref_abs
    else:
        forward_inputs["ref_action"] = forward_inputs["model_action"]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(traj, out_path)
    return {
        "file": path.name,
        "length": int(absolute_actions.shape[0]),
        "absolute_min": float(absolute_actions.min()),
        "absolute_max": float(absolute_actions.max()),
        "delta_min": float(delta_actions.min()),
        "delta_max": float(delta_actions.max()),
        "norm_min": float(norm_actions.min()),
        "norm_max": float(norm_actions.max()),
        "norm_mean": float(norm_actions.mean()),
        "norm_std": float(norm_actions.std()),
        "roundtrip_abs_mean_err": float(recon_err.mean()),
        "roundtrip_abs_max_err": float(recon_err.max()),
        "max_reward": float(traj["rewards"].float().max())
        if "rewards" in traj
        else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--norm-stats", required=True)
    parser.add_argument("--std-floor", type=float, default=1e-6)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_dir} exists; pass --overwrite")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("metadata.json", "trajectory_index.json", "reward_fix_summary.json"):
        src = input_dir / name
        if src.exists():
            shutil.copyfile(src, output_dir / name)

    mean, std = load_action_stats(Path(args.norm_stats), args.std_floor)
    records = []
    for path in sorted(input_dir.glob("trajectory_*.pt")):
        records.append(convert_file(path, output_dir / path.name, mean, std))

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "norm_stats": str(args.norm_stats),
        "std_floor": float(args.std_floor),
        "num_trajectories": len(records),
        "num_success": int(sum(record["max_reward"] > 0 for record in records)),
        "num_failure": int(sum(record["max_reward"] <= 0 for record in records)),
        "norm_min": float(min(record["norm_min"] for record in records)),
        "norm_max": float(max(record["norm_max"] for record in records)),
        "roundtrip_abs_max_err": float(
            max(record["roundtrip_abs_max_err"] for record in records)
        ),
        "records": records,
    }
    (output_dir / "conversion_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
