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

"""Encode G1 intervention replay with a trained RLinf Stage 1 checkpoint."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict

from rlinf.algorithms.rlt.transition import apply_rlt_interventions
from rlinf.data.datasets.openpi.lerobot_v3_dataset import LeRobotV3Dataset
from rlinf.data.schema.embodied_types import Trajectory
from rlinf.data.storage.replay import TrajectoryReplayBuffer
from rlinf.models.embodiment.openpi import get_model
from rlinf.models.embodiment.openpi.checkpoint import resolve_full_weights
from rlinf.models.embodiment.openpi.dataconfig.g1_dataconfig import G1ActionCoordinates
from rlinf.utils.logging import get_logger

logger = get_logger()
CAMERAS = (
    "observation.images.cam_left_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)


def sha256(path: Path) -> str:
    """Hash an artifact without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: Path, checkpoint: Path):
    """Compose the training config and select official inference behavior."""
    path = path.resolve()
    os.environ.setdefault("EMBODIED_PATH", str(path.parent.parent))
    with initialize_config_dir(version_base="1.1", config_dir=str(path.parent)):
        cfg = compose(config_name=path.stem)
    model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.actor.model, resolve=True))
    with open_dict(model_cfg):
        model_cfg.model_path = str(checkpoint.resolve())
        model_cfg.precision = "bf16"
        model_cfg.num_action_chunks = model_cfg.openpi.action_horizon
        model_cfg.openpi.task = "eval"
        model_cfg.openpi.action_chunk = model_cfg.openpi.action_horizon
        model_cfg.openpi.torch_compile = False
    if model_cfg.openpi.config_name != "pi05_g1_dualgripper":
        raise ValueError("This camera/action adapter requires pi05_g1_dualgripper")
    if not model_cfg.openpi.use_rlt or model_cfg.action_dim != 16:
        raise ValueError("Expected an RLT-enabled G1 model with 16 action dimensions")
    if resolve_full_weights(checkpoint) is None:
        raise ValueError(
            "Expected trained Stage 1 full_weights.pt, not base safetensors"
        )
    return model_cfg


def load_raw_replay(path: Path) -> list[Trajectory]:
    """Read every source trajectory through the official replay serializer."""
    buffer = TrajectoryReplayBuffer(
        enable_cache=True, cache_size=5, sample_window_size=5, auto_save=False
    )
    try:
        buffer.load_checkpoint(str(path))
        return [
            buffer._load_trajectory(
                index, buffer._trajectory_index[index]["model_weights_id"]
            )
            for index in buffer._trajectory_id_list
        ]
    finally:
        buffer.close()


def source_indices(trajectories: list[Trajectory], horizon: int = 1) -> list[int]:
    """Validate source references and return the unique observation frames."""
    indices = set()
    for trajectory in trajectories:
        current = trajectory.curr_obs["source_index"].reshape(-1)
        following = trajectory.next_obs["source_index"].reshape(-1)
        if not torch.equal(current + horizon, following):
            raise ValueError(
                "Next observations must be exactly one action horizon later"
            )
        expected = (len(current), 1, horizon * 16)
        if trajectory.actions.reshape(len(current), 1, -1).shape != expected:
            raise ValueError(
                f"Expected actions {expected}, got {trajectory.actions.shape}"
            )
        if trajectory.intervene_flags.shape != trajectory.actions.shape:
            raise ValueError("Expected per-action intervention flags matching actions")
        flags = trajectory.intervene_flags.bool()
        if not torch.equal(flags, flags[..., :1].expand_as(flags)):
            raise ValueError(
                "Expected one intervention flag expanded over all 16 joints"
            )
        for name in ("rewards", "dones", "terminations", "truncations"):
            value = getattr(trajectory, name)
            if value.reshape(len(current), 1, -1).shape != (len(current), 1, horizon):
                raise ValueError(f"Unexpected H{horizon} {name} shape")
            if not torch.isfinite(value).all():
                raise ValueError(f"Nonfinite source {name}")
        indices.update(current.tolist())
        indices.update(following.tolist())
    if not indices:
        raise ValueError("Source replay contains no transitions")
    return sorted(indices)


def chunk_trajectories(trajectories, horizon: int, stride: int = 2):
    """Keep full windows, real t+H states and the terminal window without padding."""
    source_indices(trajectories)
    chunked, audit = [], []
    for source in trajectories:
        length = len(source.actions)
        current = source.curr_obs["source_index"].reshape(-1)
        if length < horizon or not torch.equal(current[1:], current[:-1] + 1):
            raise ValueError("Insertion interval must contain contiguous full chunks")
        if not bool(source.terminations[-1].all()) or source.dones[:-1].any():
            raise ValueError("Expected a single observed termination at interval end")
        starts = torch.tensor(
            sorted(set(range(0, length - horizon + 1, stride)) | {length - horizon})
        )
        slots = starts[:, None] + torch.arange(horizon)[None]
        ends = starts + horizon - 1

        def pack(value):
            if isinstance(value, dict):
                return {key: pack(item) for key, item in value.items()}
            if torch.is_tensor(value):
                if value.shape[:2] != (length, 1):
                    raise ValueError(f"Unexpected sequence shape {value.shape}")
                return value[slots].transpose(1, 2).contiguous()
            return value

        fields = {
            field.name: pack(getattr(source, field.name))
            for field in dataclasses.fields(Trajectory)
            if field.name not in ("curr_obs", "next_obs")
        }
        fields.update(
            max_episode_length=len(starts),
            curr_obs={k: v[starts] for k, v in source.curr_obs.items()},
            next_obs={k: v[ends] for k, v in source.next_obs.items()},
        )
        for key in ("rewards", "dones", "terminations", "truncations"):
            fields[key] = fields[key].squeeze(-1)
        chunked.append(Trajectory(**fields))
        audit.append(
            {
                "episode_index": episode_id(source),
                "source_steps": length,
                "chunks": len(starts),
                "stride": stride,
                "starts": current[starts].tolist(),
                "terminal_next_index": int(source.next_obs["source_index"][-1].item()),
            }
        )
    source_indices(chunked, horizon)
    return chunked, audit


def validate_source_episodes(
    trajectories: list[Trajectory], dataset: LeRobotV3Dataset
) -> None:
    """Reject transitions that cross recording boundaries."""
    ends = np.asarray(
        [episode["dataset_to_index"] for episode in dataset.episodes.values()]
    )
    starts = np.asarray(
        [episode["dataset_from_index"] for episode in dataset.episodes.values()]
    )
    for trajectory in trajectories:
        current = trajectory.curr_obs["source_index"].reshape(-1).numpy()
        following = trajectory.next_obs["source_index"].reshape(-1).numpy()
        positions = np.searchsorted(ends, current, side="right")
        if np.any(positions >= len(ends)) or np.any(current < 0):
            raise ValueError("Source replay references an unavailable dataset row")
        if np.any(current < starts[positions]) or np.any(following >= ends[positions]):
            raise ValueError("Source transition crosses an episode boundary")


def sample_to_env(sample: dict) -> dict:
    """Use the existing AlohaInputs env contract with G1's three real cameras."""
    images = [np.moveaxis(sample[key], 0, -1) for key in CAMERAS]
    return {
        "main_images": images[0][None],
        "wrist_images": np.stack(images[1:])[None],
        "states": np.asarray(sample["observation.state"], dtype=np.float32)[None],
        "task_descriptions": [sample["prompt"]],
    }


def tensor_range(value: torch.Tensor) -> dict:
    """Report range and actor-output support violations without clipping data."""
    if not torch.isfinite(value).all():
        raise ValueError("Nonfinite values in replay features or actions")
    return {
        "min": float(value.min()) if value.numel() else None,
        "max": float(value.max()) if value.numel() else None,
        "outside_unit_interval_values": int((value.abs() > 1).sum()),
    }


def encode_trajectories(
    trajectories, features, coordinates, model_id, horizon, *, unit_bounded=True
):
    """Use official intervention replacement and preserve pure reference for validation."""
    encoded = []
    for source in trajectories:
        observations = []
        for raw_obs in (source.curr_obs, source.next_obs):
            rows = [features[int(i)] for i in raw_obs["source_index"].reshape(-1)]
            obs = {
                **raw_obs,
                **{key: torch.stack([row[key] for row in rows]) for key in rows[0]},
            }
            if not torch.equal(obs["proprio"], raw_obs["states"].float()):
                raise ValueError("Extracted proprio differs from recorded state")
            observations.append(obs)
        current, following = observations
        for obs in observations:
            obs["ref_chunk"] = obs["ref_chunk"][..., :horizon, :].clone()
        original_reference = current["ref_chunk"].clone()
        actions = coordinates.normalize(source.actions, current["states"])
        if not torch.isfinite(actions).all() or (
            unit_bounded and (actions.abs() > 1.00001).any()
        ):
            raise ValueError("Executed actions must be finite and inside actor bounds")
        if not torch.allclose(
            coordinates.denormalize(actions, current["states"]),
            source.actions,
            atol=1e-5,
            rtol=0,
        ):
            raise ValueError("Physical action round trip failed")
        for obs in observations:
            obs["ref_chunk"] = coordinates.normalize(obs["ref_chunk"], obs["states"])
            obs["proprio"] = coordinates.proprio(obs["states"])
        pure_reference = current["ref_chunk"].clone()
        human = source.intervene_flags.any(dim=-1)
        apply_rlt_interventions(current, actions, human)
        if not torch.equal(current["ref_chunk"][human], actions[human]):
            raise ValueError("Human reference replacement failed")
        encoded.append(
            dataclasses.replace(
                source,
                model_weights_id=model_id,
                actions=actions,
                curr_obs=current,
                next_obs=following,
                forward_inputs={
                    **source.forward_inputs,
                    "trajectory_success": trajectory_success_labels(source),
                    "source_actions_physical": source.actions.clone(),
                    "stage1_ref_chunk_physical": original_reference,
                    "pure_ref_chunk": pure_reference,
                },
            )
        )
    return encoded


def trajectory_success_labels(trajectory: Trajectory) -> torch.Tensor:
    """Label G1 insertion attempts: terminal reward 1 is success, otherwise 0."""
    rewards = trajectory.rewards
    if not torch.all((rewards == 0) | (rewards == 1)):
        raise ValueError("G1 outcomes require sparse binary success rewards")
    successful = rewards.reshape(rewards.shape[0], rewards.shape[1], -1).any(dim=(0, 2))
    return successful[None, :, None].expand(rewards.shape[0], -1, -1).clone()


def assert_equal(left, right) -> None:
    """Check replay serialization preserves all tensors and provenance fields."""
    if torch.is_tensor(left):
        if not torch.equal(left, right):
            raise ValueError("Replay reload changed a tensor")
    elif isinstance(left, dict):
        if left.keys() != right.keys():
            raise ValueError("Replay reload changed observation/provenance keys")
        for key in left:
            assert_equal(left[key], right[key])
    elif left != right:
        raise ValueError("Replay reload changed a scalar field")


def episode_id(trajectory: Trajectory) -> int:
    """Return the source episode, rejecting mixed-episode replay records."""
    episodes = trajectory.forward_inputs["episode_index"].unique()
    if len(episodes) != 1:
        raise ValueError("Each saved trajectory must belong to one source episode")
    return int(episodes.item())


def partition_replay(
    trajectories: list[Trajectory], holdout: list[int]
) -> dict[str, list[Trajectory]]:
    """Partition whole source episodes without resampling or changing transitions."""
    available = {episode_id(t) for t in trajectories}
    selected = set(holdout)
    if not selected <= available or selected == available:
        raise ValueError(
            f"Holdout episodes {sorted(selected)} must be a proper subset of {sorted(available)}"
        )
    groups = {"train": [t for t in trajectories if episode_id(t) not in selected]}
    if selected:
        groups["validation"] = [t for t in trajectories if episode_id(t) in selected]
    return groups


def save_replay(trajectories: list[Trajectory], output: Path, seed: int) -> None:
    """Save with the official writer and verify an exact full reload."""
    buffer = TrajectoryReplayBuffer(
        seed=seed,
        enable_cache=True,
        cache_size=len(trajectories),
        sample_window_size=len(trajectories),
        auto_save=False,
    )
    try:
        buffer.add_trajectories(trajectories)
        buffer.save_checkpoint(str(output))
    finally:
        buffer.close()
    reloaded = load_raw_replay(output)
    if len(trajectories) != len(reloaded):
        raise ValueError("Replay reload changed trajectory count")
    for expected, actual in zip(trajectories, reloaded, strict=True):
        for field in dataclasses.fields(Trajectory):
            assert_equal(getattr(expected, field.name), getattr(actual, field.name))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "stage2-config", "checkpoint", "source-cache", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--holdout-episodes", type=int, nargs="+", default=[4])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument(
        "--feature-horizon",
        type=int,
        default=None,
        help="Reference steps retained in the shared feature cache",
    )
    parser.add_argument(
        "--features-dir", type=Path, help="Existing feature shards to finalize"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    model_cfg = load_config(args.config, args.checkpoint)
    with initialize_config_dir(
        version_base="1.1", config_dir=str(args.stage2_config.resolve().parent)
    ):
        stage2 = compose(config_name=args.stage2_config.stem)
    horizon, ref_horizon = (
        int(stage2.actor.model.num_action_chunks),
        int(stage2.actor.model.ref_num_action_chunks),
    )
    vla_horizon = int(model_cfg.openpi.action_horizon)
    cache_horizon = args.feature_horizon or ref_horizon
    if not (0 < horizon == ref_horizon <= cache_horizon <= vla_horizon == 50):
        raise ValueError(
            "Expected matching actor/reference horizons within the VLA H50 cache"
        )
    if args.features_dir is not None and not args.finalize_only:
        parser.error("--features-dir is only used with --finalize-only")
    raw = load_raw_replay(args.source_cache / "raw_replay")
    partition_replay(raw, args.holdout_episodes)
    dataset = LeRobotV3Dataset(args.source_cache / "dataset", action_horizon=1)
    validate_source_episodes(raw, dataset)
    trajectories, audit = chunk_trajectories(raw, horizon)
    indices = source_indices(trajectories, horizon)
    coordinates = G1ActionCoordinates(model_cfg.openpi_data.norm_stats_path)
    weights = resolve_full_weights(args.checkpoint)
    fingerprint = {
        "checkpoint": str(weights.resolve()),
        "size": weights.stat().st_size,
        "mtime_ns": weights.stat().st_mtime_ns,
        "source_sha256": sha256(args.source_cache / "source_manifest.json"),
        "norm_sha256": sha256(Path(model_cfg.openpi_data.norm_stats_path)),
        "seed": args.seed,
        "feature_horizon": cache_horizon,
    }
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    if not args.finalize_only:
        device = args.device or f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}"
        torch.manual_seed(args.seed)
        torch.set_num_threads(4)
        model = get_model(model_cfg).to(device).eval().requires_grad_(False)
        model.rlt_module.float()
        features = {}
        selected = indices[rank::world]
        started = time.monotonic()
        with torch.inference_mode():
            for position, index in enumerate(selected):
                sample = dataset[index]
                if int(sample["index"]) != index:
                    raise ValueError("Source global row differs from dataset position")
                torch.manual_seed(args.seed + index)
                result = model.extract_rlt_obs(sample_to_env(sample))
                if result["z_rl"].shape != (1, 2048) or result["ref_chunk"].shape != (
                    1,
                    50,
                    16,
                ):
                    raise ValueError("Invalid Stage 1 feature shape")
                result["ref_chunk"] = result["ref_chunk"][:, :cache_horizon]
                if not all(torch.isfinite(v).all() for v in result.values()):
                    raise ValueError("Nonfinite feature or model reference")
                features[index] = {k: v.float().cpu() for k, v in result.items()}
                if position % 100 == 0 or position + 1 == len(selected):
                    logger.info(
                        "Rank %d encoded %d/%d frames in %.1fs",
                        rank,
                        position + 1,
                        len(selected),
                        time.monotonic() - started,
                    )
        torch.save(
            {"fingerprint": fingerprint, "world_size": world, "features": features},
            args.output / f"features_rank{rank}.pt",
        )
        return

    features = {}
    shards = sorted((args.features_dir or args.output).glob("features_rank*.pt"))
    for path in shards:
        shard = torch.load(path, map_location="cpu", weights_only=True)
        if shard["fingerprint"] != fingerprint or len(shards) != shard["world_size"]:
            raise ValueError(
                "Feature shards have inconsistent checkpoint or source identity"
            )
        if features.keys() & shard["features"].keys():
            raise ValueError("Duplicate feature frame across ranks")
        features.update(shard["features"])
    if not set(indices).issubset(features):
        raise ValueError(
            "Feature shards do not cover every current and horizon-next observation"
        )
    checkpoint_hash = sha256(weights)
    encoded = encode_trajectories(
        trajectories,
        features,
        coordinates,
        checkpoint_hash[:16],
        horizon,
        unit_bounded=stage2.actor.model.get("g1_action_mode", "state_delta")
        == "state_delta",
    )
    groups = partition_replay(encoded, args.holdout_episodes)
    for role, group in groups.items():
        destination = args.output / role
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite replay {destination}")
        save_replay(group, destination, args.seed)
    validation = groups["validation"]
    evaluation = {
        "curr_obs": {
            k: torch.cat(
                [
                    t.forward_inputs["pure_ref_chunk"]
                    if k == "ref_chunk"
                    else t.curr_obs[k]
                    for t in validation
                ]
            )[:, 0]
            for k in ("z_rl", "proprio", "ref_chunk")
        },
        "actions": torch.cat([t.actions for t in validation])[:, 0],
        "intervene_flags": torch.cat([t.intervene_flags for t in validation])[:, 0],
        "raw_states": torch.cat([t.curr_obs["states"] for t in validation])[:, 0],
        "action_scale": coordinates.span / 2,
    }
    torch.save(evaluation, args.output / "validation.pt")
    manifest = {
        **fingerprint,
        "checkpoint_sha256": checkpoint_hash,
        "reload_exact": True,
        "chunking": audit,
        "horizon": horizon,
        "ref_horizon": ref_horizon,
        "unique_frames": len(indices),
        "physical_delta_joints": 14,
        "gripper_range": [0, 100],
        "validation_reference": "pure model prediction",
        "split": {
            role: {
                "episodes": [episode_id(t) for t in group],
                "chunks": sum(len(t.actions) for t in group),
            }
            for role, group in groups.items()
        },
    }
    (args.output / "feature_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    OmegaConf.save(model_cfg, args.output / "feature_config.yaml")
    logger.info("Verified feature cache: %s", manifest["split"])


if __name__ == "__main__":
    main()
