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


def feature_contract_sha256(manifest: dict) -> str:
    """Identify the frozen VLA and the actor input coordinate transform."""
    contract = {
        key: manifest.get(key)
        for key in ("checkpoint_sha256", "action_transform", "z_center")
    }
    return hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()


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


def chunk_trajectories(
    trajectories: list[Trajectory], horizon: int, stride: int = 2
) -> tuple[list[Trajectory], list[dict]]:
    """Keep overlapping full windows and the observed terminal without padding."""
    if not 1 <= stride <= horizon:
        raise ValueError("Expected 1 <= stride <= action horizon")
    source_indices(trajectories)
    chunked, audit = [], []
    for source in trajectories:
        length = len(source.actions)
        current = source.curr_obs["source_index"].reshape(-1)
        if length < horizon or not torch.equal(current[1:], current[:-1] + 1):
            raise ValueError("Each source interval must contain contiguous full chunks")
        if not bool(source.terminations[-1].all()) or source.dones[:-1].any():
            raise ValueError("Expected one observed termination at the interval end")
        starts = torch.tensor(
            sorted(set(range(0, length - horizon + 1, stride)) | {length - horizon})
        )
        count = len(starts)
        slots = starts[:, None] + torch.arange(horizon)[None]
        ends = starts + horizon - 1

        def pack(value):
            if isinstance(value, dict):
                return {key: pack(item) for key, item in value.items()}
            if torch.is_tensor(value):
                if value.shape[:2] != (length, 1):
                    raise ValueError(f"Unexpected source sequence shape {value.shape}")
                return value[slots].transpose(1, 2).contiguous()
            return value

        fields = {
            field.name: pack(getattr(source, field.name))
            for field in dataclasses.fields(Trajectory)
            if field.name not in ("curr_obs", "next_obs")
        }
        fields.update(
            max_episode_length=count,
            curr_obs={key: value[starts] for key, value in source.curr_obs.items()},
            next_obs={key: value[ends] for key, value in source.next_obs.items()},
        )
        for key in ("rewards", "dones", "terminations", "truncations"):
            fields[key] = fields[key].squeeze(-1)
        chunked.append(Trajectory(**fields))
        audit.append(
            {
                "episode_index": episode_id(source),
                "source_action_steps": length,
                "chunks": count,
                "retained_action_steps": count * horizon,
                "stride": stride,
                "starts": current[starts].tolist(),
                "discarded_prefix_steps": 0,
                "discarded_source_indices": [],
                "discarded_human_steps": 0,
                "retained_human_steps": int(source.intervene_flags[slots, :, 0].sum()),
                "first_source_index": int(current[0]),
                "terminal_source_index": int(
                    source.next_obs["source_index"][-1].item()
                ),
            }
        )
    source_indices(chunked, horizon)
    return chunked, audit


def validate_source_episodes(
    trajectories: list[Trajectory], dataset: LeRobotV3Dataset
) -> None:
    """Reject transitions that cross recording boundaries."""
    ends = np.asarray([episode["dataset_to_index"] for episode in dataset.episodes])
    starts = np.asarray([episode["dataset_from_index"] for episode in dataset.episodes])
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
        "task_descriptions": [sample["task"]],
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


class G1ActionCoordinates:
    """Map physical actions to query-state joint deltas and absolute grippers."""

    def __init__(self, transform: dict):
        self.transform = transform
        self.scale = torch.tensor(transform["scale"], dtype=torch.float32)
        self.bias = torch.tensor(transform["bias"], dtype=torch.float32)
        self.state_scale = torch.tensor(
            transform.get("state_scale", transform["scale"]), dtype=torch.float32
        )
        self.state_bias = torch.tensor(
            transform.get("state_bias", transform["bias"]), dtype=torch.float32
        )
        self.relative_joints = int(transform.get("relative_joint_count", 0))
        values = (self.scale, self.bias, self.state_scale, self.state_bias)
        if any(v.shape != (16,) or not torch.isfinite(v).all() for v in values):
            raise ValueError("G1 coordinates require 16 finite constants per field")
        if not (self.scale > 0).all() or not (self.state_scale > 0).all():
            raise ValueError("G1 coordinate scales must be positive")
        if self.relative_joints not in (0, 14):
            raise ValueError(
                "Expected 14 relative joints or legacy absolute coordinates"
            )

    @classmethod
    def from_norm_stats(cls, path: Path, action_names: list[str]):
        """Reuse the SFT joint-delta and state quantiles without fitting holdout data."""
        stats = json.loads(path.read_text())["norm_stats"]
        low = torch.tensor(stats["actions"]["q01"][:14] + [0.0, 0.0])
        high = torch.tensor(stats["actions"]["q99"][:14] + [100.0, 100.0])
        span = high - low
        span[:14] += 1e-6
        state_low = torch.tensor(stats["state"]["q01"][:16])
        state_span = torch.tensor(stats["state"]["q99"][:16]) - state_low + 1e-6
        return cls(
            {
                "schema_version": "g1_query_state_delta_v1",
                "relative_joint_count": 14,
                "action_names": action_names,
                "scale": (span / 2).tolist(),
                "bias": (low + span / 2).tolist(),
                "state_scale": (state_span / 2).tolist(),
                "state_bias": (state_low + state_span / 2).tolist(),
                "norm_stats_path": str(path.resolve()),
                "norm_stats_sha256": sha256(path),
                "encode": "joints: (action-state_at_query-bias)/scale; grippers: (action-bias)/scale",
                "decode": "joints: normalized*scale+bias+state_at_query; grippers: normalized*scale+bias",
                "clipping": "none",
            }
        )

    def normalize(self, actions: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        delta = actions.float().clone()
        delta[..., : self.relative_joints] -= state.to(delta).unsqueeze(-2)[
            ..., : self.relative_joints
        ]
        return (delta - self.bias.to(delta)) / self.scale.to(delta)

    def denormalize(self, actions: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        physical = actions.float() * self.scale.to(actions) + self.bias.to(actions)
        physical[..., : self.relative_joints] += state.to(physical).unsqueeze(-2)[
            ..., : self.relative_joints
        ]
        return physical

    def proprio(self, state: torch.Tensor) -> torch.Tensor:
        return (state.float() - self.state_bias.to(state)) / self.state_scale.to(state)


def encode_trajectories(
    trajectories: list[Trajectory],
    features: dict[int, dict[str, torch.Tensor]],
    transform: dict,
    model_id: str,
    horizon: int,
) -> tuple[list[Trajectory], dict]:
    """Apply official human reference replacement, then reversible coordinates."""
    coordinates = G1ActionCoordinates(transform)

    encoded = []
    policy_targets = []
    human_count = 0
    ranges = {}
    for source in trajectories:
        observations = []
        for raw_obs in (source.curr_obs, source.next_obs):
            rows = [
                features[int(index)] for index in raw_obs["source_index"].reshape(-1)
            ]
            obs = {
                **raw_obs,
                **{key: torch.stack([row[key] for row in rows]) for key in rows[0]},
            }
            if not torch.equal(obs["proprio"], raw_obs["states"].float()):
                raise ValueError(
                    "Extracted proprio differs from the recorded source state"
                )
            observations.append(obs)
        current, following = observations
        original_reference = current["ref_chunk"].clone()
        human = source.intervene_flags.any(dim=-1)
        apply_rlt_interventions(current, source.actions, human)
        actions = coordinates.normalize(source.actions, current["states"])
        if not torch.isfinite(actions).all():
            raise ValueError("Executed normalized actions must be finite")
        if not torch.allclose(
            coordinates.denormalize(actions, current["states"]),
            source.actions.float(),
            atol=1e-5,
            rtol=0,
        ):
            raise ValueError("Executed-action coordinate roundtrip failed")
        for obs in observations:
            obs["ref_chunk"] = coordinates.normalize(obs["ref_chunk"], obs["states"])
            obs["proprio"] = coordinates.proprio(obs["states"])
            for key in ("z_rl", "ref_chunk", "proprio"):
                tensor_range(obs[key])
        target = current["ref_chunk"][:, 0, :horizon]
        if not torch.equal(target[human[:, 0]], actions[:, 0][human[:, 0]]):
            raise ValueError(
                "Human reference targets do not equal executed human actions"
            )
        policy_targets.append(target[~human[:, 0]])
        human_count += int(human.sum())
        encoded.append(
            dataclasses.replace(
                source,
                model_weights_id=model_id,
                actions=actions,
                curr_obs=current,
                next_obs=following,
                forward_inputs={
                    **source.forward_inputs,
                    "source_actions_physical": source.actions.clone(),
                    "stage1_ref_chunk_physical": original_reference,
                    "relative_joint_count": torch.full_like(
                        source.curr_obs["source_index"], coordinates.relative_joints
                    ),
                },
            )
        )
    for key in ("z_rl", "ref_chunk", "proprio"):
        ranges[key] = tensor_range(
            torch.cat(
                [
                    obs[key].reshape(-1)
                    for t in encoded
                    for obs in (t.curr_obs, t.next_obs)
                ]
            )
        )
    targets = torch.cat(policy_targets)
    bc_floor = (targets - targets.clamp(-1, 1)).square()
    ranges["policy_bc_target"] = {
        **tensor_range(targets),
        "action_steps": len(targets),
        "outside_unit_interval_action_step_ratio": float(
            (targets.abs() > 1).any(-1).float().mean()
        )
        if len(targets)
        else 0.0,
        "bounded_actor_mse_lower_bound": float(bc_floor.mean())
        if len(targets)
        else 0.0,
        "bounded_actor_mse_lower_bound_all_action_steps": float(bc_floor.sum())
        / (16 * horizon * sum(len(t.actions) for t in encoded)),
    }
    return encoded, {"human_action_steps": human_count, "ranges": ranges}


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


def center_replay(source: Path, output: Path, *, z_center: list[float]) -> dict:
    """Center RLT features while preserving official intervention references."""
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output must be empty: {output}")
    manifest = json.loads((source / "feature_manifest.json").read_text())
    trajectories = load_raw_replay(source)
    center = torch.tensor(z_center, dtype=torch.float32)
    if not torch.isfinite(center).all():
        raise ValueError("Feature center must contain finite values")
    for trajectory in trajectories:
        if center.shape != trajectory.curr_obs["z_rl"].shape[-1:]:
            raise ValueError("Feature center must match the RLToken dimension")
        for obs in (trajectory.curr_obs, trajectory.next_obs):
            obs["z_rl"] = obs["z_rl"].float() - center
    output.mkdir(parents=True, exist_ok=True)
    save_replay(trajectories, output, manifest["seed"])
    manifest.update(
        reference_input="official_intervention_replacement_on_current_human_steps",
        parent_replay=str(source.resolve()),
        parent_manifest_sha256=sha256(source / "feature_manifest.json"),
        z_center=z_center,
    )
    (output / "feature_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Stage 1 task YAML")
    parser.add_argument(
        "--stage2-config",
        type=Path,
        required=True,
        help="Read action/reference horizons from this task YAML",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument(
        "--action-transform",
        type=Path,
        help="Explicit transform; default uses SFT joint-delta quantiles",
    )
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--holdout-episodes", type=int, nargs="*", default=[])
    args = parser.parse_args()
    output_paths = {"train": args.output}
    if args.holdout_episodes:
        output_paths["validation"] = args.output.parent / "validation_replay"
    if len({path.resolve() for path in output_paths.values()}) != len(output_paths):
        raise ValueError("Train and validation replay paths must differ")
    for path in output_paths.values():
        if path.exists() and any(path.iterdir()):
            raise FileExistsError(f"Output must be empty: {path}")
    model_cfg = load_config(args.config, args.checkpoint)
    with initialize_config_dir(
        version_base="1.1", config_dir=str(args.stage2_config.resolve().parent)
    ):
        stage2_cfg = compose(config_name=args.stage2_config.stem)
    horizon = int(stage2_cfg.actor.model.num_action_chunks)
    ref_horizon = int(stage2_cfg.actor.model.ref_num_action_chunks)
    if not 1 <= horizon <= ref_horizon <= int(model_cfg.openpi.action_horizon):
        raise ValueError(
            "Expected 1 <= actor horizon <= reference horizon <= VLA horizon"
        )
    raw = load_raw_replay(args.source_cache / "raw_replay")
    partition_replay(raw, args.holdout_episodes)
    original_indices = source_indices(raw)
    dataset = LeRobotV3Dataset(args.source_cache / "dataset", action_horizon=1)
    validate_source_episodes(raw, dataset)
    source_manifest = json.loads(
        (args.source_cache / "source_manifest.json").read_text()
    )
    transform = (
        json.loads(args.action_transform.read_text())
        if args.action_transform
        else G1ActionCoordinates.from_norm_stats(
            Path(model_cfg.openpi_data.norm_stats_path), source_manifest["action_names"]
        ).transform
    )
    if transform["action_names"] != source_manifest["action_names"]:
        raise ValueError("Action transform joint ordering differs from source replay")
    if sum(len(t.actions) for t in raw) != source_manifest["num_transitions"]:
        raise ValueError("Replay transition count differs from source manifest")
    trajectories, chunk_audit = chunk_trajectories(raw, horizon, args.stride)
    indices = source_indices(trajectories, horizon)
    weights = resolve_full_weights(args.checkpoint)
    logger.info("Hashing trained checkpoint %s", weights)
    checkpoint_hash = sha256(weights)
    torch.manual_seed(args.seed)
    model = get_model(model_cfg).to(args.device).eval().requires_grad_(False)
    features = {}
    started = time.monotonic()
    expected_shapes = {
        "z_rl": (1, int(model.rlt_cfg.rlt_embed_dim)),
        "proprio": (1, 16),
        "ref_chunk": (1, int(model_cfg.openpi.action_horizon), 16),
    }
    with torch.inference_mode():
        for position, index in enumerate(indices):
            sample = dataset[index]
            if int(sample["index"]) != index:
                raise ValueError(
                    f"Source global row does not match dataset position {index}"
                )
            # Per-frame seeds make reference actions independent of traversal order.
            torch.manual_seed(args.seed + index)
            result = model.extract_rlt_obs(sample_to_env(sample))
            for key, shape in expected_shapes.items():
                if result[key].shape != shape or not torch.isfinite(result[key]).all():
                    raise ValueError(
                        f"Invalid {key} at source row {index}: {result[key].shape}"
                    )
            result["ref_chunk"] = result["ref_chunk"][:, :ref_horizon]
            features[index] = {
                key: value.float().cpu() for key, value in result.items()
            }
            completed = position + 1
            if completed == 1 or completed % 100 == 0 or completed == len(indices):
                elapsed = time.monotonic() - started
                logger.info(
                    "Features %d/%d; elapsed %.1fs; remaining %.1fs",
                    completed,
                    len(indices),
                    elapsed,
                    elapsed / completed * (len(indices) - completed),
                )

    encoded, diagnostics = encode_trajectories(
        trajectories, features, transform, checkpoint_hash[:16], horizon
    )
    groups = partition_replay(encoded, args.holdout_episodes)
    for name, group in groups.items():
        save_replay(group, output_paths[name], args.seed)
    manifest = {
        "schema_version": "g1_rlt_features_v2",
        "stage2_buffer_ready": True,
        "checkpoint": str(weights.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_loading": "Official get_model full_weights; required backbone and RLT keys checked by loader",
        "model_config": OmegaConf.to_container(model_cfg, resolve=True),
        "stage2_config": str(args.stage2_config.resolve()),
        "action_horizon": horizon,
        "reference_horizon": ref_horizon,
        "actor_reference_horizon": horizon,
        "actor_reference_contract": "Unmodified official RLTMLP consumes only the first actor-horizon reference actions",
        "source_cache": str(args.source_cache.resolve()),
        "source_manifest_sha256": sha256(args.source_cache / "source_manifest.json"),
        "action_transform": transform,
        "action_transform_sha256": hashlib.sha256(
            json.dumps(transform, sort_keys=True).encode()
        ).hexdigest(),
        "seed": args.seed,
        "rng": "torch.manual_seed(seed + global_source_index), one inference per unique frame",
        "unique_frames": len(indices),
        "source_unique_frames": len(original_indices),
        "source_indices": indices,
        "all_trajectories": len(encoded),
        "all_transitions": sum(len(t.actions) for t in encoded),
        "all_action_steps": sum(len(t.actions) for t in encoded) * horizon,
        "chunking": {
            "alignment": "sliding_with_terminal_window",
            "stride": args.stride,
            "padding": False,
            "episodes": chunk_audit,
        },
        "split": {
            name: sorted({episode_id(t) for t in group})
            for name, group in groups.items()
        },
        "reference_definition": "Fresh Stage1 prediction; current human action slots replaced by executed actions before normalization; next reference remains prediction",
        "coordinate_clipping": "none; target.clamp is used only to report the theoretical BC error lower bound",
        "reload_exact": True,
        "elapsed_seconds": time.monotonic() - started,
        "all_features_diagnostics": diagnostics,
    }
    for name, group in groups.items():
        split_manifest = {
            **manifest,
            "role": name,
            "trajectories": len(group),
            "transitions": sum(len(t.actions) for t in group),
            "action_steps": sum(len(t.actions) for t in group) * horizon,
            "split_human_action_steps": sum(
                int(t.intervene_flags[..., 0].sum()) for t in group
            ),
        }
        (output_paths[name] / "feature_manifest.json").write_text(
            json.dumps(split_manifest, indent=2) + "\n"
        )
        logger.info("Verified %s feature replay saved to %s", name, output_paths[name])


if __name__ == "__main__":
    main()
