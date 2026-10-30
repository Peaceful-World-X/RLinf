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

"""Evaluate a trained official G1 RLT actor/critic on held-out replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.mlp_policy import get_model
from rlinf.models.embodiment.openpi.checkpoint import resolve_full_weights
from rlinf.utils.ckpt_convertor.openpi._core import as_state_dict, strip_wrapper_prefix
from rlinf.utils.logging import get_logger
from toolkits.g1_rlt_features import (
    G1ActionCoordinates,
    episode_id,
    feature_contract_sha256,
    load_raw_replay,
    sha256,
)

logger = get_logger()


def load_actor(config, checkpoint: Path, device: str):
    """Strictly load the official architecture; never accept partial weights."""
    model = get_model(config, torch_dtype=torch.float32)
    state = strip_wrapper_prefix(
        as_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)),
        cast_dtype=None,
    )
    if not all(torch.isfinite(value).all() for value in state.values()):
        raise ValueError(f"Nonfinite checkpoint tensors: {checkpoint}")
    model.load_state_dict(state, strict=True)
    return model.float().to(device).eval().requires_grad_(False)


class G1RLTInference:
    """Adapt physical G1 observations/actions around the official RLT route."""

    def __init__(self, vla, actor, action_transform: dict, z_center=None) -> None:
        self.vla, self.actor = vla, actor
        self.coordinates = G1ActionCoordinates(action_transform)
        device = next(actor.parameters()).device
        self.scale = torch.tensor(action_transform["scale"], device=device)
        self.bias = torch.tensor(action_transform["bias"], device=device)
        self.z_center = torch.tensor(
            z_center if z_center is not None else [0.0] * actor.z_dim,
            dtype=torch.float32,
            device=device,
        )
        if (
            self.z_center.shape != (actor.z_dim,)
            or not torch.isfinite(self.z_center).all()
        ):
            raise ValueError("Expected one finite center per RLToken feature")
        if (
            self.scale.shape != (16,)
            or self.bias.shape != (16,)
            or not torch.isfinite(self.scale).all()
            or not torch.isfinite(self.bias).all()
            or (self.scale <= 0).any()
            or actor.step_action_dim != 16
            or actor.chunk_len != 10
            or actor.ref_chunk_len != 20
        ):
            raise ValueError(
                "Expected finite G1 affine coordinates and ref20/H10 actor"
            )

    def extract_rlt_obs(self, env_obs: dict) -> dict:
        """Keep VLA inputs physical and normalize only the extracted actor inputs."""
        features = self.vla.extract_rlt_obs(env_obs)
        features["z_rl"] = features["z_rl"].float() - self.z_center
        if features["ref_chunk"].shape[1] < self.actor.ref_chunk_len:
            raise ValueError("VLA must provide at least 20 reference actions")
        state = features["proprio"]
        features["ref_chunk"] = self.coordinates.normalize(
            features["ref_chunk"][:, : self.actor.ref_chunk_len], state
        )
        features["proprio"] = self.coordinates.proprio(state)
        return features

    @torch.inference_mode()
    def predict(self, env_obs: dict, *, actor_switch: bool = False) -> tuple:
        """Return physical actions and diagnostics; replay inputs stay normalized."""
        from rlinf.algorithms.rlt.rollout import predict_rlt_actions
        from rlinf.algorithms.rlt.route import RealworldRLTRoute

        batch = len(env_obs["states"])
        normalized, result = predict_rlt_actions(
            policy_model=self.actor,
            feature_model=self,
            rlt_route=RealworldRLTRoute(),
            env_obs=env_obs,
            final_obs=None,
            mode="eval",
            rlt_switch_flags=torch.full(
                (batch,), actor_switch, device=self.scale.device, dtype=torch.bool
            ),
        )
        state = torch.as_tensor(env_obs["states"], device=normalized.device)
        physical = self.coordinates.denormalize(normalized, state)
        q_values = self.actor.sac_q_forward(result["forward_inputs"], normalized)
        if physical.shape != (batch, 10, 16) or q_values.shape != (batch, 2):
            raise ValueError("Expected G1 actions [B,10,16] and twin Q [B,2]")
        if not all(torch.isfinite(value).all() for value in (physical, q_values)):
            raise ValueError("Nonfinite physical actions or critic output")
        result.update(normalized_actions=normalized, q_values=q_values)
        return physical, result


class G1OpenPIPolicy:
    """Expose physical G1 actions through the existing OpenPI wire protocol."""

    def __init__(self, inference: G1RLTInference) -> None:
        self.inference = inference

    def infer(self, observation: dict, **kwargs) -> dict:
        """Accept RGB images and physical state; return ten absolute actions."""
        if kwargs:
            raise ValueError("RLT H10 serving does not support RTC prefix inpainting")
        if "images" in observation:
            images = observation["images"]
            names = ("cam_high", "cam_left_wrist", "cam_right_wrist")
            if "cam_high" not in images and "cam_left_high" in images:
                images = {**images, "cam_high": images["cam_left_high"]}
            raw_images = [images[name] for name in names]
            state = observation["state"]
        else:
            raw_images = [
                observation[f"observation.images.{name}"]
                for name in ("cam_left_high", "cam_left_wrist", "cam_right_wrist")
            ]
            state = observation["observation.state"]
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (16,) or not np.isfinite(state).all():
            raise ValueError("Expected one finite physical G1 state with shape [16]")
        images = []
        for image in raw_images:
            image = np.asarray(image)
            if image.ndim == 3 and image.shape[0] == 3 and image.shape[-1] != 3:
                image = np.moveaxis(image, 0, -1)
            if image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError("Expected RGB images shaped [H,W,3] or [3,H,W]")
            if image.dtype != np.uint8 and not (
                np.issubdtype(image.dtype, np.floating)
                and np.isfinite(image).all()
                and image.min() >= 0
                and image.max() <= 1
            ):
                raise ValueError("RGB images must be uint8 or finite floats in [0,1]")
            images.append(image)
        if images[1].shape != images[2].shape:
            raise ValueError("The two wrist cameras must have the same image shape")
        actor_switch = observation.get("rlt_switch", False)
        if not isinstance(actor_switch, (bool, np.bool_)):
            raise ValueError("rlt_switch must be a boolean")
        task = observation.get("prompt", "Fasten the zip tie.")
        if not isinstance(task, str) or not task.strip():
            raise ValueError("prompt must be a nonempty task string")
        env_obs = {
            "main_images": images[0][None],
            "wrist_images": np.stack(images[1:])[None],
            "states": state[None],
            "task_descriptions": [task],
        }
        actions, result = self.inference.predict(
            env_obs, actor_switch=bool(actor_switch)
        )
        return {
            "actions": actions[0].float().cpu().numpy(),
            "q_values": result["q_values"][0].float().cpu().numpy(),
            "rlt_switch": bool(actor_switch),
        }


def recorded_returns(
    rewards: torch.Tensor, terminations: torch.Tensor, gamma: float
) -> torch.Tensor:
    """Discount recorded step/chunk rewards at the supplied temporal discount."""
    returns = torch.empty_like(rewards)
    value = rewards.new_zeros(())
    for index in reversed(range(len(rewards))):
        value = (
            rewards[index] + gamma * (~terminations[index]).to(rewards.dtype) * value
        )
        returns[index] = value
    return returns


def imitation_metrics(
    predictions: torch.Tensor,
    actual: torch.Tensor,
    references: torch.Tensor,
    human: torch.Tensor,
    scale: torch.Tensor,
) -> dict:
    """Compare actors and original Stage 1 references against recorded controls."""
    metrics = {}
    action_dim = scale.numel()
    predictions = predictions.reshape(-1, action_dim)
    actual = actual.reshape(-1, action_dim)
    references = references.reshape(-1, action_dim)
    human = human.reshape(-1)
    targets = torch.where(human[:, None], actual, references)
    for name, selected in {
        "all": torch.ones_like(human),
        "human": human,
        "policy": ~human,
    }.items():
        count = int(selected.sum())
        if not count:
            metrics[name] = {"action_steps": 0}
            continue
        actor_error = predictions[selected] - actual[selected]
        reference_error = references[selected] - actual[selected]
        physical_error = actor_error * scale
        reference_physical_error = reference_error * scale
        metrics[name] = {
            "action_steps": count,
            "actor_bc_target_mse_normalized": float(
                (predictions[selected] - targets[selected]).square().mean()
            ),
            "actor_actual_mse_normalized": float(actor_error.square().mean()),
            "actor_actual_mae_normalized": float(actor_error.abs().mean()),
            "stage1_reference_actual_mse_normalized": float(
                reference_error.square().mean()
            ),
            "stage1_reference_actual_mae_normalized": float(
                reference_error.abs().mean()
            ),
            "actor_joint_mae_radians": float(physical_error[:, :14].abs().mean()),
            "actor_gripper_mae_dataset_units": float(
                physical_error[:, 14:].abs().mean()
            ),
            "stage1_joint_mae_radians": float(
                reference_physical_error[:, :14].abs().mean()
            ),
            "stage1_gripper_mae_dataset_units": float(
                reference_physical_error[:, 14:].abs().mean()
            ),
        }
    return metrics


@torch.inference_mode()
def evaluate_trajectory(
    model,
    target_model,
    trajectory,
    scale,
    bias,
    gamma,
    batch_size,
    device,
    bootstrap_type,
):
    """Evaluate real features with unmodified Stage 1 references at inference."""
    length = len(trajectory.actions)
    horizon, action_dim = model.chunk_len, model.step_action_dim
    actual = trajectory.actions.reshape(length, horizon, action_dim).float()
    human = trajectory.intervene_flags.reshape(length, horizon, action_dim).any(-1)
    if not torch.equal(
        trajectory.curr_obs["source_index"] + horizon,
        trajectory.next_obs["source_index"],
    ):
        raise ValueError("Replay next observation must follow the full action chunk")
    relative_joints = torch.as_tensor(
        trajectory.forward_inputs.get("relative_joint_count", 0)
    ).unique()
    if relative_joints.numel() != 1:
        raise ValueError("Trajectory must use one action coordinate system")
    coordinates = G1ActionCoordinates(
        {
            "scale": scale.tolist(),
            "bias": bias.tolist(),
            "relative_joint_count": int(relative_joints.item()),
        }
    )
    state = trajectory.curr_obs["states"].float()
    original_ref = coordinates.normalize(
        trajectory.forward_inputs["stage1_ref_chunk_physical"].float(), state
    )
    references = original_ref.reshape(length, -1, action_dim)[:, :horizon]
    if not torch.allclose(
        coordinates.denormalize(actual, state.reshape(length, action_dim)),
        trajectory.forward_inputs["source_actions_physical"].reshape(
            length, horizon, action_dim
        ),
        atol=1e-5,
        rtol=0,
    ):
        raise ValueError("Executed-action inverse transform failed")
    predictions, q_values, q_next_values = [], [], []
    for begin in range(0, length, batch_size):
        end = min(begin + batch_size, length)
        observations = []
        for source in (trajectory.curr_obs, trajectory.next_obs):
            observations.append(
                {
                    key: source[key][begin:end, 0].to(device)
                    for key in ("z_rl", "proprio", "ref_chunk")
                }
            )
        current, following = observations
        # Keep held-out human actions out of the actor's reference input.
        current["ref_chunk"] = original_ref[begin:end, 0].to(device)
        pi, _, _ = model(forward_type=ForwardType.SAC, obs=current, deterministic=True)
        next_pi, _, _ = model(
            forward_type=ForwardType.SAC, obs=following, deterministic=True
        )
        q_data = model(
            forward_type=ForwardType.SAC_Q,
            obs=current,
            actions=actual[begin:end].to(device),
        )
        next_q = target_model(
            forward_type=ForwardType.SAC_Q, obs=following, actions=next_pi
        )
        if q_data.shape != (end - begin, 2) or next_q.shape != q_data.shape:
            raise ValueError(
                f"Expected two Q heads, got {q_data.shape}, {next_q.shape}"
            )
        for value in (pi, next_pi, q_data, next_q):
            if not torch.isfinite(value).all():
                raise ValueError("Nonfinite actor or critic output")
        if (pi.abs() > 1).any() or (next_pi.abs() > 1).any():
            raise ValueError("Actor actions must lie in [-1, 1]")
        predictions.append(pi.cpu().reshape(-1, horizon, action_dim))
        q_values.append(q_data.cpu())
        q_next_values.append(next_q.cpu())
    predictions = torch.cat(predictions)
    q_values = torch.cat(q_values)
    q_next = torch.cat(q_next_values).amin(-1)
    step_rewards = trajectory.rewards.reshape(length, horizon).float()
    step_terminals = trajectory.terminations.reshape(length, horizon).bool()
    if (
        step_terminals[:-1].any()
        or step_terminals[-1, :-1].any()
        or not step_terminals[-1, -1]
    ):
        raise ValueError("Expected full real chunks ending exactly at the terminal")
    discounts = torch.tensor(gamma).pow(torch.arange(horizon))
    rewards = (step_rewards * discounts).sum(-1)
    terminals = step_terminals.any(-1)
    bootstrap = (
        torch.ones_like(rewards) if bootstrap_type == "always" else (~terminals).float()
    )
    chunk_discount = gamma**horizon
    td_target = rewards + chunk_discount * bootstrap * q_next
    starts = trajectory.curr_obs["source_index"].reshape(-1).long()
    slots = starts[:, None] - starts[0] + torch.arange(horizon)[None]
    frame_rewards = torch.empty(int(slots.max()) + 1)
    frame_terminals = torch.empty(len(frame_rewards), dtype=torch.bool)
    frame_rewards[slots] = step_rewards
    frame_terminals[slots] = step_terminals
    mc = recorded_returns(frame_rewards, frame_terminals, gamma)[starts - starts[0]]
    physical = coordinates.denormalize(predictions, state.reshape(length, action_dim))
    roundtrip = (
        (
            coordinates.normalize(physical, state.reshape(length, action_dim))
            - predictions
        )
        .abs()
        .max()
    )
    if not torch.isfinite(physical).all() or float(roundtrip) > 1e-5:
        raise ValueError("Actor physical-action inverse transform failed")
    return {
        "predictions": predictions,
        "actual": actual,
        "references": references,
        "physical_predictions": physical,
        "physical_actual": coordinates.denormalize(
            actual, state.reshape(length, action_dim)
        ),
        "physical_references": coordinates.denormalize(
            references, state.reshape(length, action_dim)
        ),
        "source_indices": starts,
        "human": human,
        "q_values": q_values,
        "td_residual": q_values - td_target[:, None],
        "mc_error": q_values - mc[:, None],
        "mc": mc,
        "inverse_roundtrip_max_abs_error": float(roundtrip),
        "ends_in_observed_termination": bool(terminals[-1]),
    }


def summarize(values: dict, scale: torch.Tensor) -> dict:
    """Summarize imitation and observational critic diagnostics."""
    return {
        "reference_fidelity": reference_fidelity(
            values["predictions"], values["references"], scale
        ),
        "imitation": imitation_metrics(
            values["predictions"],
            values["actual"],
            values["references"],
            values["human"],
            scale,
        ),
        "actor_normalized_min": float(values["predictions"].min()),
        "actor_normalized_max": float(values["predictions"].max()),
        "q_mean_by_head": values["q_values"].mean(0).tolist(),
        "q_min_by_head": values["q_values"].amin(0).tolist(),
        "q_max_by_head": values["q_values"].amax(0).tolist(),
        "td_residual_mse_by_head": values["td_residual"].square().mean(0).tolist(),
        "td_residual_mae_by_head": values["td_residual"].abs().mean(0).tolist(),
        "recorded_mc_mse_by_head": values["mc_error"].square().mean(0).tolist(),
        "recorded_mc_mae_by_head": values["mc_error"].abs().mean(0).tolist(),
        "recorded_mc_min": float(values["mc"].min()),
        "recorded_mc_max": float(values["mc"].max()),
    }


def reference_fidelity(predictions, references, scale) -> dict:
    """Measure per-joint reference tracking in physical G1 coordinates."""
    error = ((predictions - references) * scale).reshape(-1, 16).abs()
    mae, p95 = error.mean(0), torch.quantile(error, 0.95, dim=0)
    score = max(
        float(mae[:14].max() / 0.005),
        float(p95[:14].max() / 0.02),
        float(mae[14:].max()),
    )
    return {
        "joint_mae_radians": mae[:14].tolist(),
        "joint_p95_radians": p95[:14].tolist(),
        "gripper_mae_dataset_units": mae[14:].tolist(),
        "joint_mean_mae_radians": float(mae[:14].mean()),
        "max_threshold_ratio": score,
        "passed": score <= 1.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, required=True, help="Stage 2 training YAML"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--target-checkpoint", type=Path)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--training-diagnostics", action="store_true")
    parser.add_argument(
        "--raw-source-index", type=int, help="Infer one raw global frame"
    )
    parser.add_argument(
        "--source-cache", type=Path, help="Optional raw cache for inference"
    )
    parser.add_argument(
        "--serve", action="store_true", help="Serve an OpenPI WebSocket policy"
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.source_cache is not None and args.raw_source_index is None:
        parser.error("--source-cache requires --raw-source-index")
    if args.serve and (args.raw_source_index is not None or args.training_diagnostics):
        parser.error(
            "--serve cannot be combined with raw-frame or training diagnostics"
        )
    if not args.serve and args.output is None:
        parser.error("--output is required for offline evaluation")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be in [1,65535]")
    with initialize_config_dir(
        version_base="1.1", config_dir=str(args.config.resolve().parent)
    ):
        cfg = compose(config_name=args.config.stem)
    if (
        cfg.actor.model.model_type != "rlt_mlp_policy"
        or cfg.actor.model.num_action_chunks < 1
        or cfg.actor.model.action_dim != 16
    ):
        raise ValueError("Expected official G1 RLT actor/critic config")
    gamma = float(cfg.algorithm.gamma)
    if not 0 <= gamma <= 1 or cfg.algorithm.bootstrap_type not in (
        "standard",
        "always",
    ):
        raise ValueError("Unsupported discount or bootstrap configuration")
    manifest = json.loads((args.replay / "feature_manifest.json").read_text())
    expected_contract = cfg.get("rlt_input_signature", None)
    if expected_contract is not None and expected_contract != feature_contract_sha256(
        manifest
    ):
        raise ValueError(
            "Replay VLA/feature coordinates do not match the actor configuration"
        )
    if not manifest["stage2_buffer_ready"]:
        raise ValueError("Feature replay has not passed its acceptance checks")
    horizon = int(cfg.actor.model.num_action_chunks)
    if manifest["action_horizon"] != horizon or manifest["reference_horizon"] != int(
        cfg.actor.model.ref_num_action_chunks
    ):
        raise ValueError("Replay horizons do not match the trained actor configuration")
    if (
        args.raw_source_index is None
        and manifest["role"] != "validation"
        and not args.training_diagnostics
    ):
        raise ValueError(
            "Use independent validation_replay; training-fit diagnostics require --training-diagnostics"
        )
    training = set(manifest["split"]["train"])
    validation = set(manifest["split"].get("validation", []))
    if training & validation:
        raise ValueError("Training and validation episodes overlap")
    trajectories = load_raw_replay(args.replay)
    if {episode_id(t) for t in trajectories} != set(
        manifest["split"][manifest["role"]]
    ):
        raise ValueError("Replay episode IDs do not match its split manifest")
    checkpoint = resolve_full_weights(args.checkpoint)
    if checkpoint is None:
        raise FileNotFoundError(f"No trained full_weights.pt at {args.checkpoint}")
    model = load_actor(cfg.actor.model, checkpoint, args.device)
    if args.serve:
        from openpi.serving.websocket_policy_server import WebsocketPolicyServer

        from rlinf.models.embodiment.openpi import get_model as get_vla

        vla_cfg = OmegaConf.create(manifest["model_config"])
        vla_checkpoint = resolve_full_weights(Path(vla_cfg.model_path))
        if (
            vla_checkpoint is None
            or vla_checkpoint.resolve() != Path(manifest["checkpoint"]).resolve()
            or sha256(vla_checkpoint) != manifest["checkpoint_sha256"]
        ):
            raise ValueError(
                "Serving Stage1 weights differ from the encoded feature model"
            )
        torch.manual_seed(args.seed)
        vla = get_vla(vla_cfg).to(args.device).eval().requires_grad_(False)
        policy = G1OpenPIPolicy(
            G1RLTInference(
                vla, model, manifest["action_transform"], manifest.get("z_center")
            )
        )
        # Loading weights may consume RNG; initialize the inference stream afterwards.
        torch.manual_seed(args.seed)
        WebsocketPolicyServer(
            policy,
            host=args.host,
            port=args.port,
            metadata={
                "policy": "RLinf G1 RLT actor/critic",
                "action_horizon": 10,
                "action_dim": 16,
                "reference_horizon": 20,
                "actor_reference_horizon": 10,
                "action_names": manifest["action_transform"]["action_names"],
                "action_units": "absolute joint radians (14), gripper dataset 0..100 (2)",
                "default_rlt_switch": False,
                "actor_scope": "zip-tie insertion; send rlt_switch=true to enable actor",
                "rtc_supported": False,
                "client_async_supported": True,
                "rtc_prefix_supported": False,
                "action_control_hz": 30,
                "actor_checkpoint_sha256": sha256(checkpoint),
                "stage1_checkpoint_sha256": manifest["checkpoint_sha256"],
            },
        ).serve_forever()
        return
    if args.raw_source_index is not None:
        from rlinf.data.datasets.openpi.lerobot_v3_dataset import LeRobotV3Dataset
        from rlinf.models.embodiment.openpi import get_model as get_vla
        from toolkits.g1_rlt_features import sample_to_env

        cache = args.source_cache or Path(manifest["source_cache"])
        sample = LeRobotV3Dataset(cache / "dataset", action_horizon=1)[
            args.raw_source_index
        ]
        vla_cfg = OmegaConf.create(manifest["model_config"])
        vla_checkpoint = resolve_full_weights(Path(vla_cfg.model_path))
        if (
            vla_checkpoint is None
            or vla_checkpoint.resolve() != Path(manifest["checkpoint"]).resolve()
        ):
            raise ValueError("Feature manifest specifies conflicting VLA checkpoints")
        vla = get_vla(vla_cfg).to(args.device).eval().requires_grad_(False)
        inference = G1RLTInference(
            vla, model, manifest["action_transform"], manifest.get("z_center")
        )
        torch.manual_seed(int(manifest["seed"]) + args.raw_source_index)
        actions, result = inference.predict(sample_to_env(sample), actor_switch=True)
        source_hash = sha256(cache / "source_manifest.json")
        report = {
            "status": "validated_raw_observation_to_physical_actions",
            "actor_checkpoint": str(checkpoint.resolve()),
            "stage1_checkpoint": manifest["checkpoint"],
            "source_cache": str(cache.resolve()),
            "source_manifest_sha256": source_hash,
            "same_source_manifest_as_replay": source_hash
            == manifest["source_manifest_sha256"],
            "source_index": args.raw_source_index,
            "episode_index": int(sample["episode_index"]),
            "frame_index": int(sample["frame_index"]),
            "reference_shape": list(result["forward_inputs"]["ref_chunk"].shape),
            "action_shape": list(actions.shape),
            "physical_actions": actions.cpu().tolist(),
            "normalized_actions": result["normalized_actions"].cpu().tolist(),
            "q_values": result["q_values"].cpu().tolist(),
            "scope": "Offline inference only; no robot motion or policy-success assessment.",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        logger.info("Raw-frame inference saved to %s", args.output)
        return
    target_path = args.target_checkpoint
    if target_path is None:
        candidate = (
            checkpoint.parent.parent
            / "sac_components/target_model/checkpoint_rank_0.pt"
        )
        if candidate.is_file():
            target_path = candidate
    target_model = (
        load_actor(cfg.actor.model, target_path, args.device)
        if target_path is not None
        else model
    )
    scale = torch.tensor(manifest["action_transform"]["scale"], dtype=torch.float32)
    bias = torch.tensor(manifest["action_transform"]["bias"], dtype=torch.float32)
    rows = [
        evaluate_trajectory(
            model,
            target_model,
            trajectory,
            scale,
            bias,
            gamma,
            args.batch_size,
            args.device,
            cfg.algorithm.bootstrap_type,
        )
        for trajectory in trajectories
    ]
    aggregate = {
        key: torch.cat([row[key] for row in rows])
        for key in rows[0]
        if torch.is_tensor(rows[0][key])
    }
    report = {
        "status": "validated_finite_outputs_and_coordinate_roundtrip",
        "role": manifest["role"],
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "strict_checkpoint_load": True,
        "training_config_path": str(args.config.resolve()),
        "training_config_sha256": sha256(args.config),
        "target_checkpoint": str(target_path.resolve())
        if target_path is not None
        else None,
        "td_target_critic": "saved target critic"
        if target_path is not None
        else "online critic; self-Bellman diagnostic",
        "inference": "deterministic actor, no reference dropout; original Stage1 reference even on human frames",
        "baseline": "Original Stage1 prediction before human replacement; no clipping",
        "critic_interpretation": "TD and recorded Monte Carlo errors are offline diagnostics, not policy-value ground truth or robot success estimates. Deterministic next actions differ from stochastic training targets.",
        "config": OmegaConf.to_container(cfg.actor.model, resolve=True),
        "gamma": gamma,
        "action_horizon": horizon,
        "reference_horizon": int(cfg.actor.model.ref_num_action_chunks),
        "actor_reference_horizon": horizon,
        "bootstrap_discount": gamma**horizon,
        "bootstrap_type": cfg.algorithm.bootstrap_type,
        "replay": str(args.replay.resolve()),
        "feature_manifest_sha256": sha256(args.replay / "feature_manifest.json"),
        "episode_indices": sorted({episode_id(t) for t in trajectories}),
        "transitions": sum(len(t.actions) for t in trajectories),
        "action_steps": sum(len(t.actions) for t in trajectories) * horizon,
        "inverse_roundtrip_max_abs_error": max(
            row["inverse_roundtrip_max_abs_error"] for row in rows
        ),
        "all_recorded_intervals_end_in_termination": all(
            row["ends_in_observed_termination"] for row in rows
        ),
        "aggregate": summarize(aggregate, scale),
        "per_trajectory": [
            {"episode_index": episode_id(t), **summarize(row, scale)}
            for t, row in zip(trajectories, rows, strict=True)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    logger.info("Offline actor/critic evaluation saved to %s", args.output)


if __name__ == "__main__":
    main()
