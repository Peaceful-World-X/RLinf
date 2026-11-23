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

"""G1 image-to-action inference through the official RLinf RLT route."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from rlinf.models.embodiment.openpi.checkpoint import resolve_full_weights
from rlinf.models.embodiment.openpi.dataconfig.g1_dataconfig import G1ActionCoordinates
from rlinf.utils.ckpt_convertor.openpi._core import as_state_dict, strip_wrapper_prefix
from rlinf.utils.logging import get_logger

logger = get_logger()


def load_actor(config: Path, checkpoint: Path, device: str):
    """Require complete finite actor/critic weights in the official architecture."""
    from rlinf.models.embodiment.mlp_policy import get_model

    config = config.resolve()
    with initialize_config_dir(version_base="1.1", config_dir=str(config.parent)):
        cfg = compose(config_name=config.stem)
    model = get_model(cfg.actor.model).float()
    weights = resolve_full_weights(checkpoint)
    if weights is None:
        raise FileNotFoundError(f"Missing actor full_weights.pt at {checkpoint}")
    state = strip_wrapper_prefix(
        as_state_dict(torch.load(weights, map_location="cpu", weights_only=True)),
        cast_dtype=None,
    )
    if not all(torch.isfinite(v).all() for v in state.values()):
        raise ValueError("Nonfinite actor/critic checkpoint")
    model.load_state_dict(state, strict=True)
    if (
        not (0 < model.chunk_len == model.ref_chunk_len <= 50)
        or model.step_action_dim not in (14, 16)
        or model.proprio_dim != model.step_action_dim
    ):
        raise ValueError(
            "Expected G1 matching action/reference horizons up to 50 and 14D or 16D actions/proprio"
        )
    return model.to(device).eval().requires_grad_(False)


class G1RLTPolicy:
    """Input physical states/RGB cameras; return absolute physical G1 actions."""

    def __init__(self, feature_config, actor_config, actor_checkpoint, device="cuda:0"):
        from rlinf.models.embodiment.openpi import get_model

        cfg = OmegaConf.load(feature_config)
        if not cfg.openpi.use_rlt or resolve_full_weights(cfg.model_path) is None:
            raise ValueError(
                "Feature config must point to the selected trained Stage 1"
            )
        self.actor = load_actor(Path(actor_config), Path(actor_checkpoint), device)
        self.features = get_model(cfg).to(device).eval().requires_grad_(False)
        self.features.rlt_module.float()
        self.coordinates = G1ActionCoordinates(cfg.openpi_data.norm_stats_path)
        self.device = torch.device(device)

    def extract_rlt_obs(self, env_obs: dict) -> dict:
        result = self.features.extract_rlt_obs(env_obs)
        state = result["proprio"]
        result["ref_chunk"] = self.coordinates.normalize(
            result["ref_chunk"][:, : self.actor.chunk_len], state
        )
        result["proprio"] = self.coordinates.proprio(state)
        result["states"] = state
        if self.actor.step_action_dim == 14:
            # Grippers bypass the joint actor and retain the same VLA prediction.
            result["gripper_reference"] = result["ref_chunk"][..., 14:]
            result["ref_chunk"] = result["ref_chunk"][..., :14]
            result["proprio"] = result["proprio"][..., :14]
            result["states"] = state[..., :14]
        result["base_action"] = result["ref_chunk"].clone()
        return result

    @torch.inference_mode()
    def predict_batch(self, env_obs: dict, *, use_actor=True):
        """Use actor in insertion, VLA otherwise; flags may be scalar or per sample."""
        from rlinf.algorithms.rlt.rollout import predict_rlt_actions
        from rlinf.algorithms.rlt.route import RealworldRLTRoute

        count = len(env_obs["states"])
        flags = torch.as_tensor(use_actor, device=self.device, dtype=torch.bool).expand(
            count
        )
        normalized, result = predict_rlt_actions(
            policy_model=self.actor,
            feature_model=self,
            rlt_route=RealworldRLTRoute(),
            env_obs=env_obs,
            final_obs=None,
            mode="eval",
            rlt_switch_flags=flags,
        )
        state = torch.as_tensor(env_obs["states"], device=normalized.device).float()
        full_actions = normalized
        if self.actor.step_action_dim == 14:
            full_actions = torch.cat(
                [normalized, result["forward_inputs"]["gripper_reference"]], dim=-1
            )
        physical = self.coordinates.denormalize(full_actions, state)
        if (
            physical.shape != (count, self.actor.chunk_len, 16)
            or not torch.isfinite(physical).all()
        ):
            raise ValueError("Expected finite physical actions [B,H,16]")
        result["normalized_actions"] = normalized
        return physical, result

    def predict(self, images: dict, state, task: str, *, insertion: bool):
        """Single observation interface returning physical [H,16] actions."""
        import numpy as np

        env = {
            "states": np.asarray(state, dtype=np.float32)[None],
            "main_images": np.asarray(images["cam_left_high"])[None],
            "wrist_images": np.stack(
                [images["cam_left_wrist"], images["cam_right_wrist"]]
            )[None],
            "task_descriptions": [task],
        }
        return self.predict_batch(env, use_actor=insertion)[0][0].cpu().numpy()


class G1OpenPIPolicy:
    """Serve physical G1 actions through the existing OpenPI client protocol."""

    def __init__(self, policy: G1RLTPolicy) -> None:
        self.policy = policy

    @torch.inference_mode()
    def infer(self, observation: dict, **kwargs) -> dict:
        """Accept three RGB images, a physical state and an insertion switch."""
        import numpy as np

        if kwargs:
            raise ValueError("RLT serving does not support RTC prefix inpainting")
        names = ("cam_left_high", "cam_left_wrist", "cam_right_wrist")
        if "images" in observation:
            raw = observation["images"]
            if "cam_left_high" not in raw and "cam_high" in raw:
                raw = {**raw, "cam_left_high": raw["cam_high"]}
            images = [raw[name] for name in names]
            state = observation["state"]
        else:
            images = [observation[f"observation.images.{name}"] for name in names]
            state = observation["observation.state"]
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (16,) or not np.isfinite(state).all():
            raise ValueError("Expected one finite physical G1 state with shape [16]")
        checked = []
        for image in images:
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
            checked.append(image)
        if checked[1].shape != checked[2].shape:
            raise ValueError("The two wrist cameras must have the same image shape")
        switch = observation.get("rlt_switch", True)
        if not isinstance(switch, (bool, np.bool_)):
            raise ValueError("rlt_switch must be a boolean")
        task = observation.get("prompt", "Fasten the zip tie.")
        if not isinstance(task, str) or not task.strip():
            raise ValueError("prompt must be a nonempty task string")
        env = {
            "states": state[None],
            "main_images": checked[0][None],
            "wrist_images": np.stack(checked[1:])[None],
            "task_descriptions": [task],
        }
        actions, result = self.policy.predict_batch(env, use_actor=bool(switch))
        q_values = self.policy.actor.sac_q_forward(
            result["forward_inputs"], result["normalized_actions"]
        )
        if not torch.isfinite(q_values).all():
            raise ValueError("Nonfinite critic prediction")
        return {
            "actions": actions[0].float().cpu().numpy(),
            "q_values": q_values[0].float().cpu().numpy(),
            "rlt_switch": bool(switch),
        }


@torch.inference_mode()
def evaluate_cache(actor, cache: Path, device: str) -> dict:
    """Measure every held-out chunk with pure VLA references, without smoothing."""
    data = torch.load(cache, map_location="cpu", weights_only=True)
    rows = []
    for start in range(0, len(data["actions"]), 128):
        obs = {
            k: v[start : start + 128].to(device) for k, v in data["curr_obs"].items()
        }
        rows.append(actor.predict_action_batch(obs, mode="eval")[0].cpu())
    prediction = torch.cat(rows)
    score_horizon = int(data.get("score_horizon", prediction.shape[-2]))
    if (
        not (0 < score_horizon <= prediction.shape[-2])
        or data["actions"].shape[-2] != score_horizon
    ):
        raise ValueError("Invalid evaluation score horizon")
    prediction = prediction[:, :score_horizon]
    reference = data["curr_obs"]["ref_chunk"][:, :score_horizon]
    if not torch.isfinite(prediction).all():
        raise ValueError("Nonfinite actor prediction")
    error = (prediction - data["actions"]).square()
    ref_error = (reference - data["actions"]).square()
    human = data["intervene_flags"].bool()
    physical = (prediction - data["actions"]) * data["action_scale"]
    result = {
        "chunks": len(prediction),
        "score_horizon": score_horizon,
        "frames": len(prediction) * score_horizon,
        "reference_joint_mse_radians": float(
            (ref_error[..., :14] * data["action_scale"][:14].square()).mean()
        ),
        "autonomous_joint_mse_radians": float(
            physical[..., :14][~human[..., :14]].square().mean()
        )
        if (~human[..., :14]).any()
        else None,
        "all_mse": float(error.mean()),
        "reference_all_mse": float(ref_error.mean()),
        "human_mse": float(error[human].mean()),
        "reference_human_mse": float(ref_error[human].mean()),
        "joint_mae_radians": float(physical[..., :14].abs().mean()),
        "joint_mse_radians": float(physical[..., :14].square().mean()),
        "human_joint_mse_radians": float(
            physical[..., :14][human[..., :14]].square().mean()
        ),
        "reference_human_joint_mse_radians": float(
            (
                ((reference - data["actions"]) * data["action_scale"])[..., :14][
                    human[..., :14]
                ]
            )
            .square()
            .mean()
        ),
        "action_mode": getattr(actor, "action_mode", "state_delta"),
        "reference_joint_mae_radians": float(
            ((reference - data["actions"]) * data["action_scale"])[..., :14]
            .abs()
            .mean()
        ),
        "human_joint_mae_radians": float(
            physical[..., :14][human[..., :14]].abs().mean()
        ),
        "actor_action_dim": actor.step_action_dim,
        "strict_load": True,
        "predictions_finite": True,
    }
    if physical.shape[-1] > 14:
        result["gripper_mae"] = float(physical[..., 14:].abs().mean())
    result["offline_target_passed"] = (
        result["joint_mse_radians"] < result["reference_joint_mse_radians"]
        if "score_horizon" in data
        else (
            result["human_mse"] < result["reference_human_mse"]
            and result["all_mse"] <= 1.1 * result["reference_all_mse"]
        )
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("actor-config", "actor-checkpoint"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--evaluation-cache", type=Path)
    parser.add_argument("--feature-config", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.serve:
        from openpi.serving.websocket_policy_server import WebsocketPolicyServer
        from toolkits.g1_rlt_features import sha256

        if args.feature_config is None or args.evaluation_cache is not None:
            parser.error("Serving requires --feature-config without --evaluation-cache")
        policy = G1RLTPolicy(
            args.feature_config, args.actor_config, args.actor_checkpoint, args.device
        )
        feature_cfg = OmegaConf.load(args.feature_config)
        torch.manual_seed(args.seed)
        WebsocketPolicyServer(
            G1OpenPIPolicy(policy),
            host=args.host,
            port=args.port,
            metadata={
                "policy": "RLinf G1 RLT actor/critic",
                "action_horizon": policy.actor.chunk_len,
                "reference_horizon": policy.actor.ref_chunk_len,
                "vla_horizon": 50,
                "action_dim": 16,
                "actor_action_dim": policy.actor.step_action_dim,
                "gripper_source": "vla"
                if policy.actor.step_action_dim == 14
                else "actor",
                "action_units": "absolute joint radians (14), gripper dataset 0..100 (2)",
                "default_rlt_switch": True,
                "actor_scope": "zip-tie insertion; send rlt_switch=false for VLA",
                "rtc_supported": False,
                "actor_checkpoint_sha256": sha256(
                    resolve_full_weights(args.actor_checkpoint)
                ),
                "stage1_checkpoint_sha256": sha256(
                    resolve_full_weights(feature_cfg.model_path)
                ),
            },
        ).serve_forever()
        return
    if args.output is None:
        parser.error("Verification requires --output")
    if args.evaluation_cache:
        actor = load_actor(args.actor_config, args.actor_checkpoint, args.device)
        report = evaluate_cache(actor, args.evaluation_cache, args.device)
    else:
        from toolkits.g1_rlt_features import episode_id, load_raw_replay, sample_to_env

        from rlinf.data.datasets.openpi.lerobot_v3_dataset import LeRobotV3Dataset

        if args.cache is None or args.feature_config is None:
            parser.error("Image inference requires --cache and --feature-config")
        raw = next(
            t for t in load_raw_replay(args.cache / "raw_replay") if episode_id(t) == 4
        )
        index = int(raw.curr_obs["source_index"][0].item())
        dataset = LeRobotV3Dataset(args.cache / "dataset", 1, sample_indices=[index])
        env = sample_to_env(dataset[0])
        policy = G1RLTPolicy(
            args.feature_config, args.actor_config, args.actor_checkpoint, args.device
        )
        torch.manual_seed(42)
        action, result = policy.predict_batch(env, use_actor=True)
        obs = {
            k: result["forward_inputs"][k]
            for k in ("z_rl", "proprio", "ref_chunk", "states", "base_action")
        }
        direct = policy.actor.predict_action_batch(obs, mode="eval")[0]
        if not torch.equal(direct, result["normalized_actions"]):
            raise ValueError("Insertion route did not execute native actor")
        torch.manual_seed(42)
        reference, reference_info = policy.predict_batch(env, use_actor=False)
        if not torch.equal(reference_info["normalized_actions"], obs["ref_chunk"]):
            raise ValueError("VLA route differs from pure model reference")
        if policy.actor.step_action_dim == 14 and not torch.equal(
            action[..., 14:], reference[..., 14:]
        ):
            raise ValueError("Joint-only actor changed VLA gripper actions")
        report = {
            "source_index": index,
            "action_shape": list(action.shape),
            "actor_route_exact": True,
            "vla_route_exact": True,
            "finite": True,
            "actor_action_dim": policy.actor.step_action_dim,
            "gripper_source": "vla" if policy.actor.step_action_dim == 14 else "actor",
            "actions": action.cpu().tolist(),
            "reference": reference.cpu().tolist(),
            "q_values": policy.actor.sac_q_forward(obs, direct).cpu().tolist(),
        }
    report["scope"] = (
        "Recorded data compatibility and offline errors; no real robot success measurement"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    logger.info("G1 RLT verification: %s", report)


if __name__ == "__main__":
    main()
