#!/usr/bin/env python3
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

"""Export actor-produced absolute action chunks for safe Piper replay.

The output is a copy of the input trajectory with extra fields under
``forward_inputs``:

- actor_replay_action: actor output in training space [T, 1, H*14]
- actor_replay_env_action_absolute: actor output converted to absolute joints
- actor_replay_ref_action: ref action used by actor
- actor_replay_mse_to_ref/action: quick diagnostics

Then replay with:
  python scripts/replay_piper_pt_trajectory.py --trajectory OUT.pt \
    --action-key forward_inputs.actor_replay_env_action_absolute \
    --action-space absolute ...
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rlinf.models import get_model  # noqa: E402
from rlinf.models.embodiment.openpi.rl_token_policy import ForwardType  # noqa: E402
from rlinf.utils.offline_td3_visualization import (  # noqa: E402
    actions_to_absolute_joint_targets,
    load_action_norm_stats,
)


def _tensor(value: Any) -> torch.Tensor:
    return value.detach().cpu() if torch.is_tensor(value) else torch.as_tensor(value)


def _default_trajectory() -> str:
    base = Path(
        "/home/focal/shared_disk/users/kwj/RLinf/logs/"
        "20260609-07:59:29-realworld_piper_cubeinsert_pi05_eval_with_reward/demos"
    )
    candidates = sorted(base.glob("trajectory_*.pt"))
    return str(candidates[0]) if candidates else ""


def _load_cfg(args: argparse.Namespace):
    cfg = OmegaConf.load(args.config)
    if "actor" not in cfg:
        cfg = OmegaConf.create({"actor": {"model": cfg}})
    if "model" not in cfg.actor:
        cfg.actor.model = OmegaConf.create({})
    if "model_type" not in cfg.actor.model:
        default_model_cfg = OmegaConf.load(
            REPO_ROOT / "examples/embodiment/config/model/pi0_5_rl_token.yaml"
        )
        cfg.actor.model = OmegaConf.merge(default_model_cfg, cfg.actor.model)
    overrides = [
        "actor.model.rollout_control_mode=actor",
        f"actor.model.model_path={args.model_path}",
        f"actor.model.rl_token_path={args.rl_token_path}",
        f"actor.model.action_norm_stats_path={args.norm_stats}",
        "actor.model.freeze_rl_token=True",
        "actor.model.use_robot_state=True",
        "actor.model.critic_use_robot_state=True",
        "actor.model.critic_use_ref_action=False",
        "actor.model.action_space=normalized_delta",
        "actor.model.action_norm_std_floor=1.0",
    ]
    if args.actor_hidden_dims:
        overrides.append(f"actor.model.actor_hidden_dims=[{args.actor_hidden_dims}]")
    cli_cfg = OmegaConf.from_dotlist(overrides)
    return OmegaConf.merge(cfg, cli_cfg)


def _canonical_name(name: str) -> str:
    for prefix in ("module.", "_fsdp_wrapped_module."):
        if name.startswith(prefix):
            name = name[len(prefix) :]
    marker = "._fsdp_wrapped_module."
    if marker in name:
        name = name.split(marker, 1)[1]
    return name


def _normalize_state_keys(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefix_pairs = (
        ("target_actor_head.", "actor_head."),
        ("target_critic_head_1.", "critic_head_1."),
        ("target_critic_head_2.", "critic_head_2."),
        ("target_critic_rl_token_encoder.", "critic_rl_token_encoder."),
    )
    remapped = {}
    for name, value in state.items():
        out_name = _canonical_name(name)
        for old_prefix, new_prefix in prefix_pairs:
            if out_name.startswith(old_prefix):
                out_name = new_prefix + out_name[len(old_prefix) :]
                break
        remapped[out_name] = value
    return remapped


def _load_actor_critic(model: torch.nn.Module, checkpoint: str, device: torch.device):
    payload = torch.load(checkpoint, map_location=device)
    state = payload.get("model", payload)
    state = _normalize_state_keys(state)
    params = {_canonical_name(name): param for name, param in model.named_parameters()}
    buffers = {_canonical_name(name): buf for name, buf in model.named_buffers()}
    loaded = []
    skipped = []
    with torch.no_grad():
        for name, value in state.items():
            target = params.get(name, buffers.get(name, None))
            if target is None:
                skipped.append((name, "missing"))
                continue
            if tuple(target.shape) != tuple(value.shape):
                skipped.append(
                    (name, f"shape {tuple(value.shape)}->{tuple(target.shape)}")
                )
                continue
            target.copy_(value.to(device=target.device, dtype=target.dtype))
            loaded.append(name)
    print(f"Loaded {len(loaded)} actor/critic tensor(s) from {checkpoint}")
    if skipped:
        print("Skipped:", skipped[:12], "..." if len(skipped) > 12 else "")


def _resolve_checkpoint(path: str | Path) -> Path:
    checkpoint = Path(path).expanduser()
    if checkpoint.is_file():
        return checkpoint
    candidates = [
        checkpoint / "actor_critic.pt",
        checkpoint / "actor" / "actor_critic.pt",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Cannot find actor_critic.pt under {checkpoint}. Tried: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def _get_ref_action(trajectory: dict[str, Any], fallback: torch.Tensor) -> torch.Tensor:
    curr_obs = trajectory.get("curr_obs", {})
    if isinstance(curr_obs, dict) and "ref_action" in curr_obs:
        return _tensor(curr_obs["ref_action"]).float()
    forward_inputs = trajectory.get("forward_inputs", {})
    if isinstance(forward_inputs, dict) and "ref_action" in forward_inputs:
        return _tensor(forward_inputs["ref_action"]).float()
    return fallback.float()


def _get_obs_tensor(
    trajectory: dict[str, Any],
    key: str,
    *,
    length: int,
    batch_index: int,
) -> torch.Tensor | None:
    curr_obs = trajectory.get("curr_obs", {})
    if not isinstance(curr_obs, dict) or key not in curr_obs:
        return None
    value = _tensor(curr_obs[key])[:length]
    if value.dim() >= 2 and value.shape[1] > batch_index:
        value = value[:, batch_index]
    elif value.dim() >= 2 and value.shape[1] == 1:
        value = value[:, 0]
    return value


def _task_descriptions(
    trajectory: dict[str, Any], length: int, default: str
) -> list[str]:
    for container_key in ("curr_obs", "forward_inputs"):
        container = trajectory.get(container_key, {})
        if isinstance(container, dict) and "task_descriptions" in container:
            value = container["task_descriptions"]
            if isinstance(value, str):
                return [value] * length
            if isinstance(value, (list, tuple)):
                if len(value) >= length:
                    return [str(x) for x in value[:length]]
                if len(value) == 1:
                    return [str(value[0])] * length
    return [default] * length


def _absolute_chunks_from_actions(
    actions: np.ndarray,
    states: np.ndarray,
    *,
    norm_stats: str,
    std_floor: float,
) -> np.ndarray:
    mean, std = load_action_norm_stats(norm_stats, std_floor)
    low = np.tile(np.array([-2.618, 0.0, -2.967, -1.745, -1.22, -2.0944, 0.0]), 2)
    high = np.tile(np.array([2.618, 3.14, 0.0, 1.745, 1.22, 2.0944, 1.0]), 2)
    out = []
    for action, state in zip(actions, states):
        out.append(
            actions_to_absolute_joint_targets(
                action,
                state,
                low,
                high,
                "normalized_delta",
                0.01,
                mean,
                std,
                clip_to_joint_limits=False,
            )
        )
    return np.asarray(out, dtype=np.float32)


def _copy_with_tensors(payload: Any) -> Any:
    if torch.is_tensor(payload):
        return payload.detach().cpu().clone()
    if isinstance(payload, dict):
        return {key: _copy_with_tensors(value) for key, value in payload.items()}
    if isinstance(payload, (list, tuple)):
        return type(payload)(_copy_with_tensors(value) for value in payload)
    return copy.deepcopy(payload)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run actor on an offline Piper trajectory and export replayable absolute chunks."
    )
    parser.add_argument("--trajectory", default=_default_trajectory())
    parser.add_argument(
        "--checkpoint",
        default=(
            "/home/focal/shared_disk/users/kwj/RLinf/train_logs/"
            "dense_lerobot_last20_direct_bc_1000_clean_20260618_060531/"
            "dense_lerobot_last20_direct_bc_1000_clean_20260618_060531/"
            "checkpoints/global_step_1000"
        ),
        help="Path to actor_critic.pt, actor/actor_critic.pt, or a global_step_* directory.",
    )
    parser.add_argument(
        "--config",
        default=(
            "/home/focal/shared_disk/users/kwj/RLinf/train_logs/"
            "dense_lerobot_last20_direct_bc_1000_clean_20260618_060531/"
            "tensorboard/config.yaml"
        ),
    )
    parser.add_argument(
        "--model-path",
        default="/home/focal/shared_disk/users/kwj/weight/openpi/pi05_cube_insert",
    )
    parser.add_argument(
        "--rl-token-path",
        default="/home/focal/shared_disk/users/kwj/weight/rltoken/pi05_cube_insert/pi05_cube_insert/10000",
    )
    parser.add_argument(
        "--norm-stats",
        default="/home/focal/shared_disk/users/kwj/weight/openpi/pi05_cube_insert/norm_stats.json",
    )
    parser.add_argument("--actor-hidden-dims", default=None, help="Example: 512,256")
    parser.add_argument("--max-chunks", type=int, default=-1)
    parser.add_argument("--batch-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--task-description", default="insert the cube")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    trajectory_path = Path(args.trajectory)
    if not trajectory_path.is_file():
        raise FileNotFoundError(trajectory_path)
    checkpoint = _resolve_checkpoint(args.checkpoint)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg = _load_cfg(args)
    model = get_model(cfg.actor.model)
    model.to(device)
    model.eval()
    _load_actor_critic(model, str(checkpoint), device)

    trajectory = torch.load(trajectory_path, map_location="cpu", weights_only=False)
    length = int(trajectory["actions"].shape[0])
    if args.max_chunks and args.max_chunks > 0:
        length = min(length, int(args.max_chunks))

    actions = _tensor(trajectory["actions"])[:length].float()
    ref_action = _get_ref_action(trajectory, actions)[:length].float()
    states = _tensor(trajectory["curr_obs"]["states"])[:length].float()
    if actions.dim() == 3:
        actions_flat = actions[:, args.batch_index]
    else:
        actions_flat = actions.reshape(length, -1)
    if ref_action.dim() == 3:
        ref_flat = ref_action[:, args.batch_index]
    else:
        ref_flat = ref_action.reshape(length, -1)
    if states.dim() == 3:
        states_flat = states[:, args.batch_index]
    else:
        states_flat = states.reshape(length, -1)

    horizon = int(cfg.actor.model.num_action_chunks)
    action_dim = int(cfg.actor.model.action_dim)
    actor_chunks = []
    ref_chunks = []
    actor_abs_chunks = []
    ref_abs_chunks = []
    visual_chunks = []
    q_min = []
    curr_obs = trajectory.get("curr_obs", {})
    visual_latent = None
    if isinstance(curr_obs, dict) and "visual_latent" in curr_obs:
        visual_latent = _tensor(curr_obs["visual_latent"])[:length]
        if visual_latent.dim() == 4:
            visual_latent = visual_latent[:, args.batch_index]
    main_images = _get_obs_tensor(
        trajectory, "main_images", length=length, batch_index=args.batch_index
    )
    extra_view_images = _get_obs_tensor(
        trajectory, "extra_view_images", length=length, batch_index=args.batch_index
    )
    task_descriptions = _task_descriptions(
        trajectory, length=length, default=args.task_description
    )

    with torch.no_grad():
        batch_size = max(1, int(args.batch_size))
        for start in range(0, length, batch_size):
            end = min(length, start + batch_size)
            rs = states_flat[start:end].to(
                device=device, dtype=next(model.parameters()).dtype
            )
            ref = ref_flat[start:end].to(
                device=device, dtype=next(model.parameters()).dtype
            )
            ref = ref.reshape(ref.shape[0], horizon, action_dim)
            if visual_latent is not None:
                vf = visual_latent[start:end].to(
                    device=device, dtype=next(model.parameters()).dtype
                )
                out, aux = model(
                    forward_type=ForwardType.TD3,
                    mode="actor",
                    visual_feat=vf,
                    robot_state=rs,
                    ref_action=ref,
                    ref_action_dropout_p=0.0,
                    use_target=False,
                    compute_recon_loss=False,
                )
                actor_env = model._training_action_to_absolute(out, rs)
                ref_env = model._training_action_to_absolute(ref, rs)
                visual_out = vf
            else:
                if main_images is None:
                    raise KeyError(
                        "Trajectory has no curr_obs.visual_latent and no curr_obs.main_images; "
                        "cannot run actor replay export."
                    )
                env_obs = {
                    "states": rs,
                    "main_images": main_images[start:end].to(device=device),
                    "extra_view_images": (
                        extra_view_images[start:end].to(device=device)
                        if extra_view_images is not None
                        else None
                    ),
                    "task_descriptions": task_descriptions[start:end],
                }
                actor_env, result = model.predict_action_batch(
                    env_obs=env_obs, mode="eval"
                )
                forward_inputs = result["forward_inputs"]
                out = forward_inputs.get(
                    "actor_action", forward_inputs.get("model_action")
                )
                if out is None:
                    raise KeyError(
                        "predict_action_batch did not return actor_action/model_action"
                    )
                out = out.to(device=device, dtype=rs.dtype).reshape(
                    -1, horizon, action_dim
                )
                ref_from_model = forward_inputs.get("ref_action", None)
                if ref_from_model is not None:
                    ref = ref_from_model.to(device=device, dtype=rs.dtype).reshape(
                        -1, horizon, action_dim
                    )
                ref_env_from_model = forward_inputs.get("ref_env_action_absolute", None)
                ref_env = (
                    ref_env_from_model.to(device=device, dtype=rs.dtype).reshape(
                        -1, horizon, action_dim
                    )
                    if ref_env_from_model is not None
                    else model._training_action_to_absolute(ref, rs)
                )
                actor_env = actor_env.to(device=device, dtype=rs.dtype).reshape(
                    -1, horizon, action_dim
                )
                visual_out = forward_inputs.get("visual_latent", None)
                if visual_out is not None:
                    visual_out = visual_out.to(device=device, dtype=rs.dtype)
                aux = model(
                    forward_type=ForwardType.TD3,
                    mode="actor",
                    visual_feat=visual_out if visual_out is not None else env_obs,
                    robot_state=rs,
                    ref_action=ref,
                    ref_action_dropout_p=0.0,
                    use_target=False,
                    compute_recon_loss=False,
                )[1]
            q1, q2 = model(
                forward_type=ForwardType.TD3_Q,
                rl_state=aux.get("critic_rl_state", aux["rl_state"]),
                action=out,
            )
            actor_chunks.append(out.detach().float().cpu())
            ref_chunks.append(ref.detach().float().cpu())
            actor_abs_chunks.append(actor_env.detach().float().cpu())
            ref_abs_chunks.append(ref_env.detach().float().cpu())
            if visual_out is not None:
                visual_chunks.append(visual_out.detach().float().cpu())
            q_min.append(torch.minimum(q1, q2).detach().float().cpu())
    actor_action = torch.cat(actor_chunks, dim=0)
    ref_action_model = torch.cat(ref_chunks, dim=0)
    actor_abs_tensor = torch.cat(actor_abs_chunks, dim=0)
    ref_abs_tensor = torch.cat(ref_abs_chunks, dim=0)
    q_min_tensor = torch.cat(q_min, dim=0).reshape(length, 1)

    actor_np = actor_action.numpy().reshape(length, horizon, action_dim)
    ref_np = ref_action_model.numpy().reshape(length, horizon, action_dim)
    data_np = actions_flat[:length].numpy().reshape(length, horizon, action_dim)
    states_np = states_flat[:length].numpy()
    actor_abs = actor_abs_tensor.numpy().reshape(length, horizon, action_dim)
    ref_abs = ref_abs_tensor.numpy().reshape(length, horizon, action_dim)
    data_abs = _absolute_chunks_from_actions(
        data_np,
        states_np,
        norm_stats=args.norm_stats,
        std_floor=float(cfg.actor.model.action_norm_std_floor),
    )

    actor_ref_mse = ((actor_np - ref_np) ** 2).mean(axis=(1, 2))
    actor_data_mse = ((actor_np - data_np) ** 2).mean(axis=(1, 2))
    actor_ref_abs_max = np.abs(actor_abs - ref_abs).max(axis=(1, 2))
    actor_data_abs_max = np.abs(actor_abs - data_abs).max(axis=(1, 2))

    out_payload = _copy_with_tensors(trajectory)
    fi = out_payload.setdefault("forward_inputs", {})
    fi["actor_replay_action"] = actor_action.reshape(length, 1, -1).cpu()
    fi["actor_replay_ref_action"] = ref_action_model.reshape(length, 1, -1).cpu()
    fi["actor_replay_env_action_absolute"] = torch.from_numpy(
        actor_abs.reshape(length, 1, -1)
    ).float()
    fi["actor_replay_ref_env_action_absolute"] = torch.from_numpy(
        ref_abs.reshape(length, 1, -1)
    ).float()
    fi["actor_replay_data_env_action_absolute"] = torch.from_numpy(
        data_abs.reshape(length, 1, -1)
    ).float()
    fi["actor_replay_q_min"] = q_min_tensor.cpu()
    fi["actor_replay_mse_to_ref"] = (
        torch.from_numpy(actor_ref_mse).float().reshape(length, 1, 1)
    )
    fi["actor_replay_mse_to_data"] = (
        torch.from_numpy(actor_data_mse).float().reshape(length, 1, 1)
    )
    if visual_chunks:
        out_payload.setdefault("curr_obs", {})["visual_latent"] = torch.cat(
            visual_chunks, dim=0
        ).reshape(length, 1, *visual_chunks[0].shape[1:])

    if args.output:
        output = Path(args.output)
    else:
        output_dir = Path(
            "/home/focal/shared_disk/users/kwj/RLinf/debug_actor_replay_exports"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / (
            trajectory_path.stem + "_" + checkpoint.parent.name + "_actor_replay.pt"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_payload, output)

    print(f"trajectory: {trajectory_path}")
    print(f"checkpoint: {checkpoint}")
    print(f"output: {output}")
    print(f"chunks: {length}, horizon: {horizon}, action_dim: {action_dim}")
    print(
        "actor normalized MSE to ref mean/max: {:.6g}/{:.6g}".format(
            float(actor_ref_mse.mean()), float(actor_ref_mse.max())
        )
    )
    print(
        "actor normalized MSE to data mean/max: {:.6g}/{:.6g}".format(
            float(actor_data_mse.mean()), float(actor_data_mse.max())
        )
    )
    print(
        "actor absolute max_abs to ref mean/max: {:.6g}/{:.6g}".format(
            float(actor_ref_abs_max.mean()), float(actor_ref_abs_max.max())
        )
    )
    print(
        "actor absolute max_abs to data mean/max: {:.6g}/{:.6g}".format(
            float(actor_data_abs_max.mean()), float(actor_data_abs_max.max())
        )
    )
    print("Replay command:")
    print(
        "python scripts/replay_piper_pt_trajectory.py "
        f"--trajectory {output} "
        "--action-key forward_inputs.actor_replay_env_action_absolute "
        "--action-space absolute --chunks all --substeps all "
        "--go-to-start --go-to-start-min-steps 20 --execute"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
