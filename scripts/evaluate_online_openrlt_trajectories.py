#!/usr/bin/env python3
# Copyright 2026 The GIGA Authors.
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

"""Evaluate online OpenRLT right-arm trajectories with an actor/critic checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from safetensors.torch import load_file as load_safetensors


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


sys.path.insert(0, str(_repo_root()))

from rlinf.models.embodiment.openpi.rl_token_policy import (  # noqa: E402
    OpenPiRLTokenConfig,
    OpenPiRLTokenPolicy,
)
from rlinf.utils.offline_td3_visualization import (  # noqa: E402
    SimpleUrdfKinematics,
    compute_chunk_mc_returns,
    plot_critic_timeline_with_images,
)

RIGHT_BASE_OFFSET = np.array([0.0, -0.22, 0.0], dtype=np.float64)


def _trajectory_number(path: Path) -> int:
    match = re.search(r"trajectory_(\d+)_", path.name)
    return int(match.group(1)) if match else -1


def latest_checkpoint(log_dir: Path) -> Path:
    candidates = []
    root = log_dir / "piper_td3_rl_token_online" / "checkpoints"
    for item in root.glob("global_step_*"):
        match = re.search(r"global_step_(\d+)$", item.name)
        ckpt = item / "actor" / "actor_critic.pt"
        if match and ckpt.exists():
            candidates.append((int(match.group(1)), ckpt))
    if not candidates:
        raise FileNotFoundError(f"No actor_critic.pt found under {root}")
    return max(candidates, key=lambda item: item[0])[1]


def build_policy(
    checkpoint_path: Path,
    rl_token_path: Path,
    norm_stats_path: Path,
    device: torch.device,
    action_horizon: int,
) -> OpenPiRLTokenPolicy:
    metadata = json.loads((rl_token_path / "metadata.json").read_text())
    rl_cfg = metadata["rl_config"]
    config = OpenPiRLTokenConfig(
        hidden_dim=int(rl_cfg["hidden_dim"]),
        rl_token_dim=int(rl_cfg["rl_token_dim"]),
        rl_token_encoder_layers=int(rl_cfg["encoder_layers"]),
        rl_token_decoder_layers=int(rl_cfg["decoder_layers"]),
        rl_token_num_heads=int(rl_cfg["num_heads"]),
        rl_token_max_seq_len=int(rl_cfg["max_seq_len"]),
        rl_token_dropout=float(rl_cfg.get("dropout", 0.1)),
        num_image_tokens=768,
        prefix_feature_type="image_only",
        robot_state_dim=6,
        actor_head_type="openrlt_mlp",
        critic_head_type="openrlt_mlp",
        openrlt_z_proj_dim=256,
        openrlt_state_proj_dim=64,
        openrlt_action_proj_dim=256,
        openrlt_hidden_dim=256,
        openrlt_num_layers=2,
        action_horizon=int(action_horizon),
        action_dim=6,
        actor_output_bound=1.2,
        actor_residual_ref=True,
        actor_residual_scale=1.0,
        use_robot_state=True,
        critic_use_robot_state=True,
        critic_use_ref_action=False,
        critic_use_rl_token=True,
        action_space="normalized_delta",
        action_norm_stats_path=str(norm_stats_path),
        action_norm_std_floor=1.0,
        env_action_dim=14,
        controlled_action_indices=[7, 8, 9, 10, 11, 12],
    )
    policy = OpenPiRLTokenPolicy(config)

    rlt_state = load_safetensors(str(rl_token_path / "model.safetensors"), device="cpu")
    rlt_state = {f"rl_token_autoencoder.{k}": v for k, v in rlt_state.items()}
    missing, unexpected = policy.load_state_dict(rlt_state, strict=False)
    unexpected = [k for k in unexpected if not k.startswith("rl_token_autoencoder")]
    if unexpected:
        raise RuntimeError(f"Unexpected RLT keys: {unexpected[:8]}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_state = checkpoint["model"]
    missing, unexpected = policy.load_state_dict(model_state, strict=False)
    unexpected = [k for k in unexpected if not k.startswith("target_")]
    if unexpected:
        raise RuntimeError(f"Unexpected actor/critic keys: {unexpected[:8]}")

    policy.to(device)
    policy.eval()
    return policy


def build_full_policy_from_model_config(
    checkpoint_path: Path,
    model_config_path: Path,
    rl_token_path: Path | None,
    norm_stats_path: Path,
    device: torch.device,
) -> OpenPiRLTokenPolicy:
    from rlinf.models import get_model

    source_cfg = OmegaConf.load(model_config_path)
    model_cfg = OmegaConf.create(
        OmegaConf.to_container(source_cfg.actor.model, resolve=True)
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    train_cfg = checkpoint.get("config", {})
    model_cfg.model_type = "openpi_rl_token"
    model_cfg.precision = model_cfg.get("precision", None)
    model_cfg.is_lora = bool(model_cfg.get("is_lora", False))
    if train_cfg.get("model_path") is not None:
        model_cfg.model_path = str(train_cfg["model_path"])
    resolved_rl_token_path = train_cfg.get("rl_token_path") or rl_token_path
    if resolved_rl_token_path is not None:
        model_cfg.rl_token_path = str(resolved_rl_token_path)
    elif hasattr(model_cfg, "rl_token_path"):
        model_cfg.rl_token_path = None
    model_cfg.rl_token_source = train_cfg.get(
        "rl_token_source", model_cfg.get("rl_token_source", "autoencoder")
    )
    model_cfg.prefix_feature_type = train_cfg.get(
        "prefix_feature_type", model_cfg.get("prefix_feature_type", "image_only")
    )
    model_cfg.num_image_tokens = int(
        train_cfg.get("num_image_tokens", model_cfg.get("num_image_tokens", 768))
    )
    model_cfg.actor_train_prefix_token_linear = bool(
        train_cfg.get("actor_train_prefix_token_linear", False)
    )
    model_cfg.critic_train_prefix_token_linear = bool(
        train_cfg.get("critic_train_prefix_token_linear", False)
    )
    model_cfg.action_horizon = int(
        train_cfg.get("chunk_len", model_cfg.get("num_action_chunks", 10))
    )
    model_cfg.num_action_chunks = int(model_cfg.action_horizon)
    model_cfg.action_dim = int(train_cfg.get("action_dim", 6))
    model_cfg.env_action_dim = int(model_cfg.get("env_action_dim", 14))
    model_cfg.robot_state_dim = int(train_cfg.get("proprio_dim", 6))
    model_cfg.actor_head_type = "openrlt_mlp"
    model_cfg.critic_head_type = "openrlt_mlp"
    model_cfg.openrlt_z_proj_dim = 256
    model_cfg.openrlt_state_proj_dim = 64
    model_cfg.openrlt_action_proj_dim = 256
    model_cfg.openrlt_hidden_dim = 256
    model_cfg.openrlt_num_layers = 2
    model_cfg.actor_output_bound = 1.2
    model_cfg.actor_residual_ref = bool(train_cfg.get("actor_residual_ref", True))
    model_cfg.actor_residual_scale = float(train_cfg.get("actor_residual_scale", 1.0))
    model_cfg.use_robot_state = True
    model_cfg.critic_use_robot_state = True
    model_cfg.critic_use_ref_action = False
    model_cfg.critic_use_rl_token = True
    model_cfg.action_space = "normalized_delta"
    model_cfg.action_norm_stats_path = str(norm_stats_path)
    model_cfg.action_norm_std_floor = 1.0
    model_cfg.controlled_action_indices = [7, 8, 9, 10, 11, 12]
    model_cfg.freeze_rl_token = True
    model_cfg.critic_train_rl_token_encoder = False
    model_cfg.critic_separate_rl_token_encoder = False

    policy = get_model(model_cfg)
    missing, unexpected = policy.load_state_dict(checkpoint["model"], strict=False)
    unexpected = [k for k in unexpected if not k.startswith("target_")]
    if unexpected:
        raise RuntimeError(f"Unexpected actor/critic keys: {unexpected[:8]}")
    policy.to(device)
    policy.eval()
    return policy


def right_tcp(joints: np.ndarray, kin: SimpleUrdfKinematics) -> np.ndarray:
    points = []
    for q in np.asarray(joints, dtype=np.float64).reshape(-1, 7):
        links, _ = kin.fk_positions(q, tip_link="gripper")
        points.append(links[-1] + RIGHT_BASE_OFFSET)
    return np.stack(points, axis=0)


def tcp_error_metrics(
    prefix: str,
    predicted_tcp: np.ndarray,
    target_tcp: np.ndarray,
    valid_mask: np.ndarray,
) -> dict[str, float]:
    predicted = np.asarray(predicted_tcp, dtype=np.float64)
    target = np.asarray(target_tcp, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)
    err = predicted[valid] - target[valid]
    if err.size == 0:
        return {
            f"{prefix}_tcp_mse_m2": 0.0,
            f"{prefix}_tcp_rmse_m": 0.0,
            f"{prefix}_tcp_l2_mean_m": 0.0,
            f"{prefix}_tcp_l2_max_m": 0.0,
        }
    sq = err**2
    l2 = np.linalg.norm(err, axis=-1)
    return {
        f"{prefix}_tcp_mse_m2": float(np.mean(sq)),
        f"{prefix}_tcp_rmse_m": float(np.sqrt(np.mean(sq))),
        f"{prefix}_tcp_l2_mean_m": float(np.mean(l2)),
        f"{prefix}_tcp_l2_max_m": float(np.max(l2)),
    }


def chunk_tcp(joints: np.ndarray, kin: SimpleUrdfKinematics) -> np.ndarray:
    shape = np.asarray(joints).shape[:-1]
    return right_tcp(joints, kin).reshape(*shape, 3)


def plot_right_chunks(
    out_path: Path,
    state_q: np.ndarray,
    pt_chunks: list[np.ndarray],
    gt_chunks: list[np.ndarray],
    actor_chunks: list[np.ndarray],
    kin: SimpleUrdfKinematics,
) -> None:
    fig = plt.figure(figsize=(8.0, 6.2), dpi=150)
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    state_tcp = right_tcp(state_q, kin)
    ax.scatter(
        state_tcp[:, 0],
        state_tcp[:, 1],
        state_tcp[:, 2],
        s=11,
        c="#222222",
        alpha=0.45,
        label="right robot state",
    )
    ax.scatter(
        state_tcp[0, 0],
        state_tcp[0, 1],
        state_tcp[0, 2],
        s=58,
        c="#2f8f46",
        marker="o",
        label="trajectory start",
    )
    ax.scatter(
        state_tcp[-1, 0],
        state_tcp[-1, 1],
        state_tcp[-1, 2],
        s=78,
        c="#b3472f",
        marker="x",
        label="trajectory end",
    )
    arrays = [state_tcp]
    rows = [
        ("PT absolute", pt_chunks, "#6f2b8c", "-", 1.8, 0.45),
        ("GT reconstructed", gt_chunks, "#d17a22", "--", 1.4, 0.75),
        ("Actor reconstructed", actor_chunks, "#2a6fbb", ":", 1.8, 0.9),
    ]
    for label, chunks, color, style, width, alpha in rows:
        for idx, q in enumerate(chunks):
            pts = right_tcp(q, kin)
            arrays.append(pts)
            ax.plot(
                pts[:, 0],
                pts[:, 1],
                pts[:, 2],
                color=color,
                linestyle=style,
                linewidth=width,
                alpha=alpha,
                label=label if idx == 0 else None,
            )
    pts = np.concatenate(arrays, axis=0)
    center = pts.mean(axis=0)
    radius = max(float((pts.max(axis=0) - pts.min(axis=0)).max()) / 2.0, 1e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_title("right TCP action chunks")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_q_values(
    out_path: Path,
    q_data: np.ndarray,
    q_actor: np.ndarray,
    mc_return: np.ndarray,
    mse: np.ndarray,
    rewards: np.ndarray,
    title: str,
) -> None:
    x = np.arange(len(q_data))
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 6.4), dpi=150, sharex=True)
    axes[0].plot(
        x,
        q_data,
        marker="o",
        markersize=3.2,
        color="#222222",
        label="Q(s, data right action)",
    )
    axes[0].plot(
        x,
        q_actor,
        marker="o",
        markersize=3.2,
        color="#2a6fbb",
        label="Q(s, actor right action)",
    )
    axes[0].plot(
        x,
        mc_return,
        marker="s",
        markersize=3.0,
        linestyle="--",
        color="#2f8f46",
        label="MC return",
    )
    if np.any(rewards > 0):
        hit = rewards > 0
        axes[0].scatter(
            x[hit],
            mc_return[hit],
            marker="*",
            s=95,
            color="#c0392b",
            label="reward > 0",
        )
    axes[0].set_title(title)
    axes[0].set_ylabel("Q / return")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best", ncol=2)
    axes[1].plot(
        x,
        mse,
        marker="o",
        markersize=3.0,
        color="#b3472f",
        label="normalized action MSE",
    )
    axes[1].set_xlabel("trajectory chunk index")
    axes[1].set_ylabel("MSE")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def first_step_images(traj: dict, n: int) -> np.ndarray | None:
    fwd = traj.get("forward_inputs", {})
    main = fwd.get("substep_main_images")
    extra = fwd.get("substep_extra_view_images")
    if main is None or extra is None:
        return None
    main_np = main[:n, 0, 0].cpu().numpy()
    extra_np = extra[:n, 0, 0].cpu().numpy()
    return np.stack([main_np, extra_np[:, 0], extra_np[:, 1]], axis=1)


def _select_obs_at_chunk(curr: dict, rel: int, task_description: str) -> dict:
    env_obs = {}
    for src_key, out_key in (
        ("main_images", "main_images"),
        ("extra_view_images", "extra_view_images"),
        ("states", "states"),
        ("task_descriptions", "task_descriptions"),
    ):
        if src_key not in curr:
            continue
        value = curr[src_key]
        if torch.is_tensor(value):
            env_obs[out_key] = value[rel, 0].unsqueeze(0).contiguous()
        elif isinstance(value, list):
            env_obs[out_key] = value[rel] if rel < len(value) else value[-1]
        else:
            env_obs[out_key] = value
    if "task_descriptions" not in env_obs:
        env_obs["task_descriptions"] = [task_description]
    return env_obs


def _policy_uses_image_last_linear(policy: OpenPiRLTokenPolicy) -> bool:
    source = getattr(policy, "rl_token_source", None)
    if source is None:
        source = getattr(getattr(policy, "config", None), "rl_token_source", None)
    return str(source) == "image_last_linear"


def load_prefix_feature_cache(path: Path | None) -> dict | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"feature_cache not found: {path}")
    try:
        payload = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    if "prefix_tokens" not in payload:
        raise KeyError(f"feature_cache does not contain `prefix_tokens`: {path}")
    lookup: dict[tuple[str, int], int] = {}
    for idx, meta in enumerate(payload.get("metas", [])):
        rel = int(meta.get("rel_chunk", meta.get("chunk_index", -1)))
        raw_path = str(meta.get("path", ""))
        if rel < 0 or not raw_path:
            continue
        meta_path = Path(raw_path)
        lookup[(str(meta_path), rel)] = int(idx)
        lookup[(meta_path.name, rel)] = int(idx)
    return {
        "path": path,
        "prefix_tokens": payload["prefix_tokens"],
        "lookup": lookup,
    }


def cached_prefix_for_trajectory(
    path: Path, n: int, feature_cache: dict | None
) -> torch.Tensor | None:
    if feature_cache is None:
        return None
    tokens = feature_cache["prefix_tokens"]
    lookup = feature_cache["lookup"]
    rows = []
    for rel in range(n):
        idx = lookup.get((str(path), rel), lookup.get((path.name, rel)))
        if idx is None:
            return None
        rows.append(tokens[int(idx)].float())
    return torch.stack(rows, dim=0)


def visual_latent_for_trajectory(
    path: Path,
    n: int,
    fwd: dict,
    curr: dict,
    policy: OpenPiRLTokenPolicy,
    device: torch.device,
    task_description: str,
    feature_cache: dict | None = None,
) -> torch.Tensor:
    if _policy_uses_image_last_linear(policy):
        cached = cached_prefix_for_trajectory(path, n, feature_cache)
        if cached is not None:
            print(
                f"[eval] using cached prefix_tokens from {feature_cache['path']}",
                flush=True,
            )
            return cached
    visual = fwd.get("visual_latent", curr.get("visual_latent", None))
    if visual is not None:
        visual = visual[:, 0].float()
        if not _policy_uses_image_last_linear(policy):
            return visual
        expected_len = int(getattr(policy.config, "num_image_tokens", 768)) + 1
        if visual.dim() >= 3 and visual.shape[1] >= expected_len:
            return visual
        print(
            "[eval] stored visual_latent is image-only; recomputing full prefix "
            "for image_last_linear",
            flush=True,
        )
    rows = []
    with torch.no_grad():
        for rel in range(n):
            env_obs = _select_obs_at_chunk(curr, rel, task_description)
            env_obs = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in env_obs.items()
            }
            prefix_output, _, _ = policy._build_prefix_cache_from_obs(env_obs)
            if _policy_uses_image_last_linear(policy):
                rows.append(prefix_output.squeeze(0).cpu())
            else:
                rows.append(
                    policy._select_prefix_features(prefix_output).squeeze(0).cpu()
                )
    return torch.stack(rows, dim=0).float()


def evaluate_trajectory(
    path: Path,
    out_dir: Path,
    policy: OpenPiRLTokenPolicy,
    kin: SimpleUrdfKinematics,
    device: torch.device,
    gamma: float,
    batch_size: int,
    action_horizon: int,
    task_description: str,
    feature_cache: dict | None = None,
) -> dict:
    traj = torch.load(path, map_location="cpu", weights_only=False)
    fwd = traj.get("forward_inputs", {})
    curr = traj["curr_obs"]
    states = curr["states"][:, 0].reshape(-1, 14)
    rewards_full = traj["rewards"][:, 0].float()
    dones_full = traj["dones"][:, 0].bool()
    n = int(states.shape[0])
    visual = visual_latent_for_trajectory(
        path, n, fwd, curr, policy, device, task_description, feature_cache
    )
    valid_mask = fwd.get("chunk_valid_mask")
    if valid_mask is None:
        valid = torch.ones(n, action_horizon, dtype=torch.bool)
    else:
        valid = valid_mask[:, 0].bool()

    if "executed_env_action_absolute" in fwd:
        pt_abs_full_tensor = (
            fwd["executed_env_action_absolute"][:, 0]
            .reshape(n, action_horizon, 14)
            .float()
        )
        data_action_source = "executed_env_action_absolute->normalized_delta"
        converted_actions = []
        with torch.no_grad():
            for start in range(0, n, batch_size):
                end = min(n, start + batch_size)
                converted_actions.append(
                    policy._absolute_to_training_action(
                        pt_abs_full_tensor[start:end].to(device),
                        states[start:end].to(device=device, dtype=torch.float32),
                    )
                    .cpu()
                    .float()
                )
        actions = torch.cat(converted_actions, dim=0).reshape(n, action_horizon, 6)
    elif "env_action_absolute" in fwd:
        pt_abs_full_tensor = (
            fwd["env_action_absolute"][:, 0].reshape(n, action_horizon, 14).float()
        )
        data_action_source = "env_action_absolute->normalized_delta"
        converted_actions = []
        with torch.no_grad():
            for start in range(0, n, batch_size):
                end = min(n, start + batch_size)
                converted_actions.append(
                    policy._absolute_to_training_action(
                        pt_abs_full_tensor[start:end].to(device),
                        states[start:end].to(device=device, dtype=torch.float32),
                    )
                    .cpu()
                    .float()
                )
        actions = torch.cat(converted_actions, dim=0).reshape(n, action_horizon, 6)
    else:
        actions = (
            fwd.get("executed_action", traj["actions"])[:, 0]
            .reshape(-1, action_horizon, 6)
            .float()
        )
        pt_abs_full_tensor = None
        data_action_source = "executed_action"

    ref_action = actions.clone()

    stored_source = fwd.get("executed_action", traj["actions"])
    if int(stored_source.shape[-1]) == action_horizon * 6:
        stored_action = stored_source[:, 0].reshape(n, action_horizon, 6).float()
    else:
        stored_action = actions
    stored_action_mse_vs_executed_abs = float(
        ((stored_action - actions) ** 2).mean().item()
    )

    actor_chunks = []
    gt_abs_chunks = []
    q_data = []
    q_actor = []

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(n, start + batch_size)
            batch_visual = visual[start:end].to(device=device, dtype=torch.float32)
            batch_state = states[start:end].to(device=device, dtype=torch.float32)
            batch_ref = ref_action[start:end].to(device=device, dtype=torch.float32)
            batch_action = actions[start:end].to(device=device, dtype=torch.float32)
            actor_action, aux = policy._td3_actor_forward(
                visual_feat=batch_visual,
                robot_state=batch_state,
                ref_action=batch_ref,
                ref_action_dropout_p=0.0,
                use_target=False,
            )
            q1, q2 = policy._compute_q(
                aux["critic_rl_state"], batch_action, use_target=False
            )
            aq1, aq2 = policy._compute_q(
                aux["critic_rl_state"], actor_action, use_target=False
            )
            q_data.append(torch.minimum(q1, q2).reshape(-1).cpu())
            q_actor.append(torch.minimum(aq1, aq2).reshape(-1).cpu())
            actor_chunks.append(actor_action.cpu())
            gt_abs_chunks.append(
                policy._training_action_to_absolute(batch_action, batch_state).cpu()
            )
    actor_action = torch.cat(actor_chunks, dim=0)
    gt_abs = torch.cat(gt_abs_chunks, dim=0).numpy()
    q_data_np = torch.cat(q_data, dim=0).numpy()
    q_actor_np = torch.cat(q_actor, dim=0).numpy()
    action_np = actions.numpy()
    actor_np = actor_action.numpy()
    state_np = states.numpy()

    with torch.no_grad():
        actor_abs = []
        for start in range(0, n, batch_size):
            end = min(n, start + batch_size)
            actor_abs.append(
                policy._training_action_to_absolute(
                    actor_action[start:end].to(device),
                    states[start:end].to(device=device, dtype=torch.float32),
                )
                .cpu()
                .numpy()
            )
    actor_abs = np.concatenate(actor_abs, axis=0)

    if pt_abs_full_tensor is not None:
        pt_abs_full = pt_abs_full_tensor.numpy()
    else:
        pt_abs_full = np.zeros((n, action_horizon, 14), dtype=np.float32)
        pt_abs_full[..., 7:13] = gt_abs
        pt_abs_full[..., 13:14] = state_np[:, None, 13:14]
        pt_abs_full[..., :7] = state_np[:, None, :7]
    pt_right = pt_abs_full[..., 7:14]
    gt_abs_full = (
        gt_abs[..., 7:14]
        if gt_abs.shape[-1] == 14
        else np.concatenate([gt_abs, pt_right[..., 6:7]], axis=-1)
    )
    actor_abs_full = (
        actor_abs[..., 7:14]
        if actor_abs.shape[-1] == 14
        else np.concatenate([actor_abs, pt_right[..., 6:7]], axis=-1)
    )

    reward_chunk = rewards_full.max(dim=-1).values.numpy()
    done_chunk = dones_full.any(dim=-1).numpy()
    mc = compute_chunk_mc_returns(
        reward_chunk, done_chunk, gamma=gamma, action_horizon=action_horizon
    )
    mse = ((actor_np - action_np) ** 2).mean(axis=(1, 2))
    mse_matrix = (actor_np - action_np) ** 2

    pt_chunks = []
    gt_chunks = []
    actor_plot_chunks = []
    for idx in range(n):
        steps = int(valid[idx].sum().item())
        if steps <= 0:
            continue
        pt_chunks.append(pt_right[idx, :steps])
        gt_chunks.append(gt_abs_full[idx, :steps])
        actor_plot_chunks.append(actor_abs_full[idx, :steps])

    out_dir.mkdir(parents=True, exist_ok=True)
    title = f"{path.name}, reward_sum={float(rewards_full.sum()):.1f}"
    plot_q_values(
        out_dir / "q_values.png", q_data_np, q_actor_np, mc, mse, reward_chunk, title
    )
    images = first_step_images(traj, n)
    plot_critic_timeline_with_images(
        str(out_dir / "critic_timeline_images.png"),
        chunk_indices=np.arange(n),
        critic_q_data=q_data_np,
        critic_q_actor=q_actor_np,
        mc_return=mc,
        rewards=reward_chunk,
        images=images,
        title=title,
    )
    plot_right_chunks(
        out_dir / "right_tcp_chunks.png",
        state_np[:, 7:14],
        pt_chunks,
        gt_chunks,
        actor_plot_chunks,
        kin,
    )

    valid_np = valid.numpy()
    pt_gt_err = np.abs(pt_right[valid_np][..., :6] - gt_abs_full[valid_np][..., :6])
    actor_pt_err = np.abs(
        actor_abs_full[valid_np][..., :6] - pt_right[valid_np][..., :6]
    )
    pt_tcp = chunk_tcp(pt_right, kin)
    gt_tcp = chunk_tcp(gt_abs_full, kin)
    actor_tcp = chunk_tcp(actor_abs_full, kin)
    tcp_metrics = {
        **tcp_error_metrics("gt_pt", gt_tcp, pt_tcp, valid_np),
        **tcp_error_metrics("actor_pt", actor_tcp, pt_tcp, valid_np),
    }
    metrics = {
        "trajectory": str(path),
        "trajectory_name": path.name,
        "num_chunks": n,
        "valid_low_level_steps": int(valid_np.sum()),
        "reward_sum": float(rewards_full.sum().item()),
        "success": bool(rewards_full.sum().item() > 0),
        "intervention_low_level_entries": int(traj["intervene_flags"].sum().item())
        if "intervene_flags" in traj
        else 0,
        "data_action_source": data_action_source,
        "stored_norm_action_mse_vs_executed_abs": stored_action_mse_vs_executed_abs,
        "q_data_mean": float(np.mean(q_data_np)),
        "q_actor_mean": float(np.mean(q_actor_np)),
        "q_gap_actor_minus_data_mean": float(np.mean(q_actor_np - q_data_np)),
        "q_data_final": float(q_data_np[-1]),
        "q_actor_final": float(q_actor_np[-1]),
        "normalized_action_mse_mean": float(np.mean(mse)),
        "normalized_action_mse_final": float(mse[-1]),
        "pt_gt_right_abs_max_error": float(np.max(pt_gt_err))
        if pt_gt_err.size
        else 0.0,
        "pt_gt_right_abs_mean_error": float(np.mean(pt_gt_err))
        if pt_gt_err.size
        else 0.0,
        "actor_pt_right_abs_max_error": float(np.max(actor_pt_err))
        if actor_pt_err.size
        else 0.0,
        "actor_pt_right_abs_mean_error": float(np.mean(actor_pt_err))
        if actor_pt_err.size
        else 0.0,
        **tcp_metrics,
        "q_data": q_data_np.tolist(),
        "q_actor": q_actor_np.tolist(),
        "mc_return": mc.tolist(),
        "reward_chunk": reward_chunk.tolist(),
        "normalized_action_mse": mse.tolist(),
        "normalized_action_mse_matrix": mse_matrix.tolist(),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def plot_aggregate(out_path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    names = [r["trajectory_name"].split("_")[1] for r in rows]
    x = np.arange(len(rows))
    success = np.array([r["success"] for r in rows], dtype=bool)
    q_gap = np.array([r["q_gap_actor_minus_data_mean"] for r in rows], dtype=float)
    mse = np.array([r["normalized_action_mse_mean"] for r in rows], dtype=float)
    reward = np.array([r["reward_sum"] for r in rows], dtype=float)
    fig, axes = plt.subplots(
        3, 1, figsize=(max(12, len(rows) * 0.18), 8), dpi=150, sharex=True
    )
    colors = np.where(success, "#2f8f46", "#b3472f")
    axes[0].bar(x, q_gap, color=colors, alpha=0.85)
    axes[0].axhline(0.0, color="#222222", linewidth=0.8)
    axes[0].set_ylabel("mean Q_actor - Q_data")
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[1].bar(x, mse, color="#2a6fbb", alpha=0.75)
    axes[1].set_ylabel("mean normalized MSE")
    axes[1].grid(True, axis="y", alpha=0.25)
    axes[2].bar(x, reward, color=colors, alpha=0.85)
    axes[2].set_ylabel("reward sum")
    axes[2].set_xlabel("trajectory index")
    axes[2].grid(True, axis="y", alpha=0.25)
    stride = max(1, int(np.ceil(len(rows) / 30)))
    axes[2].set_xticks(x[::stride])
    axes[2].set_xticklabels(names[::stride], rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def checkpoint_action_horizon(checkpoint_path: Path) -> int:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    return int(cfg.get("chunk_len", cfg.get("action_horizon", 10)))


def select_trajectories(
    trajectories: list[Path],
    *,
    sample_success: int,
    sample_failure: int,
    seed: int,
) -> list[Path]:
    if sample_success <= 0 and sample_failure <= 0:
        return trajectories
    success_paths: list[Path] = []
    failure_paths: list[Path] = []
    for path in trajectories:
        traj = torch.load(path, map_location="cpu", weights_only=False)
        reward_sum = float(torch.as_tensor(traj["rewards"]).float().sum().item())
        (success_paths if reward_sum > 0 else failure_paths).append(path)
    rng = random.Random(int(seed))
    selected: list[Path] = []
    if sample_success > 0:
        selected.extend(
            rng.sample(success_paths, min(int(sample_success), len(success_paths)))
        )
    if sample_failure > 0:
        selected.extend(
            rng.sample(failure_paths, min(int(sample_failure), len(failure_paths)))
        )
    return sorted(selected, key=_trajectory_number)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--model-config", type=Path)
    parser.add_argument("--feature-cache", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--rl-token-path", type=Path)
    parser.add_argument("--norm-stats-path", type=Path, required=True)
    parser.add_argument("--urdf-path", type=Path, required=True)
    parser.add_argument("--gamma", type=float, default=0.89)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sample-success", type=int, default=0)
    parser.add_argument("--sample-failure", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--task-description", default="peg and insertion")
    args = parser.parse_args()

    data_dir = args.data_dir or (args.log_dir / "replay_buffer" / "rank_0")
    trajectories = sorted(data_dir.glob("trajectory_*.pt"), key=_trajectory_number)
    if args.limit > 0:
        trajectories = trajectories[: args.limit]
    if not trajectories:
        raise FileNotFoundError(f"No trajectory_*.pt found in {data_dir}")

    checkpoint = args.checkpoint or latest_checkpoint(args.log_dir)
    trajectories = select_trajectories(
        trajectories,
        sample_success=args.sample_success,
        sample_failure=args.sample_failure,
        seed=args.seed,
    )
    step_match = re.search(r"global_step_(\d+)", str(checkpoint))
    step = step_match.group(1) if step_match else "latest"
    output_dir = args.output_dir or (args.log_dir / f"eval_latest_step{step}_all_trajs")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    action_horizon = checkpoint_action_horizon(checkpoint)
    print(f"[eval] trajectories={len(trajectories)}")
    print(f"[eval] data_dir={data_dir}")
    print(f"[eval] model_config={args.model_config}")
    print(f"[eval] feature_cache={args.feature_cache}")
    print(f"[eval] checkpoint={checkpoint}")
    print(f"[eval] output_dir={output_dir}")
    print(
        f"[eval] device={device}, gamma={args.gamma}, batch_size={args.batch_size}, "
        f"action_horizon={action_horizon}",
        flush=True,
    )

    if args.model_config is not None:
        policy = build_full_policy_from_model_config(
            checkpoint,
            args.model_config,
            args.rl_token_path,
            args.norm_stats_path,
            device,
        )
    else:
        policy = build_policy(
            checkpoint, args.rl_token_path, args.norm_stats_path, device, action_horizon
        )
    kin = SimpleUrdfKinematics(str(args.urdf_path))
    feature_cache = load_prefix_feature_cache(args.feature_cache)

    rows = []
    for idx, path in enumerate(trajectories):
        traj_dir = output_dir / f"{_trajectory_number(path):04d}_{path.stem}"
        print(f"[eval] {idx + 1}/{len(trajectories)} {path.name}", flush=True)
        metrics = evaluate_trajectory(
            path=path,
            out_dir=traj_dir,
            policy=policy,
            kin=kin,
            device=device,
            gamma=args.gamma,
            batch_size=args.batch_size,
            action_horizon=action_horizon,
            task_description=args.task_description,
            feature_cache=feature_cache,
        )
        rows.append(metrics)
        print(
            "[eval] done "
            f"success={int(metrics['success'])} reward={metrics['reward_sum']:.1f} "
            f"q_gap={metrics['q_gap_actor_minus_data_mean']:.4g} "
            f"mse={metrics['normalized_action_mse_mean']:.4g} "
            f"pt_gt_max={metrics['pt_gt_right_abs_max_error']:.3g} "
            f"actor_tcp_l2={metrics['actor_pt_tcp_l2_mean_m']:.4g}m",
            flush=True,
        )

    summary_fields = [
        "trajectory_name",
        "num_chunks",
        "valid_low_level_steps",
        "success",
        "reward_sum",
        "intervention_low_level_entries",
        "data_action_source",
        "stored_norm_action_mse_vs_executed_abs",
        "q_data_mean",
        "q_actor_mean",
        "q_gap_actor_minus_data_mean",
        "q_data_final",
        "q_actor_final",
        "normalized_action_mse_mean",
        "normalized_action_mse_final",
        "pt_gt_right_abs_max_error",
        "pt_gt_right_abs_mean_error",
        "actor_pt_right_abs_max_error",
        "actor_pt_right_abs_mean_error",
        "gt_pt_tcp_mse_m2",
        "gt_pt_tcp_rmse_m",
        "gt_pt_tcp_l2_mean_m",
        "gt_pt_tcp_l2_max_m",
        "actor_pt_tcp_mse_m2",
        "actor_pt_tcp_rmse_m",
        "actor_pt_tcp_l2_mean_m",
        "actor_pt_tcp_l2_max_m",
    ]
    with (output_dir / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in summary_fields})
    summary = {
        "checkpoint": str(checkpoint),
        "log_dir": str(args.log_dir),
        "num_trajectories": len(rows),
        "num_success": int(sum(1 for r in rows if r["success"])),
        "num_failure": int(sum(1 for r in rows if not r["success"])),
        "gamma": args.gamma,
        "mean_q_gap_actor_minus_data": float(
            np.mean([r["q_gap_actor_minus_data_mean"] for r in rows])
        ),
        "mean_normalized_action_mse": float(
            np.mean([r["normalized_action_mse_mean"] for r in rows])
        ),
        "mean_actor_pt_right_abs_mean_error": float(
            np.mean([r["actor_pt_right_abs_mean_error"] for r in rows])
        ),
        "mean_pt_gt_right_abs_max_error": float(
            np.mean([r["pt_gt_right_abs_max_error"] for r in rows])
        ),
        "mean_gt_pt_tcp_mse_m2": float(np.mean([r["gt_pt_tcp_mse_m2"] for r in rows])),
        "mean_gt_pt_tcp_l2_mean_m": float(
            np.mean([r["gt_pt_tcp_l2_mean_m"] for r in rows])
        ),
        "mean_actor_pt_tcp_mse_m2": float(
            np.mean([r["actor_pt_tcp_mse_m2"] for r in rows])
        ),
        "mean_actor_pt_tcp_l2_mean_m": float(
            np.mean([r["actor_pt_tcp_l2_mean_m"] for r in rows])
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    plot_aggregate(output_dir / "aggregate_summary.png", rows)
    print("[eval] summary", json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
