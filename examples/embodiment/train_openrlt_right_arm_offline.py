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

"""OpenPI-RLT style offline TD3+BC for Piper right-arm TCP-local chunks.

The actor/critic train only the six right-arm joints. The right gripper is kept
from the reference/VLA action during rollout and from the recorded PT action for
visualization/FK reconstruction.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def _find_repo_root(start: Path) -> Path:
    for path in [start, *start.parents]:
        if (path / "rlinf").is_dir() and (path / "examples").is_dir():
            return path
    raise RuntimeError(f"Could not find RLinf repo root from {start}")


REPO_ROOT = _find_repo_root(Path(__file__).resolve().parent)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rlinf.utils.offline_td3_visualization import SimpleUrdfKinematics  # noqa: E402

RIGHT_ARM_FULL = slice(7, 14)
RIGHT_ARM_JOINTS = slice(7, 13)
RIGHT_LOCAL_JOINTS = slice(0, 6)
RIGHT_BASE_OFFSET = np.array([0.0, -0.22, 0.0], dtype=np.float64)
DEFAULT_TCP_TARGET = (0.2818035953, 0.0385303870, 0.0602531905)


@dataclass
class TrainConfig:
    data_dir: str
    feature_cache: str
    norm_stats_path: str
    output_dir: str
    urdf_path: str
    model_path: str | None = None
    rl_token_path: str | None = None
    rl_token_source: str = "autoencoder"
    prefix_feature_type: str = "image_only"
    num_image_tokens: int = 768
    actor_train_prefix_token_linear: bool = False
    critic_train_prefix_token_linear: bool = False
    max_steps: int = 5000
    batch_size: int = 256
    gamma: float = 0.94
    n_step: int = 10
    actor_lr: float = 1e-4
    critic_lr: float = 1e-4
    bc_weight: float = 20.0
    q_weight: float = 0.0
    delta_weight: float = 10.0
    joint_abs_weight: float = 100.0
    tcp_weight: float = 5000.0
    tcp_boundary_weight: float = 1000.0
    q_warmup_steps: int = 100000000
    tau: float = 0.005
    actor_update_period: int = 2
    fixed_std: float = 0.0
    reference_dropout_prob: float = 0.0
    actor_residual_ref: bool = True
    actor_residual_scale: float = 1.0
    save_interval: int = 50
    log_interval: int = 10
    patience: int = 1200
    min_delta: float = 1e-6
    seed: int = 1234
    gpu: int = 0
    viz_per_class: int = 3
    z_dim: int = 2048
    proprio_dim: int = 6
    chunk_len: int = 10
    action_dim: int = 6
    hidden_dim: int = 256
    num_layers: int = 2
    tcp_target: tuple[float, float, float] | list[float] | None = DEFAULT_TCP_TARGET
    tcp_radius: float = 0.10
    tcp_filter_mode: str = "chunk_center"
    warm_up_chunks: int = 0
    generate_feature_cache_if_missing: bool = False
    feature_cache_batch_size: int = 8
    feature_cache_task_description: str = "peg and insertion"
    feature_cache_model_config: str | None = None
    realtime_prefix_features: bool = False


def layer_norm_no_params(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return (x - x.mean(dim=-1, keepdim=True)) / torch.sqrt(
        x.var(dim=-1, unbiased=False, keepdim=True) + eps
    )


def uses_prefix_token_training(cfg: TrainConfig) -> bool:
    return str(cfg.rl_token_source).lower() == "image_last_linear" and (
        bool(cfg.actor_train_prefix_token_linear)
        or bool(cfg.critic_train_prefix_token_linear)
    )


def cache_feature_key(cfg: TrainConfig) -> str:
    return "prefix_tokens" if uses_prefix_token_training(cfg) else "rltoken"


class MLP(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, num_layers: int, output_dim: int
    ):
        super().__init__()
        dims = [input_dim] + [hidden_dim] * int(num_layers) + [output_dim]
        layers: list[nn.Module] = []
        for idx in range(len(dims) - 2):
            layers.append(nn.Linear(dims[idx], dims[idx + 1]))
            layers.append(nn.LayerNorm(dims[idx + 1]))
            layers.append(nn.GELU())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class OfflinePrefixTokenEncoder(nn.Module):
    def __init__(self, cfg: TrainConfig, role: str = "both"):
        super().__init__()
        self.cfg = cfg
        self.role = str(role)
        seq_len = int(cfg.num_image_tokens) + 1
        self.actor_prefix_token_linear = None
        self.critic_prefix_token_linear_1 = None
        self.critic_prefix_token_linear_2 = None
        if self.role in {"actor", "both"}:
            self.actor_prefix_token_linear = nn.Linear(seq_len, 1, bias=False)
            self._init_prefix_token_linear(self.actor_prefix_token_linear)
        if self.role in {"critic", "both"}:
            self.critic_prefix_token_linear_1 = nn.Linear(seq_len, 1, bias=False)
            self.critic_prefix_token_linear_2 = nn.Linear(seq_len, 1, bias=False)
            self._init_prefix_token_linear(self.critic_prefix_token_linear_1)
            self._init_prefix_token_linear(self.critic_prefix_token_linear_2)

    @staticmethod
    def _init_prefix_token_linear(module: nn.Linear) -> None:
        nn.init.zeros_(module.weight)
        module.weight.data[0, -1] = 1.0

    def _linear_prefix_token(
        self, module: nn.Linear | None, prefix_tokens: torch.Tensor
    ) -> torch.Tensor:
        if module is None:
            raise RuntimeError("prefix token linear module is not initialized")
        if prefix_tokens.dim() != 3:
            return prefix_tokens
        expected_len = int(self.cfg.num_image_tokens) + 1
        if prefix_tokens.shape[1] != expected_len:
            raise ValueError(
                "prefix token cache has incompatible sequence length: "
                f"got {prefix_tokens.shape[1]}, expected {expected_len}"
            )
        return module(prefix_tokens.transpose(1, 2)).squeeze(-1)

    def actor_token(self, prefix_tokens: torch.Tensor) -> torch.Tensor:
        return self._linear_prefix_token(self.actor_prefix_token_linear, prefix_tokens)

    def critic_tokens(
        self, prefix_tokens: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if prefix_tokens.dim() != 3:
            return prefix_tokens
        return (
            self._linear_prefix_token(self.critic_prefix_token_linear_1, prefix_tokens),
            self._linear_prefix_token(self.critic_prefix_token_linear_2, prefix_tokens),
        )


class ChunkActor(nn.Module):
    def __init__(
        self,
        cfg: TrainConfig,
        prefix_encoder: OfflinePrefixTokenEncoder | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.prefix_encoder = prefix_encoder
        flat_action_dim = cfg.chunk_len * cfg.action_dim
        self.z_proj = nn.Linear(cfg.z_dim, 256)
        self.state_proj = nn.Linear(cfg.proprio_dim, 64)
        self.ref_proj = nn.Linear(flat_action_dim, 256)
        self.trunk = MLP(
            256 + 64 + 256, cfg.hidden_dim, cfg.num_layers, flat_action_dim
        )
        self.fixed_std = float(cfg.fixed_std)
        self.residual_ref = bool(getattr(cfg, "actor_residual_ref", True))
        self.residual_scale = float(getattr(cfg, "actor_residual_scale", 1.0))
        if self.residual_ref:
            last = self.trunk.net[-1]
            if isinstance(last, nn.Linear):
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)

    def encode(
        self,
        z: torch.Tensor,
        proprio: torch.Tensor,
        ref_chunk: torch.Tensor,
    ) -> torch.Tensor:
        bsz = z.shape[0]
        ref_flat = ref_chunk.reshape(bsz, -1)
        if z.dim() == 3:
            if self.prefix_encoder is None:
                raise RuntimeError("prefix token input requires actor prefix encoder")
            z = self.prefix_encoder.actor_token(z)
        z_feat = layer_norm_no_params(self.z_proj(z))
        proprio_feat = torch.tanh(layer_norm_no_params(self.state_proj(proprio)))
        ref_feat = torch.tanh(layer_norm_no_params(self.ref_proj(ref_flat)))
        return torch.cat([z_feat, proprio_feat, ref_feat], dim=-1)

    def mean(
        self,
        z: torch.Tensor,
        proprio: torch.Tensor,
        ref_chunk: torch.Tensor,
    ) -> torch.Tensor:
        bsz = z.shape[0]
        out = self.trunk(self.encode(z, proprio, ref_chunk))
        out = out.reshape(bsz, self.cfg.chunk_len, self.cfg.action_dim)
        if self.residual_ref:
            return ref_chunk + self.residual_scale * out
        return out

    def sample(
        self,
        z: torch.Tensor,
        proprio: torch.Tensor,
        ref_chunk: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        mu = self.mean(z, proprio, ref_chunk)
        if deterministic or self.fixed_std <= 0:
            return mu
        return mu + torch.randn_like(mu) * self.fixed_std


class QNetwork(nn.Module):
    def __init__(self, cfg: TrainConfig, critic_index: int = 1):
        super().__init__()
        self.critic_index = int(critic_index)
        flat_action_dim = cfg.chunk_len * cfg.action_dim
        self.z_proj = nn.Linear(cfg.z_dim, 256)
        self.state_proj = nn.Linear(cfg.proprio_dim, 64)
        self.action_proj = nn.Linear(flat_action_dim, 256)
        self.trunk = MLP(256 + 64 + 256, cfg.hidden_dim, cfg.num_layers, 1)

    def forward(
        self, z: torch.Tensor, proprio: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        if isinstance(z, tuple):
            z = z[0] if self.critic_index == 1 else z[1]
        bsz = z.shape[0]
        action_flat = action_chunk.reshape(bsz, -1)
        z_feat = layer_norm_no_params(self.z_proj(z))
        proprio_feat = torch.tanh(layer_norm_no_params(self.state_proj(proprio)))
        action_feat = torch.tanh(layer_norm_no_params(self.action_proj(action_flat)))
        return self.trunk(torch.cat([z_feat, proprio_feat, action_feat], dim=-1))


class TwinCritic(nn.Module):
    def __init__(
        self,
        cfg: TrainConfig,
        prefix_encoder: OfflinePrefixTokenEncoder | None = None,
    ):
        super().__init__()
        self.prefix_encoder = prefix_encoder
        self.q1 = QNetwork(cfg, critic_index=1)
        self.q2 = QNetwork(cfg, critic_index=2)

    def forward(
        self, z: torch.Tensor, proprio: torch.Tensor, action_chunk: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if z.dim() == 3:
            if self.prefix_encoder is None:
                raise RuntimeError("prefix token input requires critic prefix encoder")
            z = self.prefix_encoder.critic_tokens(z)
        return self.q1(z, proprio, action_chunk), self.q2(z, proprio, action_chunk)

    def q_min(
        self, z: torch.Tensor, proprio: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        q1, q2 = self(z, proprio, action_chunk)
        return torch.minimum(q1, q2)


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.mul_(1.0 - tau).add_(source_param.data, alpha=tau)


def trainable_params(module: nn.Module):
    return [param for param in module.parameters() if param.requires_grad]


def build_actor_optimizer(actor: ChunkActor, cfg: TrainConfig) -> torch.optim.Optimizer:
    if actor.prefix_encoder is not None and not bool(
        cfg.actor_train_prefix_token_linear
    ):
        for param in actor.prefix_encoder.parameters():
            param.requires_grad_(False)
    return torch.optim.Adam(trainable_params(actor), lr=cfg.actor_lr)


def build_critic_optimizer(
    critic: TwinCritic, cfg: TrainConfig
) -> torch.optim.Optimizer:
    if critic.prefix_encoder is not None and not bool(
        cfg.critic_train_prefix_token_linear
    ):
        for param in critic.prefix_encoder.parameters():
            param.requires_grad_(False)
    return torch.optim.Adam(trainable_params(critic), lr=cfg.critic_lr)


def actor_head_state_dict(actor: ChunkActor) -> dict[str, torch.Tensor]:
    return {
        key: value
        for key, value in actor.state_dict().items()
        if not key.startswith("prefix_encoder.")
    }


def critic_head_state_dict(critic: TwinCritic, head: str) -> dict[str, torch.Tensor]:
    module = critic.q1 if head == "q1" else critic.q2
    return module.state_dict()


def prefix_state_dict(
    model: ChunkActor | TwinCritic, target: bool = False
) -> dict[str, torch.Tensor]:
    encoder = getattr(model, "prefix_encoder", None)
    if encoder is None:
        return {}
    prefix = "target_" if target else ""
    state = {}
    for name, tensor in encoder.state_dict().items():
        if name.startswith("actor_prefix_token_linear."):
            state[f"{prefix}{name}"] = tensor.detach().cpu()
        elif name.startswith("critic_prefix_token_linear_"):
            state[f"{prefix}{name}"] = tensor.detach().cpu()
    return state


def load_norm_stats(path: str, std_floor: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    stats = payload.get("norm_stats", payload)["actions"]
    mean = np.asarray(stats["mean"][:14], dtype=np.float64)[RIGHT_ARM_JOINTS]
    std = np.asarray(stats["std"][:14], dtype=np.float64)[RIGHT_ARM_JOINTS]
    std = np.where(np.abs(std) < float(std_floor), 1.0, std)
    return mean, std


def norm_delta_to_abs_right_joints(
    norm_action: np.ndarray,
    right_state_full: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    gripper_chunk: np.ndarray | None = None,
) -> np.ndarray:
    action = np.asarray(norm_action, dtype=np.float64).reshape(-1, 6)
    state = np.asarray(right_state_full, dtype=np.float64).reshape(7)
    delta = action * (std.reshape(1, 6) + 1e-6) + mean.reshape(1, 6)
    joints = delta + state[:6]
    if gripper_chunk is None:
        gripper = np.full((joints.shape[0], 1), state[6], dtype=np.float64)
    else:
        gripper = np.asarray(gripper_chunk, dtype=np.float64).reshape(-1, 1)
    return np.concatenate([joints, gripper], axis=-1)


def norm_delta_to_abs_joints_tensor(
    norm_action: torch.Tensor,
    state: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    delta = norm_action * (std.reshape(1, 1, -1) + 1e-6) + mean.reshape(1, 1, -1)
    return delta + state[:, None, :6]


def _rot_x(angle: torch.Tensor) -> torch.Tensor:
    c = torch.cos(angle)
    s = torch.sin(angle)
    z = torch.zeros_like(angle)
    o = torch.ones_like(angle)
    return torch.stack(
        [
            torch.stack([o, z, z], dim=-1),
            torch.stack([z, c, -s], dim=-1),
            torch.stack([z, s, c], dim=-1),
        ],
        dim=-2,
    )


def _rot_y(angle: torch.Tensor) -> torch.Tensor:
    c = torch.cos(angle)
    s = torch.sin(angle)
    z = torch.zeros_like(angle)
    o = torch.ones_like(angle)
    return torch.stack(
        [
            torch.stack([c, z, s], dim=-1),
            torch.stack([z, o, z], dim=-1),
            torch.stack([-s, z, c], dim=-1),
        ],
        dim=-2,
    )


def _rot_z(angle: torch.Tensor) -> torch.Tensor:
    c = torch.cos(angle)
    s = torch.sin(angle)
    z = torch.zeros_like(angle)
    o = torch.ones_like(angle)
    return torch.stack(
        [
            torch.stack([c, -s, z], dim=-1),
            torch.stack([s, c, z], dim=-1),
            torch.stack([z, z, o], dim=-1),
        ],
        dim=-2,
    )


def _axis_angle(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    axis = axis.to(device=angle.device, dtype=angle.dtype)
    axis = axis / torch.clamp(torch.linalg.norm(axis), min=1e-12)
    x, y, z = axis[0], axis[1], axis[2]
    c = torch.cos(angle)
    s = torch.sin(angle)
    one_c = 1.0 - c
    return torch.stack(
        [
            torch.stack(
                [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
                dim=-1,
            ),
            torch.stack(
                [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
                dim=-1,
            ),
            torch.stack(
                [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
                dim=-1,
            ),
        ],
        dim=-2,
    )


class DifferentiablePiperFK(nn.Module):
    def __init__(
        self,
        urdf_path: str,
        base_link: str = "base_link",
        tip_link: str = "gripper",
        base_offset: np.ndarray = RIGHT_BASE_OFFSET,
    ):
        super().__init__()
        kin = SimpleUrdfKinematics(urdf_path, base_link=base_link)
        joints = kin.chain_to(tip_link)
        xyz = []
        rpy = []
        axis = []
        is_movable = []
        for joint in joints:
            xyz.append(joint.xyz)
            rpy.append(joint.rpy)
            axis.append(joint.axis)
            is_movable.append(
                joint.joint_type in {"revolute", "continuous", "prismatic"}
            )
            if joint.joint_type == "prismatic":
                raise ValueError(
                    "DifferentiablePiperFK currently expects no prismatic joints."
                )
        self.register_buffer("xyz", torch.tensor(np.stack(xyz), dtype=torch.float32))
        self.register_buffer("rpy", torch.tensor(np.stack(rpy), dtype=torch.float32))
        self.register_buffer("axis", torch.tensor(np.stack(axis), dtype=torch.float32))
        self.register_buffer("is_movable", torch.tensor(is_movable, dtype=torch.bool))
        self.register_buffer(
            "base_offset", torch.tensor(base_offset, dtype=torch.float32)
        )

    def _origin_rotation(
        self, rpy: torch.Tensor, batch_shape: torch.Size
    ) -> torch.Tensor:
        roll, pitch, yaw = rpy[0], rpy[1], rpy[2]
        rot = (
            _rot_z(yaw.expand(batch_shape))
            @ _rot_y(pitch.expand(batch_shape))
            @ _rot_x(roll.expand(batch_shape))
        )
        return rot

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        orig_shape = q.shape[:-1]
        q_flat = q.reshape(-1, q.shape[-1])
        dtype = q_flat.dtype
        device = q_flat.device
        rot = (
            torch.eye(3, device=device, dtype=dtype)
            .expand(q_flat.shape[0], 3, 3)
            .clone()
        )
        pos = torch.zeros(q_flat.shape[0], 3, device=device, dtype=dtype)
        movable_idx = 0
        for joint_idx in range(self.xyz.shape[0]):
            xyz = self.xyz[joint_idx].to(device=device, dtype=dtype)
            rpy = self.rpy[joint_idx].to(device=device, dtype=dtype)
            origin_rot = self._origin_rotation(rpy, torch.Size([q_flat.shape[0]])).to(
                dtype=dtype
            )
            pos = pos + torch.matmul(rot, xyz.reshape(1, 3, 1)).squeeze(-1)
            rot = rot @ origin_rot
            if bool(self.is_movable[joint_idx].item()):
                angle = (
                    q_flat[:, movable_idx]
                    if movable_idx < q_flat.shape[1]
                    else torch.zeros(q_flat.shape[0], device=device, dtype=dtype)
                )
                rot = rot @ _axis_angle(self.axis[joint_idx], angle).to(dtype=dtype)
                movable_idx += 1
        pos = pos + self.base_offset.to(device=device, dtype=dtype).reshape(1, 3)
        return pos.reshape(*orig_shape, 3)


def actor_execution_losses(
    pred: torch.Tensor,
    target: torch.Tensor,
    state: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    fk: DifferentiablePiperFK,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    pred_abs = norm_delta_to_abs_joints_tensor(pred, state, mean, std)
    target_abs = norm_delta_to_abs_joints_tensor(target, state, mean, std)
    joint_abs_loss = F.mse_loss(pred_abs, target_abs)
    pred_tcp = fk(pred_abs)
    target_tcp = fk(target_abs)
    tcp_sq = (pred_tcp - target_tcp).square().sum(dim=-1)
    tcp_loss = tcp_sq.mean()
    tcp_boundary_loss = tcp_sq[:, 0].mean()
    with torch.no_grad():
        tcp_err = torch.sqrt(torch.clamp(tcp_sq, min=0.0))
        joint_abs = (pred_abs - target_abs).abs()
        metrics = {
            "tcp_err_mean_m": float(tcp_err.mean().detach().cpu()),
            "tcp_err_max_m": float(tcp_err.max().detach().cpu()),
            "tcp_boundary_err_mean_m": float(tcp_err[:, 0].mean().detach().cpu()),
            "joint_abs_err_mean_rad": float(joint_abs.mean().detach().cpu()),
            "joint_abs_err_max_rad": float(joint_abs.max().detach().cpu()),
        }
    return joint_abs_loss, tcp_loss, tcp_boundary_loss, metrics


def compute_mc_returns(
    rewards: np.ndarray, dones: np.ndarray, gamma: float
) -> np.ndarray:
    rewards = np.asarray(rewards, dtype=np.float64).reshape(-1)
    dones = np.asarray(dones, dtype=bool).reshape(-1)
    out = np.zeros_like(rewards)
    running = 0.0
    for idx in range(len(rewards) - 1, -1, -1):
        running = rewards[idx] + (0.0 if dones[idx] else gamma * running)
        out[idx] = running
    return out


def reshape_right_full(flat: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    env_action_dim = 14
    flat_dim = int(flat.shape[-1])
    if flat_dim % env_action_dim != 0:
        raise ValueError(f"Expected flat action dim divisible by 14, got {flat_dim}")
    horizon = flat_dim // env_action_dim
    if torch.is_tensor(flat):
        return flat.reshape(flat.shape[0], horizon, env_action_dim)[..., 7:14]
    return np.asarray(flat).reshape(flat.shape[0], horizon, env_action_dim)[..., 7:14]


def reshape_right_joints(flat: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    right = reshape_right_full(flat)
    return right[..., :6]


def _resolve_meta_path(path: str, data_dir: str) -> str:
    p = Path(path)
    if p.exists():
        return str(p)
    candidate = Path(data_dir) / p.name
    if candidate.exists():
        return str(candidate)
    return str(p)


def _extract_success_terminal_tcp(
    traj_paths: list[Path], kin: SimpleUrdfKinematics
) -> np.ndarray | None:
    terminal = []
    for path in traj_paths:
        traj = torch.load(path, map_location="cpu", weights_only=False)
        rewards = traj.get("rewards", None)
        if rewards is None or float(torch.as_tensor(rewards).sum().item()) <= 0:
            continue
        env_abs = traj.get("forward_inputs", {}).get("env_action_absolute", None)
        if env_abs is None:
            continue
        q = reshape_right_full(env_abs[-1, 0].float().reshape(1, -1))[0][-1].numpy()
        terminal.append(right_tcp(q, kin, RIGHT_BASE_OFFSET)[0])
    if not terminal:
        return None
    return np.stack(terminal, axis=0).mean(axis=0)


def _chunk_tcp_point(
    state_right_full: torch.Tensor,
    env_abs_right_full: torch.Tensor | None,
    kin: SimpleUrdfKinematics,
    mode: str,
) -> np.ndarray:
    mode = str(mode)
    if mode == "state" or env_abs_right_full is None:
        q = state_right_full.detach().cpu().numpy()
        return right_tcp(q, kin, RIGHT_BASE_OFFSET)[0]
    chunk = env_abs_right_full.detach().cpu().numpy()
    if mode == "chunk_end":
        q = chunk[-1]
    else:
        pts = right_tcp(chunk, kin, RIGHT_BASE_OFFSET)
        return pts.mean(axis=0)
    return right_tcp(q, kin, RIGHT_BASE_OFFSET)[0]


def _trajectory_index_from_path(path: Path, fallback: int) -> int:
    stem = path.stem
    parts = stem.split("_")
    if len(parts) >= 2 and parts[0] == "trajectory":
        try:
            return int(parts[1])
        except ValueError:
            return int(fallback)
    return int(fallback)


def _select_obs_at_chunk(obs: dict[str, Any], rel: int, task_description: str) -> dict:
    env_obs = {}
    for src_key, out_key in (
        ("main_images", "main_images"),
        ("extra_view_images", "extra_view_images"),
        ("states", "states"),
        ("task_descriptions", "task_descriptions"),
    ):
        if src_key not in obs:
            continue
        value = obs[src_key]
        if torch.is_tensor(value):
            selected = value[rel, 0].unsqueeze(0).contiguous()
            env_obs[out_key] = selected
        elif isinstance(value, list):
            item = value[rel] if rel < len(value) else value[-1]
            env_obs[out_key] = item
        else:
            env_obs[out_key] = value
    if "task_descriptions" not in env_obs:
        env_obs["task_descriptions"] = [task_description]
    return env_obs


@torch.no_grad()
def _extract_rl_token(model, env_obs: dict) -> torch.Tensor:
    prefix_output, _, _ = model._build_prefix_cache_from_obs(env_obs)
    features = model._select_token_features(prefix_output)
    token = model._encode_actor_token(features, use_target=False)
    return token.detach().cpu().float()


@torch.no_grad()
def _extract_prefix_tokens(
    model, env_obs: dict, cfg: TrainConfig, to_cpu: bool = True
) -> torch.Tensor:
    prefix_output, _, _ = model._build_prefix_cache_from_obs(env_obs)
    features = prefix_output
    expected_len = int(cfg.num_image_tokens) + 1
    if features.shape[1] < expected_len:
        raise ValueError(
            "prefix sequence is too short for offline image_last_linear training: "
            f"got {features.shape[1]} tokens, need at least {expected_len}"
        )
    image_tokens = features[:, : int(cfg.num_image_tokens), :]
    last_token = features[:, -1:, :]
    out = torch.cat([image_tokens, last_token], dim=1).detach().float()
    return out.cpu() if to_cpu else out


def _collate_env_obs(obs_list: list[dict]) -> dict:
    """Batch a list of single-chunk env-obs dicts along dim 0.

    Each entry comes from `_select_obs_at_chunk`, whose tensors carry a leading
    batch dim of 1; concatenating restores a real batch. Lists (e.g.
    ``task_descriptions``) are flattened; everything else falls back to a list.
    """
    if not obs_list:
        return {}
    collated: dict[str, Any] = {}
    for key in obs_list[0].keys():
        values = [obs[key] for obs in obs_list if key in obs]
        first = values[0]
        if torch.is_tensor(first):
            collated[key] = torch.cat(values, dim=0)
        elif isinstance(first, list):
            merged: list[Any] = []
            for value in values:
                merged.extend(value)
            collated[key] = merged
        else:
            collated[key] = values
    return collated


class RealtimeFeatureProvider:
    """Compute prefix tokens on the fly from raw observations via the frozen VLA.

    Drop-in replacement for the precomputed ``data["z"]`` tensor: supports
    ``.shape`` and ``__getitem__`` over dataset *row* indices, returning
    prefix-token tensors of shape ``(len(indices), num_image_tokens + 1, z_dim)``.
    Nothing is cached, so peak memory stays bounded by one sub-batch of features.
    """

    def __init__(
        self,
        model,
        rows: list[dict],
        trajectories: dict[str, dict],
        cfg: TrainConfig,
        device: torch.device,
    ):
        self.model = model
        self.rows = rows
        self.trajectories = trajectories
        self.cfg = cfg
        self.device = device
        self.task_description = str(cfg.feature_cache_task_description)
        seq_len = int(cfg.num_image_tokens) + 1
        self.shape = torch.Size([len(rows), seq_len, int(cfg.z_dim)])

    def _row_obs(self, row_idx: int) -> dict:
        row = self.rows[int(row_idx)]
        traj = self.trajectories[row["path"]]
        return _select_obs_at_chunk(
            traj["curr_obs"], int(row["rel"]), self.task_description
        )

    def _to_index_list(self, indices) -> list[int]:
        if torch.is_tensor(indices):
            return indices.reshape(-1).tolist()
        if isinstance(indices, (list, tuple)):
            return [int(i) for i in indices]
        return [int(indices)]

    def __getitem__(self, indices) -> torch.Tensor:
        idx_list = self._to_index_list(indices)
        if not idx_list:
            return torch.empty(0, *self.shape[1:], device=self.device)
        feats: list[torch.Tensor] = []
        sub_bs = max(1, int(self.cfg.feature_cache_batch_size))
        for start in range(0, len(idx_list), sub_bs):
            sub = idx_list[start : start + sub_bs]
            env_obs = _collate_env_obs([self._row_obs(i) for i in sub])
            feats.append(
                _extract_prefix_tokens(self.model, env_obs, self.cfg, to_cpu=False)
            )
        return torch.cat(feats, dim=0).to(self.device)


class CachedFeatureProvider:
    """Lazily index a memory-mapped, precomputed feature cache.

    Drop-in replacement for the eagerly-stacked ``data["z"]`` tensor: supports
    ``.shape`` and ``__getitem__`` over dataset *row* indices, translating each
    row to its cache row via ``feature_idx`` before slicing. Rows are only
    copied out of the (mmapped) cache when indexed, so peak memory stays
    bounded by the requested batch instead of the full cache.
    """

    def __init__(self, z_all: torch.Tensor, feature_idx: torch.Tensor):
        self.z_all = z_all
        self.feature_idx = feature_idx
        self.shape = torch.Size([feature_idx.shape[0], *z_all.shape[1:]])

    def __getitem__(self, indices) -> torch.Tensor:
        if not torch.is_tensor(indices):
            indices = torch.as_tensor(indices, dtype=torch.long)
        cache_idx = self.feature_idx[indices.reshape(-1)]
        return self.z_all[cache_idx].float()


def load_feature_model(cfg: TrainConfig, device: torch.device):
    """Instantiate the frozen OpenPI RL-token model used for prefix extraction."""
    from rlinf.models import get_model

    model_cfg = build_feature_model_cfg(cfg)
    model = get_model(model_cfg)
    model.to(device)
    model.eval()
    return model


def build_trajectory_metas(
    cfg: TrainConfig, loaded: dict[str, dict] | None = None
) -> tuple[list[dict], int]:
    """Scan trajectory files to build feature metas without a feature cache.

    Mirrors the trajectory/chunk enumeration in ``generate_feature_cache``. When
    ``loaded`` is provided, each trajectory is stored there (keyed by path) so the
    caller can reuse it and avoid loading every file twice.
    """
    metas: list[dict] = []
    traj_paths = sorted(Path(cfg.data_dir).glob("*.pt"))
    for fallback_idx, traj_path in enumerate(traj_paths):
        traj = torch.load(traj_path, map_location="cpu", weights_only=False)
        curr_obs = traj.get("curr_obs")
        if not isinstance(curr_obs, dict) or "states" not in curr_obs:
            continue
        if loaded is not None:
            loaded[str(traj_path)] = traj
        num_chunks = int(curr_obs["states"].shape[0])
        traj_reward_sum = float(torch.as_tensor(traj.get("rewards", 0)).float().sum())
        traj_index = _trajectory_index_from_path(traj_path, fallback_idx)
        for rel in range(num_chunks):
            metas.append(
                {
                    "path": str(traj_path),
                    "rel_chunk": int(rel),
                    "traj_index": int(traj_index),
                    "success": bool(traj_reward_sum > 0),
                }
            )
    return metas, len(traj_paths)


def generate_feature_cache(cfg: TrainConfig) -> None:
    if cfg.model_path is None:
        raise ValueError(
            "Feature cache is missing. Set `model_path`, or provide an existing "
            "`feature_cache`."
        )
    if cfg.rl_token_path is None and not uses_prefix_token_training(cfg):
        raise ValueError(
            "Feature cache is missing. Set `rl_token_path` for compressed RL-token "
            "cache generation, or enable image_last_linear prefix-token training."
        )

    device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")
    model = load_feature_model(cfg, device)

    metas: list[dict[str, Any]] = []
    features_cache: list[torch.Tensor] = []
    traj_paths = sorted(Path(cfg.data_dir).glob("*.pt"))
    for fallback_idx, traj_path in enumerate(traj_paths):
        traj = torch.load(traj_path, map_location="cpu", weights_only=False)
        curr_obs = traj.get("curr_obs")
        if not isinstance(curr_obs, dict) or "states" not in curr_obs:
            continue
        num_chunks = int(curr_obs["states"].shape[0])
        traj_reward_sum = float(torch.as_tensor(traj.get("rewards", 0)).float().sum())
        traj_index = _trajectory_index_from_path(traj_path, fallback_idx)
        for rel in range(num_chunks):
            env_obs = _select_obs_at_chunk(
                curr_obs,
                rel,
                str(cfg.feature_cache_task_description),
            )
            if uses_prefix_token_training(cfg):
                feature = _extract_prefix_tokens(model, env_obs, cfg)
            else:
                feature = _extract_rl_token(model, env_obs)
            features_cache.append(feature.squeeze(0))
            metas.append(
                {
                    "path": str(traj_path),
                    "rel_chunk": int(rel),
                    "traj_index": int(traj_index),
                    "success": bool(traj_reward_sum > 0),
                }
            )

    feature_key = cache_feature_key(cfg)
    if features_cache:
        feature_tensor = torch.stack(features_cache, dim=0).float().cpu()
    else:
        shape = (
            (0, int(cfg.num_image_tokens) + 1, int(cfg.z_dim))
            if feature_key == "prefix_tokens"
            else (0, int(cfg.z_dim))
        )
        feature_tensor = torch.empty(*shape, dtype=torch.float32)
    out_path = Path(cfg.feature_cache)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            feature_key: feature_tensor,
            "metas": metas,
            "source_log_dir": str(Path(cfg.data_dir).parent),
            "num_trajectories": len(traj_paths),
            "feature_source": {
                "model_path": cfg.model_path,
                "rl_token_path": cfg.rl_token_path,
                "rl_token_source": cfg.rl_token_source,
                "feature_key": feature_key,
                "task_description": cfg.feature_cache_task_description,
            },
        },
        out_path,
    )
    print(
        json.dumps(
            {
                "feature_cache": {
                    "path": str(out_path),
                    "feature_key": feature_key,
                    "num_features": int(feature_tensor.shape[0]),
                    "num_trajectories": len(traj_paths),
                }
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def build_feature_model_cfg(cfg: TrainConfig):
    from omegaconf import OmegaConf

    if cfg.feature_cache_model_config:
        source_cfg = OmegaConf.load(cfg.feature_cache_model_config)
        model_cfg = OmegaConf.create(
            OmegaConf.to_container(source_cfg.actor.model, resolve=True)
        )
    else:
        model_cfg = OmegaConf.create({})
    if not model_cfg.get("openpi", None):
        raise ValueError(
            "Feature cache generation requires OpenPI model config. "
            "Set `feature_cache_model_config` to the data-collection yaml."
        )
    model_cfg.model_type = "openpi_rl_token"
    model_cfg.model_path = cfg.model_path
    model_cfg.rl_token_path = cfg.rl_token_path
    model_cfg.rl_token_source = cfg.rl_token_source
    model_cfg.prefix_feature_type = cfg.prefix_feature_type
    model_cfg.num_image_tokens = int(cfg.num_image_tokens)
    model_cfg.actor_train_prefix_token_linear = bool(
        cfg.actor_train_prefix_token_linear
    )
    model_cfg.critic_train_prefix_token_linear = bool(
        cfg.critic_train_prefix_token_linear
    )
    if not model_cfg.get("precision", None):
        model_cfg.precision = "bf16"
    model_cfg.is_lora = bool(model_cfg.get("is_lora", False))
    model_cfg.action_horizon = int(model_cfg.get("num_action_chunks", cfg.chunk_len))
    model_cfg.num_action_chunks = int(model_cfg.get("num_action_chunks", cfg.chunk_len))
    model_cfg.action_dim = int(model_cfg.get("action_dim", 14))
    model_cfg.env_action_dim = int(
        model_cfg.get("env_action_dim", model_cfg.action_dim)
    )
    model_cfg.robot_state_dim = int(model_cfg.get("robot_state_dim", 14))
    model_cfg.freeze_rl_token = True
    return model_cfg


def ensure_feature_cache(cfg: TrainConfig, generator=generate_feature_cache) -> None:
    feature_cache = Path(cfg.feature_cache)
    if feature_cache.exists():
        return
    if not cfg.generate_feature_cache_if_missing:
        raise FileNotFoundError(
            f"Feature cache not found: {feature_cache}. "
            "Set `generate_feature_cache_if_missing: true` to create it first."
        )
    generator(cfg)
    if not feature_cache.exists():
        raise FileNotFoundError(
            f"Feature cache generator did not create: {feature_cache}"
        )


def load_dataset(
    cfg: TrainConfig,
    feature_model=None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    feature_key = cache_feature_key(cfg)
    realtime = bool(cfg.realtime_prefix_features)
    loaded: dict[str, dict] = {}
    if realtime:
        if feature_model is None:
            raise ValueError(
                "realtime_prefix_features requires a loaded feature_model."
            )
        metas, _ = build_trajectory_metas(cfg, loaded=loaded)
        z_all = None
    else:
        cache = torch.load(
            cfg.feature_cache, map_location="cpu", mmap=True, weights_only=False
        )
        metas = cache["metas"]
        if feature_key not in cache:
            if feature_key == "prefix_tokens" and "rltoken" in cache:
                raise ValueError(
                    "This config enables offline image_last_linear prefix training, "
                    "but the feature cache only contains compressed `rltoken`. "
                    "Delete/regenerate the cache with this config to store `prefix_tokens`."
                )
            raise KeyError(f"Feature cache missing expected key: {feature_key}")
        z_all = cache[feature_key]
    kin = SimpleUrdfKinematics(cfg.urdf_path)
    traj_paths = sorted(Path(cfg.data_dir).glob("*.pt"))
    if cfg.tcp_target is None:
        inferred = _extract_success_terminal_tcp(traj_paths, kin)
        if inferred is None:
            inferred = np.asarray(DEFAULT_TCP_TARGET, dtype=np.float64)
        tcp_target = inferred
    else:
        tcp_target = np.asarray(cfg.tcp_target, dtype=np.float64).reshape(3)
    rows = []
    skipped_radius = 0
    skipped_missing = 0
    skipped_warmup = 0
    for feature_idx, meta in enumerate(metas):
        path = _resolve_meta_path(str(meta["path"]), cfg.data_dir)
        rel = int(meta["rel_chunk"])
        if rel < int(cfg.warm_up_chunks):
            skipped_warmup += 1
            continue
        if path not in loaded:
            if not Path(path).exists():
                skipped_missing += 1
                continue
            loaded[path] = torch.load(path, map_location="cpu", weights_only=False)
        traj = loaded[path]
        fwd = traj.get("forward_inputs", {})
        action_src = fwd.get("executed_action", fwd.get("action", traj["actions"]))
        action_flat = action_src[rel, 0].float().reshape(-1)
        if action_flat.numel() % 14 != 0:
            actions = action_flat.reshape(-1, 6)
        else:
            actions = reshape_right_joints(action_flat.reshape(1, -1))[0]
        ref = fwd.get("ref_action", None)
        if ref is None:
            ref_right = actions
        else:
            ref_flat = ref[rel, 0].float().reshape(-1)
            if ref_flat.numel() % 14 != 0:
                ref_right = ref_flat.reshape(-1, 6)
            else:
                ref_right = reshape_right_joints(ref_flat.reshape(1, -1))[0]
        state = traj["curr_obs"]["states"][rel, 0].float()
        next_state = traj["next_obs"]["states"][rel, 0].float()
        reward = float(traj["rewards"][rel, 0].float().reshape(-1).sum().item())
        traj_reward_sum = float(torch.as_tensor(traj["rewards"]).float().sum().item())
        done_src = traj.get("dones", traj.get("terminations"))
        done = bool(done_src[rel, 0].bool().reshape(-1).any().item())
        env_abs = fwd.get(
            "executed_env_action_absolute", fwd.get("env_action_absolute", None)
        )
        env_abs_right = None
        if env_abs is not None:
            env_abs_flat = env_abs[rel, 0].float().reshape(-1)
            if env_abs_flat.numel() % 14 == 0:
                env_abs_right = reshape_right_full(env_abs_flat.reshape(1, -1))[0]
        state_right_full = state[RIGHT_ARM_FULL]
        tcp_point = _chunk_tcp_point(
            state_right_full, env_abs_right, kin, cfg.tcp_filter_mode
        )
        tcp_distance = float(np.linalg.norm(tcp_point - tcp_target))
        if cfg.tcp_radius > 0 and tcp_distance > float(cfg.tcp_radius):
            skipped_radius += 1
            continue
        row_idx = len(rows)
        rows.append(
            {
                "idx": row_idx,
                "feature_idx": feature_idx,
                "path": path,
                "rel": rel,
                "traj_index": int(meta.get("traj_index", -1)),
                "success": int(meta.get("success", traj_reward_sum > 0)),
                "state_right": state[RIGHT_ARM_JOINTS],
                "state_right_full": state_right_full,
                "state_full": state,
                "next_state_right": next_state[RIGHT_ARM_JOINTS],
                "next_state_right_full": next_state[RIGHT_ARM_FULL],
                "action": actions,
                "ref": ref_right,
                "reward": reward,
                "done": done,
                "env_abs_right": env_abs_right,
                "tcp_distance": tcp_distance,
            }
        )
    if not rows:
        raise RuntimeError(
            f"Dataset filter kept zero rows: warm_up_chunks={cfg.warm_up_chunks}, "
            f"tcp_radius={cfg.tcp_radius}, target={tcp_target.tolist()}, "
            f"missing_paths={skipped_missing}"
        )

    by_key = {(r["path"], r["rel"]): r["idx"] for r in rows}
    for row in rows:
        next_idx = by_key.get((row["path"], row["rel"] + 1), row["idx"])
        row["next_idx"] = int(next_idx)

    data = {
        "z": (
            RealtimeFeatureProvider(feature_model, rows, loaded, cfg, device)
            if realtime
            else CachedFeatureProvider(
                z_all,
                torch.tensor([r["feature_idx"] for r in rows], dtype=torch.long),
            )
        ),
        "state": torch.stack([r["state_right"] for r in rows]),
        "next_state": torch.stack([r["next_state_right"] for r in rows]),
        "action": torch.stack([r["action"] for r in rows]),
        "ref": torch.stack([r["ref"] for r in rows]),
        "reward": torch.tensor(
            [r["reward"] for r in rows], dtype=torch.float32
        ).reshape(-1, 1),
        "done": torch.tensor([r["done"] for r in rows], dtype=torch.float32).reshape(
            -1, 1
        ),
        "next_idx": torch.tensor([r["next_idx"] for r in rows], dtype=torch.long),
        "success": torch.tensor([r["success"] for r in rows], dtype=torch.long),
        "rel": torch.tensor([r["rel"] for r in rows], dtype=torch.long),
        "traj_index": torch.tensor([r["traj_index"] for r in rows], dtype=torch.long),
        "rows": rows,
        "trajectories": loaded,
        "tcp_target": torch.tensor(tcp_target, dtype=torch.float32),
        "tcp_radius": float(cfg.tcp_radius),
        "warm_up_chunks": int(cfg.warm_up_chunks),
        "skipped_radius": int(skipped_radius),
        "skipped_missing": int(skipped_missing),
        "skipped_warmup": int(skipped_warmup),
    }
    data["next_ref"] = data["ref"][data["next_idx"]]
    summary = {
        "num_rows": len(rows),
        "num_trajectories": len({r["path"] for r in rows}),
        "success_rows": int(sum(int(r["success"]) for r in rows)),
        "failure_rows": int(sum(1 - int(r["success"]) for r in rows)),
        "feature_key": feature_key,
        "feature_shape": list(data["z"].shape),
        "tcp_target": tcp_target.tolist(),
        "tcp_radius": float(cfg.tcp_radius),
        "tcp_filter_mode": str(cfg.tcp_filter_mode),
        "warm_up_chunks": int(cfg.warm_up_chunks),
        "tcp_distance_min": float(min(r["tcp_distance"] for r in rows)),
        "tcp_distance_max": float(max(r["tcp_distance"] for r in rows)),
        "tcp_distance_mean": float(np.mean([r["tcp_distance"] for r in rows])),
        "skipped_radius": int(skipped_radius),
        "skipped_missing": int(skipped_missing),
        "skipped_warmup": int(skipped_warmup),
    }
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(cfg.output_dir) / "dataset_filter_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps({"dataset": summary}, ensure_ascii=False), flush=True)
    return data


def build_n_step_arrays(data: dict[str, Any], gamma: float, n_step: int) -> None:
    rows = data["rows"]
    by_key = {(r["path"], r["rel"]): i for i, r in enumerate(rows)}
    n_returns = []
    n_dones = []
    n_next_idx = []
    for row in rows:
        ret = 0.0
        discount = 1.0
        done = False
        next_idx = row["idx"]
        for k in range(max(1, int(n_step))):
            idx = by_key.get((row["path"], row["rel"] + k))
            if idx is None:
                done = True
                break
            cur = rows[idx]
            ret += discount * float(cur["reward"])
            next_idx = idx
            if cur["done"]:
                done = True
                break
            discount *= float(gamma)
            next_idx = by_key.get((row["path"], row["rel"] + k + 1), idx)
        n_returns.append(ret)
        n_dones.append(done)
        n_next_idx.append(next_idx)
    data["n_return"] = torch.tensor(n_returns, dtype=torch.float32).reshape(-1, 1)
    data["n_done"] = torch.tensor(n_dones, dtype=torch.float32).reshape(-1, 1)
    data["n_next_idx"] = torch.tensor(n_next_idx, dtype=torch.long)


def batch(
    data: dict[str, Any], indices: torch.Tensor, device: torch.device
) -> dict[str, torch.Tensor]:
    nidx = data["n_next_idx"][indices]
    return {
        "z": data["z"][indices].to(device),
        "state": data["state"][indices].to(device),
        "action": data["action"][indices].to(device),
        "ref": data["ref"][indices].to(device),
        "n_return": data["n_return"][indices].to(device),
        "n_done": data["n_done"][indices].to(device),
        "next_z": data["z"][nidx].to(device),
        "next_state": data["state"][nidx].to(device),
        "next_ref": data["ref"][nidx].to(device),
    }


def dropout_ref(ref: torch.Tensor, prob: float) -> torch.Tensor:
    if prob <= 0:
        return ref
    keep = torch.rand(ref.shape[0], 1, 1, device=ref.device) >= prob
    return ref * keep.to(ref.dtype)


def smooth_delta_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    state: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    pred_abs = pred * (std.reshape(1, 1, -1) + 1e-6) + mean.reshape(1, 1, -1)
    target_abs = target * (std.reshape(1, 1, -1) + 1e-6) + mean.reshape(1, 1, -1)
    pred_abs = pred_abs.clone()
    target_abs = target_abs.clone()
    pred_abs[..., :6] = pred_abs[..., :6] + state[:, None, :6]
    target_abs[..., :6] = target_abs[..., :6] + state[:, None, :6]
    pred_vel = pred_abs[:, 1:, :6] - pred_abs[:, :-1, :6]
    target_vel = target_abs[:, 1:, :6] - target_abs[:, :-1, :6]
    return F.mse_loss(pred_vel, target_vel)


def right_tcp(
    q: np.ndarray, kin: SimpleUrdfKinematics, base_offset: np.ndarray
) -> np.ndarray:
    out = []
    for item in np.asarray(q, dtype=np.float64).reshape(-1, 7):
        links, _ = kin.fk_positions(item, tip_link="gripper")
        out.append(links[-1] + base_offset)
    return np.stack(out, axis=0)


def plot_right_chunks(
    out_path: Path,
    state_q: np.ndarray,
    pt_chunks: list[np.ndarray],
    gt_chunks: list[np.ndarray],
    actor_chunks: list[np.ndarray],
    kin: SimpleUrdfKinematics,
) -> None:
    base = np.array([0.0, -0.22, 0.0], dtype=np.float64)
    fig = plt.figure(figsize=(7.5, 6.0), dpi=150)
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    state_tcp = right_tcp(state_q, kin, base)
    ax.scatter(
        state_tcp[:, 0],
        state_tcp[:, 1],
        state_tcp[:, 2],
        s=12,
        c="#222222",
        alpha=0.45,
        label="right robot state",
    )
    ax.scatter(
        state_tcp[0, 0],
        state_tcp[0, 1],
        state_tcp[0, 2],
        s=60,
        c="#2f8f46",
        marker="o",
        label="trajectory start",
    )
    ax.scatter(
        state_tcp[-1, 0],
        state_tcp[-1, 1],
        state_tcp[-1, 2],
        s=80,
        c="#b3472f",
        marker="x",
        label="trajectory end",
    )
    arrays = [state_tcp]
    rows = [
        ("PT absolute", pt_chunks, "#6f2b8c", "-", 1.8, 0.45),
        ("GT reconstructed", gt_chunks, "#d17a22", "--", 1.4, 0.75),
        ("Actor reconstructed", actor_chunks, "#2a6fbb", ":", 1.7, 0.90),
    ]
    for label, chunks, color, style, width, alpha in rows:
        for idx, q in enumerate(chunks):
            pts = right_tcp(q, kin, base)
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


def plot_q(
    out_path: Path,
    q_data: np.ndarray,
    q_actor: np.ndarray,
    mc: np.ndarray,
    mse: np.ndarray,
    tcp_err_m: np.ndarray | None = None,
) -> None:
    x = np.arange(len(q_data))
    num_rows = 3 if tcp_err_m is not None else 2
    fig, axes = plt.subplots(
        num_rows, 1, figsize=(8.5, 7.2 if num_rows == 3 else 6.0), dpi=150, sharex=True
    )
    axes[0].plot(
        x, q_data, marker="o", color="#222222", label="Q(s, data right action)"
    )
    axes[0].plot(
        x, q_actor, marker="o", color="#2a6fbb", label="Q(s, actor right action)"
    )
    axes[0].plot(x, mc, marker="s", linestyle="--", color="#2f8f46", label="MC return")
    axes[0].set_ylabel("Q / return")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")
    axes[1].plot(x, mse, marker="o", color="#b3472f", label="normalized action MSE")
    axes[1].set_xlabel("selected trajectory chunk index")
    axes[1].set_ylabel("MSE")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")
    if tcp_err_m is not None:
        axes[2].plot(
            x,
            np.asarray(tcp_err_m) * 1000.0,
            marker="o",
            color="#6f2b8c",
            label="TCP error",
        )
        axes[2].set_xlabel("selected trajectory chunk index")
        axes[2].set_ylabel("TCP err (mm)")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_mse_heatmap(out_path: Path, mse_matrix: np.ndarray) -> None:
    data = np.asarray(mse_matrix, dtype=np.float64).T
    fig, ax = plt.subplots(figsize=(6.5, 3.8), dpi=150)
    im = ax.imshow(data, aspect="auto", origin="lower", cmap="magma")
    ax.set_xlabel("small step")
    ax.set_ylabel("right joint")
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_title(f"actor vs data MSE, mean={float(np.mean(mse_matrix)):.4g}")
    fig.colorbar(im, ax=ax, label="squared error")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_tcp_error_heatmap(out_path: Path, tcp_err_m: np.ndarray) -> None:
    data = np.asarray(tcp_err_m, dtype=np.float64).reshape(1, -1) * 1000.0
    fig, ax = plt.subplots(figsize=(6.5, 1.8), dpi=150)
    im = ax.imshow(data, aspect="auto", cmap="viridis")
    ax.set_xlabel("small step")
    ax.set_yticks([])
    ax.set_title(
        f"actor TCP error, mean={float(data.mean()):.2f} mm, max={float(data.max()):.2f} mm"
    )
    ax.set_xticks(np.arange(data.shape[1]))
    fig.colorbar(im, ax=ax, label="mm")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


@torch.no_grad()
def visualize(
    step: int,
    cfg: TrainConfig,
    data: dict[str, Any],
    actor: ChunkActor,
    critic: TwinCritic,
    device: torch.device,
    action_mean: np.ndarray,
    action_std: np.ndarray,
) -> None:
    actor.eval()
    critic.eval()
    kin = SimpleUrdfKinematics(cfg.urdf_path)
    out_root = Path(cfg.output_dir) / "debug" / f"step_{step:06d}"
    out_root.mkdir(parents=True, exist_ok=True)

    rows = data["rows"]
    by_traj: dict[str, list[int]] = {}
    for idx, row in enumerate(rows):
        by_traj.setdefault(row["path"], []).append(idx)
    success_paths = []
    fail_paths = []
    for path, idxs in by_traj.items():
        reward_sum = sum(float(rows[i]["reward"]) for i in idxs)
        (success_paths if reward_sum > 0 else fail_paths).append(path)
    selected = [("success", p) for p in sorted(success_paths)[: cfg.viz_per_class]] + [
        ("failure", p) for p in sorted(fail_paths)[: cfg.viz_per_class]
    ]

    summary = []
    for label, path in selected:
        idxs = sorted(by_traj[path], key=lambda i: rows[i]["rel"])
        z = data["z"][idxs].to(device)
        state = data["state"][idxs].to(device)
        ref = data["ref"][idxs].to(device)
        target = data["action"][idxs].to(device)
        actor_action = actor.sample(z, state, ref, deterministic=True)
        q_data = critic.q_min(z, state, target).squeeze(-1).cpu().numpy()
        q_actor = critic.q_min(z, state, actor_action).squeeze(-1).cpu().numpy()
        rewards = np.asarray([rows[i]["reward"] for i in idxs], dtype=np.float64)
        dones = np.asarray([rows[i]["done"] for i in idxs], dtype=bool)
        mc = compute_mc_returns(rewards, dones, cfg.gamma)
        mse = ((actor_action - target) ** 2).mean(dim=(1, 2)).cpu().numpy()

        stem = Path(path).stem
        pt_chunks = []
        gt_chunks = []
        actor_chunks = []
        state_q = []
        tcp_err_per_chunk = []
        tcp_err_segments = []
        joint_abs_err_segments = []
        actor_np = actor_action.cpu().numpy()
        target_np = target.cpu().numpy()
        for local, row_idx in enumerate(idxs):
            row = rows[row_idx]
            state_right_full = row["state_right_full"].numpy()
            state_q.append(state_right_full)
            pt = row["env_abs_right"]
            if pt is None:
                pt_abs = norm_delta_to_abs_right_joints(
                    target_np[local], state_right_full, action_mean, action_std
                )
            else:
                pt_abs = pt.numpy()
            gt_abs = norm_delta_to_abs_right_joints(
                target_np[local],
                state_right_full,
                action_mean,
                action_std,
                gripper_chunk=pt_abs[:, 6],
            )
            actor_abs = norm_delta_to_abs_right_joints(
                actor_np[local],
                state_right_full,
                action_mean,
                action_std,
                gripper_chunk=pt_abs[:, 6],
            )
            pt_chunks.append(pt_abs)
            gt_chunks.append(gt_abs)
            actor_chunks.append(actor_abs)
            gt_tcp = right_tcp(gt_abs, kin, RIGHT_BASE_OFFSET)
            actor_tcp = right_tcp(actor_abs, kin, RIGHT_BASE_OFFSET)
            tcp_err = np.linalg.norm(actor_tcp - gt_tcp, axis=1)
            tcp_err_segments.append(tcp_err)
            tcp_err_per_chunk.append(float(np.mean(tcp_err)))
            joint_abs_err_segments.append(np.abs(actor_abs[:, :6] - gt_abs[:, :6]))

        traj_dir = out_root / label / stem
        traj_dir.mkdir(parents=True, exist_ok=True)
        plot_q(
            traj_dir / "q_values.png",
            q_data,
            q_actor,
            mc,
            mse,
            np.asarray(tcp_err_per_chunk),
        )
        plot_right_chunks(
            traj_dir / "right_tcp_chunks.png",
            np.stack(state_q, axis=0),
            pt_chunks,
            gt_chunks,
            actor_chunks,
            kin,
        )
        worst = int(np.argmax(mse))
        plot_mse_heatmap(
            traj_dir / f"mse_heatmap_worst_chunk_{worst:02d}.png",
            ((actor_np[worst] - target_np[worst]) ** 2),
        )
        plot_tcp_error_heatmap(
            traj_dir / f"tcp_error_worst_chunk_{worst:02d}.png",
            tcp_err_segments[worst],
        )
        tcp_all = np.concatenate(tcp_err_segments, axis=0)
        joint_all = np.concatenate(joint_abs_err_segments, axis=0)
        summary.append(
            {
                "label": label,
                "path": path,
                "q_data_mean": float(np.mean(q_data)),
                "q_actor_mean": float(np.mean(q_actor)),
                "mc_final": float(mc[-1]) if len(mc) else 0.0,
                "mse_mean": float(np.mean(mse)),
                "mse_max": float(np.max(mse)),
                "tcp_err_mean_m": float(np.mean(tcp_all)),
                "tcp_err_p90_m": float(np.percentile(tcp_all, 90)),
                "tcp_err_max_m": float(np.max(tcp_all)),
                "joint_abs_err_mean_rad": float(np.mean(joint_all)),
                "joint_abs_err_max_rad": float(np.max(joint_all)),
            }
        )
    (out_root / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    actor.train()
    critic.train()


def save_checkpoint(
    path: Path,
    step: int,
    cfg: TrainConfig,
    actor: ChunkActor,
    critic: TwinCritic,
    target_actor: ChunkActor,
    target_critic: TwinCritic,
    actor_opt: torch.optim.Optimizer,
    critic_opt: torch.optim.Optimizer,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": int(step),
            "config": asdict(cfg),
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "target_actor": target_actor.state_dict(),
            "target_critic": target_critic.state_dict(),
            "actor_opt": actor_opt.state_dict(),
            "critic_opt": critic_opt.state_dict(),
        },
        path / "actor_critic_standalone.pt",
    )
    model_state = {
        **{
            f"actor_head.{k}": v.detach().cpu()
            for k, v in actor_head_state_dict(actor).items()
        },
        **{
            f"critic_head_1.{k}": v.detach().cpu()
            for k, v in critic_head_state_dict(critic, "q1").items()
        },
        **{
            f"critic_head_2.{k}": v.detach().cpu()
            for k, v in critic_head_state_dict(critic, "q2").items()
        },
        **prefix_state_dict(actor),
        **prefix_state_dict(critic),
    }
    target_model_state = {
        **{
            f"actor_head.{k}": v.detach().cpu()
            for k, v in actor_head_state_dict(target_actor).items()
        },
        **{
            f"critic_head_1.{k}": v.detach().cpu()
            for k, v in critic_head_state_dict(target_critic, "q1").items()
        },
        **{
            f"critic_head_2.{k}": v.detach().cpu()
            for k, v in critic_head_state_dict(target_critic, "q2").items()
        },
        **prefix_state_dict(target_actor, target=True),
        **prefix_state_dict(target_critic, target=True),
    }
    torch.save(
        {
            "format": "rlinf_td3_actor_critic_only",
            "global_step": int(step),
            "update_step": int(step),
            "config": asdict(cfg),
            "model": model_state,
            "target_model": target_model_state,
            "actor_optimizer": None,
            "critic_optimizer": None,
            "actor_lr_scheduler": None,
            "critic_lr_scheduler": None,
            "notes": (
                "OpenRLT right-arm 6-joint actor/critic checkpoint. "
                "The right gripper is not controlled by actor/critic."
            ),
        },
        path / "actor_critic.pt",
    )
    (path / "checkpoint_summary.json").write_text(
        json.dumps(
            {
                "step": int(step),
                "format": "rlinf_td3_actor_critic_only",
                "action_dim": int(cfg.action_dim),
                "proprio_dim": int(cfg.proprio_dim),
                "bc_weight": float(cfg.bc_weight),
                "q_weight": float(cfg.q_weight),
                "delta_weight": float(cfg.delta_weight),
                "joint_abs_weight": float(cfg.joint_abs_weight),
                "tcp_weight": float(cfg.tcp_weight),
                "tcp_boundary_weight": float(cfg.tcp_boundary_weight),
                "actor_residual_ref": bool(cfg.actor_residual_ref),
                "actor_residual_scale": float(cfg.actor_residual_scale),
                "model_keys": sorted(model_state.keys()),
                "target_model_keys": sorted(target_model_state.keys()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default="/shared_disk/users/angen.ye/data/gigarlinf/rlinf_cubeinsert_20260615/demos_fail_reward0_normdelta_last20_gamma089",
    )
    parser.add_argument(
        "--feature-cache",
        default="/shared_disk/users/angen.ye/code/giga_rlinf/train_logs/rltoken_feature_probe/last20_gamma089_allchunks_20260617_061511/features.pt",
    )
    parser.add_argument(
        "--norm-stats-path",
        default="/shared_disk/users/angen.ye/data/gigarlinf/rlinf_cubeinsert_20260615/pi05_cube_insert/norm_stats.json",
    )
    parser.add_argument(
        "--output-dir",
        default="/shared_disk/users/angen.ye/code/giga_rlinf/train_logs/openrlt_right_arm/openrlt_right_arm_tcp10cm_6joints",
    )
    parser.add_argument("--urdf-path", default="assets/piper_local_assets/piper.urdf")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--rl-token-path", default=None)
    parser.add_argument(
        "--rl-token-source",
        choices=["autoencoder", "last_token", "image_last_linear"],
        default="autoencoder",
    )
    parser.add_argument(
        "--prefix-feature-type",
        choices=["image_only", "full_prefix"],
        default="image_only",
    )
    parser.add_argument("--num-image-tokens", type=int, default=768)
    parser.add_argument(
        "--actor-train-prefix-token-linear",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--critic-train-prefix-token-linear",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--patience", type=int, default=1200)
    parser.add_argument("--viz-per-class", type=int, default=3)
    parser.add_argument("--chunk-len", type=int, default=10)
    parser.add_argument("--action-dim", type=int, default=6)
    parser.add_argument("--gamma", type=float, default=0.94)
    parser.add_argument("--n-step", type=int, default=10)
    parser.add_argument("--bc-weight", type=float, default=20.0)
    parser.add_argument("--q-weight", type=float, default=0.0)
    parser.add_argument("--delta-weight", type=float, default=10.0)
    parser.add_argument("--joint-abs-weight", type=float, default=100.0)
    parser.add_argument("--tcp-weight", type=float, default=5000.0)
    parser.add_argument("--tcp-boundary-weight", type=float, default=1000.0)
    parser.add_argument("--q-warmup-steps", type=int, default=100000000)
    parser.add_argument("--fixed-std", type=float, default=0.0)
    parser.add_argument("--reference-dropout-prob", type=float, default=0.0)
    parser.add_argument(
        "--actor-residual-ref", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--actor-residual-scale", type=float, default=1.0)
    parser.add_argument("--tcp-radius", type=float, default=0.10)
    parser.add_argument(
        "--tcp-filter-mode",
        choices=["chunk_center", "chunk_end", "state"],
        default="chunk_center",
    )
    parser.add_argument(
        "--tcp-target", type=float, nargs=3, default=list(DEFAULT_TCP_TARGET)
    )
    parser.add_argument("--warm-up-chunks", type=int, default=0)
    parser.add_argument(
        "--generate-feature-cache-if-missing",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--feature-cache-batch-size", type=int, default=8)
    parser.add_argument(
        "--feature-cache-task-description",
        default="peg and insertion",
    )
    parser.add_argument("--feature-cache-model-config", default=None)
    parser.add_argument(
        "--realtime-prefix-features",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    args = parser.parse_args()

    cfg = TrainConfig(**vars(args))
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(cfg.output_dir) / "config.json").write_text(
        json.dumps(asdict(cfg), indent=2), encoding="utf-8"
    )

    if cfg.realtime_prefix_features:
        if not uses_prefix_token_training(cfg):
            raise ValueError(
                "--realtime-prefix-features requires image_last_linear prefix training "
                "(rl_token_source=image_last_linear with actor/critic prefix linears)."
            )
        feature_model = load_feature_model(cfg, device)
        data = load_dataset(cfg, feature_model=feature_model, device=device)
    else:
        ensure_feature_cache(cfg)
        data = load_dataset(cfg)
    build_n_step_arrays(data, cfg.gamma, cfg.n_step)
    action_mean, action_std = load_norm_stats(cfg.norm_stats_path)
    mean_t = torch.tensor(action_mean, device=device, dtype=torch.float32)
    std_t = torch.tensor(action_std, device=device, dtype=torch.float32)
    fk = DifferentiablePiperFK(cfg.urdf_path).to(device)

    actor_prefix = (
        OfflinePrefixTokenEncoder(cfg, role="actor")
        if uses_prefix_token_training(cfg)
        else None
    )
    critic_prefix = (
        OfflinePrefixTokenEncoder(cfg, role="critic")
        if uses_prefix_token_training(cfg)
        else None
    )
    target_actor_prefix = (
        OfflinePrefixTokenEncoder(cfg, role="actor")
        if uses_prefix_token_training(cfg)
        else None
    )
    target_critic_prefix = (
        OfflinePrefixTokenEncoder(cfg, role="critic")
        if uses_prefix_token_training(cfg)
        else None
    )
    actor = ChunkActor(cfg, prefix_encoder=actor_prefix).to(device)
    critic = TwinCritic(cfg, prefix_encoder=critic_prefix).to(device)
    target_actor = ChunkActor(cfg, prefix_encoder=target_actor_prefix).to(device)
    target_critic = TwinCritic(cfg, prefix_encoder=target_critic_prefix).to(device)
    target_actor.load_state_dict(actor.state_dict())
    target_critic.load_state_dict(critic.state_dict())
    target_actor.requires_grad_(False)
    target_critic.requires_grad_(False)
    actor_opt = build_actor_optimizer(actor, cfg)
    critic_opt = build_critic_optimizer(critic, cfg)

    metrics_path = Path(cfg.output_dir) / "metrics.jsonl"
    num_samples = int(data["z"].shape[0])
    best_actor_metric = float("inf")
    last_improve = 0
    visualize(0, cfg, data, actor, critic, device, action_mean, action_std)
    save_checkpoint(
        Path(cfg.output_dir) / "checkpoints" / "global_step_0",
        0,
        cfg,
        actor,
        critic,
        target_actor,
        target_critic,
        actor_opt,
        critic_opt,
    )

    for step in range(1, cfg.max_steps + 1):
        idx = torch.randint(0, num_samples, (cfg.batch_size,))
        b = batch(data, idx, device)
        with torch.no_grad():
            next_ref = dropout_ref(b["next_ref"], cfg.reference_dropout_prob)
            next_action = target_actor.sample(
                b["next_z"], b["next_state"], next_ref, deterministic=False
            )
            target_q = target_critic.q_min(b["next_z"], b["next_state"], next_action)
            discount = cfg.gamma ** max(1, cfg.n_step)
            y = b["n_return"] + (1.0 - b["n_done"]) * discount * target_q
        q1, q2 = critic(b["z"], b["state"], b["action"])
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
        critic_opt.step()

        actor_loss = torch.zeros((), device=device)
        bc_loss = torch.zeros((), device=device)
        delta_loss = torch.zeros((), device=device)
        joint_abs_loss = torch.zeros((), device=device)
        tcp_loss = torch.zeros((), device=device)
        tcp_boundary_loss = torch.zeros((), device=device)
        actor_q = torch.zeros((), device=device)
        actor_exec_metrics = {
            "tcp_err_mean_m": 0.0,
            "tcp_err_max_m": 0.0,
            "tcp_boundary_err_mean_m": 0.0,
            "joint_abs_err_mean_rad": 0.0,
            "joint_abs_err_max_rad": 0.0,
        }
        if step % cfg.actor_update_period == 0:
            ref_in = dropout_ref(b["ref"], cfg.reference_dropout_prob)
            actor_action = actor.sample(b["z"], b["state"], ref_in, deterministic=True)
            bc_loss = F.mse_loss(actor_action, b["action"])
            delta_loss = smooth_delta_loss(
                actor_action, b["action"], b["state"], mean_t, std_t
            )
            (
                joint_abs_loss,
                tcp_loss,
                tcp_boundary_loss,
                actor_exec_metrics,
            ) = actor_execution_losses(
                actor_action,
                b["action"],
                b["state"],
                mean_t,
                std_t,
                fk,
            )
            actor_q = critic(b["z"], b["state"], actor_action)[0].mean()
            q_weight = float(cfg.q_weight) if step >= int(cfg.q_warmup_steps) else 0.0
            actor_loss = (
                cfg.bc_weight * bc_loss
                + cfg.delta_weight * delta_loss
                + cfg.joint_abs_weight * joint_abs_loss
                + cfg.tcp_weight * tcp_loss
                + cfg.tcp_boundary_weight * tcp_boundary_loss
                - q_weight * actor_q
            )
            actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            actor_opt.step()
            soft_update(target_actor, actor, cfg.tau)
            soft_update(target_critic, critic, cfg.tau)

        with torch.no_grad():
            row = {
                "step": step,
                "critic_loss": float(critic_loss.detach().cpu()),
                "actor_loss": float(actor_loss.detach().cpu()),
                "bc_loss": float(bc_loss.detach().cpu()),
                "delta_loss": float(delta_loss.detach().cpu()),
                "joint_abs_loss": float(joint_abs_loss.detach().cpu()),
                "tcp_loss": float(tcp_loss.detach().cpu()),
                "tcp_boundary_loss": float(tcp_boundary_loss.detach().cpu()),
                "tcp_err_mean_mm": float(actor_exec_metrics["tcp_err_mean_m"] * 1000.0),
                "tcp_err_max_mm": float(actor_exec_metrics["tcp_err_max_m"] * 1000.0),
                "tcp_boundary_err_mean_mm": float(
                    actor_exec_metrics["tcp_boundary_err_mean_m"] * 1000.0
                ),
                "joint_abs_err_mean_rad": float(
                    actor_exec_metrics["joint_abs_err_mean_rad"]
                ),
                "joint_abs_err_max_rad": float(
                    actor_exec_metrics["joint_abs_err_max_rad"]
                ),
                "actor_q": float(actor_q.detach().cpu()),
                "target_q_mean": float(y.detach().mean().cpu()),
            }
        if step % cfg.log_interval == 0:
            print(json.dumps(row), flush=True)
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

        actor_metric = row["tcp_err_mean_mm"] + 100.0 * row["joint_abs_err_mean_rad"]
        if actor_metric + cfg.min_delta < best_actor_metric and row["bc_loss"] > 0:
            best_actor_metric = actor_metric
            last_improve = step
        if step % cfg.save_interval == 0:
            save_checkpoint(
                Path(cfg.output_dir) / "checkpoints" / f"global_step_{step}",
                step,
                cfg,
                actor,
                critic,
                target_actor,
                target_critic,
                actor_opt,
                critic_opt,
            )
            visualize(step, cfg, data, actor, critic, device, action_mean, action_std)
        if step - last_improve > cfg.patience and step > 1000:
            print(
                f"early stop at step {step}; best actor metric {best_actor_metric:.6g}",
                flush=True,
            )
            break

    save_checkpoint(
        Path(cfg.output_dir) / "checkpoints" / "final",
        step,
        cfg,
        actor,
        critic,
        target_actor,
        target_critic,
        actor_opt,
        critic_opt,
    )
    visualize(step, cfg, data, actor, critic, device, action_mean, action_std)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
