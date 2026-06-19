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

"""OpenPI-RLT style offline TD3+BC for Piper right-arm last20 chunks."""

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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rlinf.utils.offline_td3_visualization import SimpleUrdfKinematics  # noqa: E402

RIGHT_ARM = slice(7, 14)
RIGHT_DELTA_MASK = np.array([True, True, True, True, True, True, False])


@dataclass
class TrainConfig:
    data_dir: str
    feature_cache: str
    norm_stats_path: str
    output_dir: str
    urdf_path: str
    max_steps: int = 5000
    batch_size: int = 256
    gamma: float = 0.89
    n_step: int = 10
    actor_lr: float = 1e-4
    critic_lr: float = 1e-4
    bc_weight: float = 10.0
    q_weight: float = 0.1
    delta_weight: float = 10.0
    tau: float = 0.005
    actor_update_period: int = 2
    fixed_std: float = 0.002
    reference_dropout_prob: float = 0.5
    save_interval: int = 50
    log_interval: int = 10
    patience: int = 1200
    min_delta: float = 1e-6
    seed: int = 1234
    gpu: int = 0
    viz_per_class: int = 3
    z_dim: int = 2048
    proprio_dim: int = 7
    chunk_len: int = 10
    action_dim: int = 7
    hidden_dim: int = 256
    num_layers: int = 2


def layer_norm_no_params(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return (x - x.mean(dim=-1, keepdim=True)) / torch.sqrt(
        x.var(dim=-1, unbiased=False, keepdim=True) + eps
    )


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


class ChunkActor(nn.Module):
    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.cfg = cfg
        flat_action_dim = cfg.chunk_len * cfg.action_dim
        self.z_proj = nn.Linear(cfg.z_dim, 256)
        self.proprio_proj = nn.Linear(cfg.proprio_dim, 64)
        self.ref_proj = nn.Linear(flat_action_dim, 256)
        self.trunk = MLP(
            256 + 64 + 256, cfg.hidden_dim, cfg.num_layers, flat_action_dim
        )
        self.fixed_std = float(cfg.fixed_std)

    def encode(
        self,
        z: torch.Tensor,
        proprio: torch.Tensor,
        ref_chunk: torch.Tensor,
    ) -> torch.Tensor:
        bsz = z.shape[0]
        ref_flat = ref_chunk.reshape(bsz, -1)
        z_feat = layer_norm_no_params(self.z_proj(z))
        proprio_feat = torch.tanh(layer_norm_no_params(self.proprio_proj(proprio)))
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
        return out.reshape(bsz, self.cfg.chunk_len, self.cfg.action_dim)

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
    def __init__(self, cfg: TrainConfig):
        super().__init__()
        flat_action_dim = cfg.chunk_len * cfg.action_dim
        self.z_proj = nn.Linear(cfg.z_dim, 256)
        self.proprio_proj = nn.Linear(cfg.proprio_dim, 64)
        self.action_proj = nn.Linear(flat_action_dim, 256)
        self.trunk = MLP(256 + 64 + 256, cfg.hidden_dim, cfg.num_layers, 1)

    def forward(
        self, z: torch.Tensor, proprio: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        bsz = z.shape[0]
        action_flat = action_chunk.reshape(bsz, -1)
        z_feat = layer_norm_no_params(self.z_proj(z))
        proprio_feat = torch.tanh(layer_norm_no_params(self.proprio_proj(proprio)))
        action_feat = torch.tanh(layer_norm_no_params(self.action_proj(action_flat)))
        return self.trunk(torch.cat([z_feat, proprio_feat, action_feat], dim=-1))


class TwinCritic(nn.Module):
    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.q1 = QNetwork(cfg)
        self.q2 = QNetwork(cfg)

    def forward(
        self, z: torch.Tensor, proprio: torch.Tensor, action_chunk: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
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


def load_norm_stats(path: str, std_floor: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    stats = payload.get("norm_stats", payload)["actions"]
    mean = np.asarray(stats["mean"][:14], dtype=np.float64)[RIGHT_ARM]
    std = np.asarray(stats["std"][:14], dtype=np.float64)[RIGHT_ARM]
    std = np.where(np.abs(std) < float(std_floor), 1.0, std)
    return mean, std


def norm_delta_to_abs_right(
    norm_action: np.ndarray,
    right_state: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    action = np.asarray(norm_action, dtype=np.float64).reshape(-1, 7)
    state = np.asarray(right_state, dtype=np.float64).reshape(7)
    delta = action * (std.reshape(1, 7) + 1e-6) + mean.reshape(1, 7)
    absolute = delta.copy()
    absolute[:, RIGHT_DELTA_MASK] += state[RIGHT_DELTA_MASK]
    return absolute


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


def reshape_right(flat: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    if torch.is_tensor(flat):
        return flat.reshape(flat.shape[0], 10, 14)[..., 7:14]
    return np.asarray(flat).reshape(flat.shape[0], 10, 14)[..., 7:14]


def load_dataset(cfg: TrainConfig) -> dict[str, Any]:
    cache = torch.load(cfg.feature_cache, map_location="cpu")
    metas = cache["metas"]
    z_all = cache["rltoken"].float()
    rows = []
    loaded: dict[str, dict] = {}
    for row_idx, meta in enumerate(metas):
        path = str(meta["path"])
        rel = int(meta["rel_chunk"])
        if path not in loaded:
            loaded[path] = torch.load(path, map_location="cpu", weights_only=False)
        traj = loaded[path]
        actions = reshape_right(traj["actions"][rel, 0].float().reshape(1, -1))[0]
        ref = traj.get("forward_inputs", {}).get("ref_action", None)
        if ref is None:
            ref_right = actions
        else:
            ref_right = reshape_right(ref[rel, 0].float().reshape(1, -1))[0]
        state = traj["curr_obs"]["states"][rel, 0].float()
        next_state = traj["next_obs"]["states"][rel, 0].float()
        reward = float(traj["rewards"][rel, 0].float().reshape(-1)[0].item())
        done = bool(
            traj.get("dones", traj.get("terminations"))[rel, 0]
            .bool()
            .reshape(-1)[0]
            .item()
        )
        env_abs = traj.get("forward_inputs", {}).get("env_action_absolute", None)
        env_abs_right = None
        if env_abs is not None:
            env_abs_right = reshape_right(env_abs[rel, 0].float().reshape(1, -1))[0]
        rows.append(
            {
                "idx": row_idx,
                "path": path,
                "rel": rel,
                "traj_index": int(meta.get("traj_index", -1)),
                "success": int(meta.get("success", reward > 0)),
                "z": z_all[row_idx],
                "state_right": state[RIGHT_ARM],
                "state_full": state,
                "next_state_right": next_state[RIGHT_ARM],
                "action": actions,
                "ref": ref_right,
                "reward": reward,
                "done": done,
                "env_abs_right": env_abs_right,
            }
        )

    by_key = {(r["path"], r["rel"]): r["idx"] for r in rows}
    for row in rows:
        next_idx = by_key.get((row["path"], row["rel"] + 1), row["idx"])
        row["next_idx"] = int(next_idx)

    data = {
        "z": torch.stack([r["z"] for r in rows]),
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
    }
    data["next_z"] = data["z"][data["next_idx"]]
    data["next_ref"] = data["ref"][data["next_idx"]]
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
) -> None:
    x = np.arange(len(q_data))
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 6.0), dpi=150, sharex=True)
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
    axes[1].set_xlabel("last20 chunk index")
    axes[1].set_ylabel("MSE")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")
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
        traj_dir = out_root / label / stem
        traj_dir.mkdir(parents=True, exist_ok=True)
        plot_q(traj_dir / "q_values.png", q_data, q_actor, mc, mse)

        pt_chunks = []
        gt_chunks = []
        actor_chunks = []
        state_q = []
        actor_np = actor_action.cpu().numpy()
        target_np = target.cpu().numpy()
        for local, row_idx in enumerate(idxs):
            row = rows[row_idx]
            state_right = row["state_right"].numpy()
            state_q.append(state_right)
            pt = row["env_abs_right"]
            if pt is None:
                pt_abs = norm_delta_to_abs_right(
                    target_np[local], state_right, action_mean, action_std
                )
            else:
                pt_abs = pt.numpy()
            gt_abs = norm_delta_to_abs_right(
                target_np[local], state_right, action_mean, action_std
            )
            actor_abs = norm_delta_to_abs_right(
                actor_np[local], state_right, action_mean, action_std
            )
            pt_chunks.append(pt_abs)
            gt_chunks.append(gt_abs)
            actor_chunks.append(actor_abs)
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
        summary.append(
            {
                "label": label,
                "path": path,
                "q_data_mean": float(np.mean(q_data)),
                "q_actor_mean": float(np.mean(q_actor)),
                "mc_final": float(mc[-1]) if len(mc) else 0.0,
                "mse_mean": float(np.mean(mse)),
                "mse_max": float(np.max(mse)),
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
        path / "actor_critic.pt",
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
        default="/shared_disk/users/angen.ye/code/giga_rlinf/train_logs/openrlt_right_arm/openrlt_right_arm_last20",
    )
    parser.add_argument("--urdf-path", default="assets/piper_local_assets/piper.urdf")
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--gamma", type=float, default=0.89)
    parser.add_argument("--n-step", type=int, default=10)
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

    data = load_dataset(cfg)
    build_n_step_arrays(data, cfg.gamma, cfg.n_step)
    action_mean, action_std = load_norm_stats(cfg.norm_stats_path)
    mean_t = torch.tensor(action_mean, device=device, dtype=torch.float32)
    std_t = torch.tensor(action_std, device=device, dtype=torch.float32)

    actor = ChunkActor(cfg).to(device)
    critic = TwinCritic(cfg).to(device)
    target_actor = ChunkActor(cfg).to(device)
    target_critic = TwinCritic(cfg).to(device)
    target_actor.load_state_dict(actor.state_dict())
    target_critic.load_state_dict(critic.state_dict())
    actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr)

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
        actor_q = torch.zeros((), device=device)
        if step % cfg.actor_update_period == 0:
            ref_in = dropout_ref(b["ref"], cfg.reference_dropout_prob)
            actor_action = actor.sample(b["z"], b["state"], ref_in, deterministic=False)
            bc_loss = F.mse_loss(actor_action, b["action"])
            delta_loss = smooth_delta_loss(
                actor_action, b["action"], b["state"], mean_t, std_t
            )
            actor_q = critic.q1(b["z"], b["state"], actor_action).mean()
            actor_loss = (
                cfg.bc_weight * bc_loss
                + cfg.delta_weight * delta_loss
                - cfg.q_weight * actor_q
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
                "actor_q": float(actor_q.detach().cpu()),
                "target_q_mean": float(y.detach().mean().cpu()),
            }
        if step % cfg.log_interval == 0:
            print(json.dumps(row), flush=True)
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

        actor_metric = row["bc_loss"] + row["delta_loss"]
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
