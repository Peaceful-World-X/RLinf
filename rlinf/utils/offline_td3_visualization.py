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

"""Visualization helpers for offline TD3 actor/critic snapshots."""

from __future__ import annotations

import json
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.utils.nested_dict_process import put_tensor_device


@dataclass
class UrdfJoint:
    name: str
    joint_type: str
    parent: str
    child: str
    xyz: np.ndarray
    rpy: np.ndarray
    axis: np.ndarray


class SimpleUrdfKinematics:
    """Small URDF FK parser for serial robot chains.

    It supports the joint types used by Piper's URDF: fixed, revolute,
    continuous, and prismatic. Mesh geometry is intentionally ignored; the
    visualizer only needs link origins / TCP positions.
    """

    def __init__(self, urdf_path: str, base_link: str = "base_link"):
        self.urdf_path = str(urdf_path)
        root = ET.parse(urdf_path).getroot()
        self.base_link = base_link
        self.joints: list[UrdfJoint] = []
        for elem in root.findall("joint"):
            origin = elem.find("origin")
            axis = elem.find("axis")
            parent = elem.find("parent")
            child = elem.find("child")
            if parent is None or child is None:
                continue
            xyz = _parse_floats(
                origin.get("xyz", "0 0 0") if origin is not None else "0 0 0"
            )
            rpy = _parse_floats(
                origin.get("rpy", "0 0 0") if origin is not None else "0 0 0"
            )
            axis_xyz = _parse_floats(
                axis.get("xyz", "0 0 1") if axis is not None else "0 0 1"
            )
            self.joints.append(
                UrdfJoint(
                    name=elem.get("name", ""),
                    joint_type=elem.get("type", "fixed"),
                    parent=parent.get("link", ""),
                    child=child.get("link", ""),
                    xyz=xyz,
                    rpy=rpy,
                    axis=axis_xyz,
                )
            )
        self.child_to_joint = {joint.child: joint for joint in self.joints}

    def chain_to(self, tip_link: str) -> list[UrdfJoint]:
        chain: list[UrdfJoint] = []
        link = tip_link
        while link != self.base_link:
            joint = self.child_to_joint.get(link)
            if joint is None:
                raise ValueError(
                    f"Cannot find URDF chain from {self.base_link} to {tip_link}"
                )
            chain.append(joint)
            link = joint.parent
        chain.reverse()
        return chain

    def fk_positions(
        self, q: np.ndarray, tip_link: str = "gripper"
    ) -> tuple[np.ndarray, np.ndarray]:
        chain = self.chain_to(tip_link)
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        transform = np.eye(4, dtype=np.float64)
        positions = [transform[:3, 3].copy()]
        movable_idx = 0
        for joint in chain:
            transform = transform @ _transform_from_xyz_rpy(joint.xyz, joint.rpy)
            if joint.joint_type in {"revolute", "continuous"}:
                angle = q[movable_idx] if movable_idx < len(q) else 0.0
                transform = transform @ _rotation_about_axis(joint.axis, angle)
                movable_idx += 1
            elif joint.joint_type == "prismatic":
                value = q[movable_idx] if movable_idx < len(q) else 0.0
                transform = transform @ _translation(joint.axis * value)
                movable_idx += 1
            positions.append(transform[:3, 3].copy())
        return np.stack(positions, axis=0), transform


def _parse_floats(text: str) -> np.ndarray:
    return np.array([float(v) for v in text.split()], dtype=np.float64)


def _translation(xyz: np.ndarray) -> np.ndarray:
    t = np.eye(4, dtype=np.float64)
    t[:3, 3] = xyz
    return t


def _rotation_matrix_from_rpy(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


def _transform_from_xyz_rpy(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    t = _translation(xyz)
    t[:3, :3] = _rotation_matrix_from_rpy(rpy)
    return t


def _rotation_about_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    norm = np.linalg.norm(axis)
    if norm == 0:
        return np.eye(4, dtype=np.float64)
    x, y, z = axis / norm
    c, s = math.cos(angle), math.sin(angle)
    c1 = 1.0 - c
    rot = np.array(
        [
            [c + x * x * c1, x * y * c1 - z * s, x * z * c1 + y * s],
            [y * x * c1 + z * s, c + y * y * c1, y * z * c1 - x * s],
            [z * x * c1 - y * s, z * y * c1 + x * s, c + z * z * c1],
        ],
        dtype=np.float64,
    )
    t = np.eye(4, dtype=np.float64)
    t[:3, :3] = rot
    return t


def denormalize_absolute_action(
    actions: np.ndarray, low: np.ndarray, high: np.ndarray
) -> np.ndarray:
    actions = np.clip(actions, -1.0, 1.0)
    return (actions + 1.0) / 2.0 * (high - low) + low


def reconstruct_executed_joint_targets(
    actions: np.ndarray,
    start_qpos: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    joint_action_mode: str,
    delta_action_scale: float,
) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float64)
    qpos = np.asarray(start_qpos, dtype=np.float64).reshape(14)
    executed = []
    for raw in actions.reshape(-1, 14):
        clipped = (
            np.clip(raw, -1.0, 1.0)
            if joint_action_mode in {"delta", "absolute_normalized"}
            else raw
        )
        if joint_action_mode == "absolute_normalized":
            target = denormalize_absolute_action(clipped, low, high)
            if delta_action_scale > 0:
                scale = float(delta_action_scale)
                target = np.clip(target, qpos - scale, qpos + scale)
        elif joint_action_mode == "delta":
            target = np.clip(qpos + clipped * float(delta_action_scale), low, high)
        else:
            target = np.clip(raw, low, high)
        executed.append(target.copy())
        qpos = target
    return np.stack(executed, axis=0)


def tcp_trajectories_from_joint_targets(
    joint_targets: np.ndarray,
    kinematics: SimpleUrdfKinematics,
    base_offsets: np.ndarray,
    tip_link: str = "gripper",
) -> dict[str, np.ndarray]:
    left_tcp = []
    right_tcp = []
    for q in joint_targets:
        left_q = np.asarray(q[:7], dtype=np.float64).copy()
        right_q = np.asarray(q[7:14], dtype=np.float64).copy()
        left_links, _ = kinematics.fk_positions(left_q, tip_link=tip_link)
        right_links, _ = kinematics.fk_positions(right_q, tip_link=tip_link)
        left_tcp.append(left_links[-1] + base_offsets[0])
        right_tcp.append(right_links[-1] + base_offsets[1])
    return {"left": np.stack(left_tcp, axis=0), "right": np.stack(right_tcp, axis=0)}


def _split_segments(points: np.ndarray, segment_lengths: list[int] | None):
    if segment_lengths is None:
        yield points
        return
    start = 0
    for length in segment_lengths:
        end = start + int(length)
        if end > start:
            yield points[start:end]
        start = end


def plot_trajectory_3d(
    out_path: str,
    gt_tcp: dict[str, np.ndarray],
    actor_tcp: dict[str, np.ndarray],
    q_actor: np.ndarray,
    segment_lengths: list[int] | None = None,
):
    fig = plt.figure(figsize=(11, 5), dpi=150)
    axes = [
        fig.add_subplot(1, 2, 1, projection="3d"),
        fig.add_subplot(1, 2, 2, projection="3d"),
    ]
    sides = ["left", "right"]
    q_color = np.asarray(q_actor, dtype=np.float64).reshape(-1)
    for ax, side in zip(axes, sides):
        gt = gt_tcp[side]
        actor = actor_tcp[side]
        for seg_idx, segment in enumerate(_split_segments(gt, segment_lengths)):
            ax.plot(
                segment[:, 0],
                segment[:, 1],
                segment[:, 2],
                color="#222222",
                linewidth=2.0,
                label="GT" if seg_idx == 0 else None,
            )
        sc = ax.scatter(
            actor[:, 0],
            actor[:, 1],
            actor[:, 2],
            c=q_color,
            cmap="viridis",
            s=18,
            label="Actor Q",
        )
        for seg_idx, segment in enumerate(_split_segments(actor, segment_lengths)):
            ax.plot(
                segment[:, 0],
                segment[:, 1],
                segment[:, 2],
                color="#2a6fbb",
                linewidth=1.2,
                alpha=0.55,
                label="Actor" if seg_idx == 0 else None,
            )
        ax.set_title(f"{side} TCP")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
        _set_equal_3d(ax, [gt, actor])
        ax.legend(loc="best")
    fig.colorbar(sc, ax=axes, shrink=0.75, label="Q(actor action)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_q_values(
    out_path: str,
    critic_q_data: np.ndarray,
    critic_q_actor: np.ndarray,
    mse: np.ndarray,
    chunk_indices: np.ndarray | None = None,
    mc_return: np.ndarray | None = None,
):
    steps = (
        np.asarray(chunk_indices, dtype=np.int64)
        if chunk_indices is not None
        else np.arange(len(critic_q_data))
    )
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), dpi=150, sharex=True)
    q_line = axes[0].plot(
        steps,
        critic_q_data,
        marker="o",
        label="critic Q(s, data action)",
        color="#222222",
    )
    actor_line = axes[0].plot(
        steps,
        critic_q_actor,
        marker="o",
        label="critic Q(s, actor action)",
        color="#2a6fbb",
    )
    axes[0].set_ylabel("Q / MC return")
    lines = q_line + actor_line
    if mc_return is not None:
        return_line = axes[0].plot(
            steps,
            mc_return,
            marker="s",
            linestyle="--",
            label="MC return",
            color="#2f8f46",
        )
        lines += return_line
    axes[0].legend(lines, [line.get_label() for line in lines], loc="best")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(steps, mse, marker="o", color="#b3472f", label="action MSE")
    axes[1].set_xlabel("trajectory chunk index")
    axes[1].set_ylabel("MSE")
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _resize_uint8_image(image: np.ndarray, size: int) -> np.ndarray:
    image = np.asarray(image)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    if image.shape[-1] > 3:
        image = image[..., :3]
    h, w = image.shape[:2]
    y_idx = np.linspace(0, max(h - 1, 0), int(size)).astype(np.int64)
    x_idx = np.linspace(0, max(w - 1, 0), int(size)).astype(np.int64)
    return image[np.ix_(y_idx, x_idx)]


def extract_three_view_images(
    trajectory: dict,
    chunk_indices: np.ndarray,
    thumb_size: int = 64,
) -> np.ndarray | None:
    """Return [num_chunks, 3, thumb, thumb, 3] uint8 curr_obs images."""

    curr_obs = trajectory.get("curr_obs", {})
    if not isinstance(curr_obs, dict):
        return None
    main = curr_obs.get("main_images", None)
    extra = curr_obs.get("extra_view_images", None)
    if not torch.is_tensor(main) or not torch.is_tensor(extra):
        return None
    main_np = main.detach().cpu().numpy()
    extra_np = extra.detach().cpu().numpy()
    chunk_indices = np.asarray(chunk_indices, dtype=np.int64).reshape(-1)
    views = []
    for idx in chunk_indices:
        if idx < 0 or idx >= main_np.shape[0] or idx >= extra_np.shape[0]:
            continue
        main_img = main_np[idx, 0]
        extra_views = extra_np[idx, 0]
        if extra_views.shape[0] < 2:
            return None
        views.append(
            [
                _resize_uint8_image(main_img, thumb_size),
                _resize_uint8_image(extra_views[0], thumb_size),
                _resize_uint8_image(extra_views[1], thumb_size),
            ]
        )
    if len(views) != len(chunk_indices):
        return None
    return np.asarray(views, dtype=np.uint8)


def load_key_segment_start_map(path: str | None) -> dict[str, int]:
    if not path:
        return {}
    path_obj = Path(path)
    if not path_obj.is_file():
        return {}
    with open(path_obj, "r", encoding="utf-8") as f:
        payload = json.load(f)
    out: dict[str, int] = {}
    for item in payload.get("trajectories", []):
        name = item.get("file", None)
        if name is None:
            continue
        out[str(name)] = int(item.get("start_chunk", 0))
    return out


def plot_critic_timeline_with_images(
    out_path: str,
    chunk_indices: np.ndarray,
    critic_q_data: np.ndarray,
    critic_q_actor: np.ndarray,
    mc_return: np.ndarray,
    rewards: np.ndarray | None = None,
    critic_mc_mse: np.ndarray | None = None,
    images: np.ndarray | None = None,
    train_mask: np.ndarray | None = None,
    title: str | None = None,
) -> None:
    """Plot whole-trajectory critic values aligned with three camera views."""

    chunk_indices = np.asarray(chunk_indices, dtype=np.int64).reshape(-1)
    n = int(len(chunk_indices))
    if n == 0:
        return
    x = np.arange(n, dtype=np.int64)
    q_data = np.asarray(critic_q_data, dtype=np.float64).reshape(-1)[:n]
    q_actor = np.asarray(critic_q_actor, dtype=np.float64).reshape(-1)[:n]
    mc = np.asarray(mc_return, dtype=np.float64).reshape(-1)[:n]
    rewards_arr = (
        None
        if rewards is None
        else np.asarray(rewards, dtype=np.float64).reshape(-1)[:n]
    )
    loss_arr = (
        np.square(q_data - mc)
        if critic_mc_mse is None
        else np.asarray(critic_mc_mse, dtype=np.float64).reshape(-1)[:n]
    )
    train_arr = (
        None
        if train_mask is None
        else np.asarray(train_mask, dtype=bool).reshape(-1)[:n]
    )

    has_images = images is not None and len(images) == n
    height_ratios = [2.4, 1.0] + ([0.9, 0.9, 0.9] if has_images else [])
    width = min(max(12.0, 0.42 * n), 42.0)
    fig, axes = plt.subplots(
        len(height_ratios),
        1,
        figsize=(width, 2.0 + sum(height_ratios) * 1.15),
        dpi=150,
        sharex=True,
        gridspec_kw={"height_ratios": height_ratios},
    )
    axes = np.asarray(axes).reshape(-1)

    def shade_train_regions(ax):
        if train_arr is None or not train_arr.any():
            return
        active = np.flatnonzero(train_arr)
        start = int(active[0])
        prev = int(active[0])
        for idx in active[1:]:
            idx = int(idx)
            if idx != prev + 1:
                ax.axvspan(start - 0.5, prev + 0.5, color="#f2c94c", alpha=0.18)
                start = idx
            prev = idx
        ax.axvspan(start - 0.5, prev + 0.5, color="#f2c94c", alpha=0.18)

    ax_q = axes[0]
    shade_train_regions(ax_q)
    ax_q.plot(x, q_data, color="#222222", linewidth=1.7, label="Q(s, data action)")
    ax_q.plot(
        x,
        q_actor,
        color="#2a6fbb",
        linewidth=1.4,
        linestyle="--",
        label="Q(s, actor action)",
    )
    ax_q.plot(
        x,
        mc,
        color="#2f8f46",
        linewidth=1.5,
        linestyle=":",
        label="MC return",
    )
    if rewards_arr is not None and np.any(rewards_arr > 0):
        reward_x = x[rewards_arr > 0]
        ax_q.scatter(
            reward_x,
            mc[rewards_arr > 0],
            color="#c0392b",
            marker="*",
            s=95,
            zorder=5,
            label="reward > 0",
        )
    ax_q.set_ylabel("Q / return")
    ax_q.grid(True, alpha=0.25)
    ax_q.legend(loc="best", ncol=4)
    ax_q.set_title(title or "Critic Q timeline")

    ax_loss = axes[1]
    shade_train_regions(ax_loss)
    ax_loss.plot(
        x,
        loss_arr,
        color="#b3472f",
        linewidth=1.4,
        marker="o",
        markersize=2.5,
        label="(Q_data - MC return)^2",
    )
    ax_loss.set_ylabel("Q-MC MSE")
    ax_loss.grid(True, alpha=0.25)
    ax_loss.legend(loc="best")
    if n <= 100:
        finite_loss = loss_arr[np.isfinite(loss_arr)]
        y_span = (
            max(float(finite_loss.max() - finite_loss.min()), 1e-8)
            if finite_loss.size
            else 1.0
        )
        for pos, value in enumerate(loss_arr):
            if not np.isfinite(value):
                continue
            ax_loss.text(
                pos,
                float(value) + 0.025 * y_span,
                f"{float(value):.1e}",
                fontsize=4.5,
                rotation=90,
                ha="center",
                va="bottom",
                color="#7a2f20",
            )

    if has_images:
        view_names = ["main", "extra0", "extra1"]
        for view_idx in range(3):
            ax_img = axes[2 + view_idx]
            strip = np.concatenate([images[i, view_idx] for i in range(n)], axis=1)
            ax_img.imshow(strip, aspect="auto", extent=(-0.5, n - 0.5, 0, 1))
            ax_img.set_yticks([])
            ax_img.set_ylabel(
                view_names[view_idx], rotation=0, labelpad=28, va="center"
            )
            for pos in range(n + 1):
                ax_img.axvline(pos - 0.5, color="white", linewidth=0.25, alpha=0.45)
            if train_arr is not None and train_arr.any():
                for pos, trained in enumerate(train_arr):
                    if trained:
                        ax_img.axvspan(
                            pos - 0.5, pos + 0.5, color="#f2c94c", alpha=0.12
                        )

    tick_stride = max(1, int(math.ceil(n / 18)))
    tick_positions = x[::tick_stride]
    tick_labels = [str(int(v)) for v in chunk_indices[::tick_stride]]
    axes[-1].set_xticks(tick_positions)
    axes[-1].set_xticklabels(tick_labels, rotation=0)
    axes[-1].set_xlabel("trajectory chunk index")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _set_equal_3d(ax, arrays: list[np.ndarray]) -> None:
    pts = np.concatenate(arrays, axis=0)
    centers = pts.mean(axis=0)
    ranges = pts.max(axis=0) - pts.min(axis=0)
    radius = max(float(ranges.max()) / 2.0, 1e-3)
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def _equal_3d_limits(
    arrays: list[np.ndarray],
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    pts = np.concatenate(
        [np.asarray(arr, dtype=np.float64).reshape(-1, 3) for arr in arrays], axis=0
    )
    centers = pts.mean(axis=0)
    ranges = pts.max(axis=0) - pts.min(axis=0)
    radius = max(float(ranges.max()) / 2.0, 1e-3)
    return (
        (float(centers[0] - radius), float(centers[0] + radius)),
        (float(centers[1] - radius), float(centers[1] + radius)),
        (float(centers[2] - radius), float(centers[2] + radius)),
    )


def _apply_3d_limits(ax, limits) -> None:
    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])
    ax.set_zlim(*limits[2])


def trajectory_id_from_name(path: Path) -> int:
    match = re.search(r"trajectory_(\d+)_", path.name)
    if not match:
        return -1
    return int(match.group(1))


def load_validation_trajectories(
    data_paths: list[str],
    count: int,
    success: bool | None = None,
) -> list[tuple[Path, dict]]:
    candidates: list[Path] = []
    for data_path in data_paths:
        path = Path(data_path)
        if path.is_dir():
            candidates.extend(path.glob("trajectory_*.pt"))
    candidates = sorted(candidates, key=trajectory_id_from_name)
    if success is not None:
        filtered = []
        for path in candidates:
            trajectory = torch.load(path, map_location="cpu")
            is_success = bool(
                float(trajectory.get("rewards", torch.zeros(1)).float().max()) > 0.0
            )
            if is_success == success:
                filtered.append((path, trajectory))
        return filtered[:count] if count > 0 else filtered
    if count > 0:
        candidates = candidates[:count]
    out = []
    for path in candidates:
        out.append((path, torch.load(path, map_location="cpu")))
    return out


def build_validation_batch(
    trajectory: dict,
    chunk_indices: list[int],
    task_description: str,
    action_horizon: int,
) -> dict[str, Any]:
    batch: dict[str, Any] = {}
    tensor_keys = ["actions", "rewards", "terminations", "truncations", "dones"]
    for key in tensor_keys:
        if key in trajectory and torch.is_tensor(trajectory[key]):
            batch[key] = torch.stack(
                [trajectory[key][idx, 0] for idx in chunk_indices], dim=0
            )
    for obs_key in ["curr_obs", "next_obs", "forward_inputs"]:
        if obs_key not in trajectory or not isinstance(trajectory[obs_key], dict):
            continue
        nested = {}
        for key, value in trajectory[obs_key].items():
            if torch.is_tensor(value):
                nested[key] = torch.stack(
                    [value[idx, 0] for idx in chunk_indices], dim=0
                )
        if nested:
            batch[obs_key] = nested
    if "curr_obs" in batch:
        bsz = len(chunk_indices)
        batch["curr_obs"]["task_descriptions"] = [task_description] * bsz
    if "next_obs" in batch:
        bsz = len(chunk_indices)
        batch["next_obs"]["task_descriptions"] = [task_description] * bsz
    if "actions" in batch:
        batch["actions"] = batch["actions"].reshape(
            batch["actions"].shape[0], action_horizon, -1
        )
    return batch


def make_validation_indices(
    length: int,
    count: int,
    rewards: np.ndarray | None = None,
    terminations: np.ndarray | None = None,
    mode: str = "terminal_window",
) -> list[int]:
    if length <= 0:
        return []
    count = min(max(int(count), 1), length)
    mode = str(mode or "terminal_window")
    if mode == "linspace":
        if count <= 1:
            return [0]
        last = max(0, length - 1)
        indices = np.linspace(0, last, num=count, dtype=int).tolist()
        seen = []
        for idx in indices:
            if idx not in seen:
                seen.append(idx)
        return seen

    end = length - 1
    if rewards is not None:
        reward_indices = np.flatnonzero(np.asarray(rewards).reshape(-1) > 0)
        if len(reward_indices) > 0:
            end = int(reward_indices[0])
        elif terminations is not None:
            done_indices = np.flatnonzero(np.asarray(terminations).reshape(-1) > 0)
            if len(done_indices) > 0:
                end = int(done_indices[0])
    elif terminations is not None:
        done_indices = np.flatnonzero(np.asarray(terminations).reshape(-1) > 0)
        if len(done_indices) > 0:
            end = int(done_indices[0])
    if count <= 1:
        return [end]
    start = max(0, end - count + 1)
    return list(range(start, end + 1))


def compute_chunk_mc_returns(
    rewards: np.ndarray,
    terminations: np.ndarray | None,
    gamma: float,
    action_horizon: int,
) -> np.ndarray:
    rewards = np.asarray(rewards, dtype=np.float64).reshape(-1)
    if terminations is None:
        dones = np.zeros_like(rewards, dtype=bool)
    else:
        dones = np.asarray(terminations).reshape(-1).astype(bool)
    # One reward is recorded per high-level action chunk, so gamma is already
    # the chunk-level discount. The low-level horizon only describes the action
    # representation inside that transition.
    discount = float(gamma)
    returns = np.zeros_like(rewards, dtype=np.float64)
    running = 0.0
    for idx in range(len(rewards) - 1, -1, -1):
        running = float(rewards[idx]) + (0.0 if dones[idx] else discount * running)
        returns[idx] = running
    return returns


DELTA_ACTION_MASK = np.array([True] * 6 + [False] + [True] * 6 + [False], dtype=bool)


def load_action_norm_stats(
    norm_stats_path: str, std_floor: float = 1e-6
) -> tuple[np.ndarray, np.ndarray]:
    with open(norm_stats_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    stats = payload.get("norm_stats", payload)["actions"]
    mean = np.asarray(stats["mean"][:14], dtype=np.float64)
    std = np.asarray(stats["std"][:14], dtype=np.float64)
    std = np.where(np.abs(std) < float(std_floor), 1.0, std)
    return mean, std


def normalized_delta_to_absolute(
    actions: np.ndarray,
    start_qpos: np.ndarray,
    action_mean: np.ndarray,
    action_std: np.ndarray,
) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float64).reshape(-1, 14)
    qpos = np.asarray(start_qpos, dtype=np.float64).reshape(14)
    mean = np.asarray(action_mean, dtype=np.float64).reshape(14)
    std = np.asarray(action_std, dtype=np.float64).reshape(14)
    delta = actions * (std + 1e-6) + mean
    absolute = delta.copy()
    absolute[:, DELTA_ACTION_MASK] = (
        absolute[:, DELTA_ACTION_MASK] + qpos[DELTA_ACTION_MASK]
    )
    return absolute


def actions_to_absolute_joint_targets(
    actions: np.ndarray,
    start_qpos: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    joint_action_mode: str,
    delta_action_scale: float,
    action_mean: np.ndarray | None = None,
    action_std: np.ndarray | None = None,
    clip_to_joint_limits: bool = False,
) -> np.ndarray:
    if joint_action_mode == "normalized_delta":
        if action_mean is None or action_std is None:
            raise ValueError(
                "normalized_delta visualization requires action norm stats."
            )
        absolute = normalized_delta_to_absolute(
            actions, start_qpos, action_mean, action_std
        )
        if clip_to_joint_limits:
            return np.clip(absolute, low, high)
        return absolute
    return reconstruct_executed_joint_targets(
        actions,
        start_qpos,
        low,
        high,
        joint_action_mode,
        delta_action_scale,
    )


def plot_action_mse_heatmaps(
    out_path: str,
    mse_matrix: np.ndarray,
    chunk_indices: np.ndarray,
    chunk_mse: np.ndarray,
) -> None:
    mse_matrix = np.asarray(mse_matrix, dtype=np.float64)
    num_chunks = mse_matrix.shape[0]
    cols = min(3, max(1, num_chunks))
    rows = int(math.ceil(num_chunks / cols))
    fig, axes = plt.subplots(
        rows, cols, figsize=(4.2 * cols, 3.0 * rows), dpi=150, squeeze=False
    )
    vmax = max(float(np.nanmax(mse_matrix)), 1e-8)
    last_im = None
    for idx in range(rows * cols):
        ax = axes[idx // cols][idx % cols]
        if idx >= num_chunks:
            ax.axis("off")
            continue
        data = mse_matrix[idx].T
        last_im = ax.imshow(
            data, aspect="auto", origin="lower", cmap="magma", vmin=0.0, vmax=vmax
        )
        ax.set_title(
            f"chunk {int(chunk_indices[idx])}, mse={float(chunk_mse[idx]):.4g}"
        )
        ax.set_xlabel("small step")
        ax.set_ylabel("joint")
        ax.set_xticks(np.arange(data.shape[1]))
        ax.set_yticks(np.arange(data.shape[0]))
    if last_im is not None:
        fig.colorbar(last_im, ax=axes, shrink=0.8, label="squared error")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_boundary_trajectory_3d(
    out_path: str,
    state_tcp: dict[str, np.ndarray],
    data_boundary_tcp: dict[str, np.ndarray],
    actor_boundary_tcp: dict[str, np.ndarray] | None,
    chunk_indices: np.ndarray,
    q_data: np.ndarray | None = None,
    q_actor: np.ndarray | None = None,
) -> None:
    """Plot a readable high-level trajectory instead of overlapping forecast chunks.

    Each saved action is a chunk forecast. The first substep is the chunk boundary
    action that aligns with the high-level state sequence, so this view is the
    least ambiguous way to sanity-check denormalized PT/actor trajectories.
    """

    chunk_indices = np.asarray(chunk_indices, dtype=np.int64).reshape(-1)
    q_data_arr = (
        None if q_data is None else np.asarray(q_data, dtype=np.float64).reshape(-1)
    )
    q_actor_arr = (
        None if q_actor is None else np.asarray(q_actor, dtype=np.float64).reshape(-1)
    )

    fig = plt.figure(figsize=(12, 5.5), dpi=150)
    axes = [
        fig.add_subplot(1, 2, 1, projection="3d"),
        fig.add_subplot(1, 2, 2, projection="3d"),
    ]
    for ax, side in zip(axes, ["left", "right"]):
        state = np.asarray(state_tcp[side], dtype=np.float64)
        data = np.asarray(data_boundary_tcp[side], dtype=np.float64)
        actor = (
            None
            if actor_boundary_tcp is None
            else np.asarray(actor_boundary_tcp[side], dtype=np.float64)
        )
        arrays = [state, data]

        ax.scatter(
            state[:, 0],
            state[:, 1],
            state[:, 2],
            color="#222222",
            s=12,
            alpha=0.65,
            label="robot state",
        )
        ax.scatter(
            state[0, 0],
            state[0, 1],
            state[0, 2],
            marker="o",
            color="#2f8f46",
            s=42,
            label="start",
        )
        ax.scatter(
            state[-1, 0],
            state[-1, 1],
            state[-1, 2],
            marker="x",
            color="#b3472f",
            s=58,
            label="end",
        )
        ax.scatter(
            data[:, 0],
            data[:, 1],
            data[:, 2],
            color="#6f2b8c",
            marker="o",
            s=22,
            label="pt denorm substep0",
        )
        if actor is not None:
            arrays.append(actor)
            ax.scatter(
                actor[:, 0],
                actor[:, 1],
                actor[:, 2],
                color="#2a6fbb",
                marker="x",
                s=28,
                label="actor denorm substep0",
            )

        for local_idx, global_idx in enumerate(chunk_indices):
            if local_idx >= len(data):
                break
            label = f"{local_idx}({int(global_idx)})"
            if q_data_arr is not None and local_idx < len(q_data_arr):
                label += f"\nQd {float(q_data_arr[local_idx]):.3g}"
            if q_actor_arr is not None and local_idx < len(q_actor_arr):
                label += f"\nQa {float(q_actor_arr[local_idx]):.3g}"
            ax.text(
                data[local_idx, 0],
                data[local_idx, 1],
                data[local_idx, 2],
                label,
                fontsize=6,
                color="#111111",
            )

        ax.set_title(f"{side} TCP (high-level boundary view)")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
        _set_equal_3d(ax, arrays)
        ax.legend(loc="best")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_pt_state_trajectory_3d(
    out_path: str,
    state_tcp: dict[str, np.ndarray],
    pt_boundary_tcp: dict[str, np.ndarray],
) -> None:
    """Plot recorded states as points against denormalized PT action[t, 0]."""

    fig = plt.figure(figsize=(12, 5.5), dpi=150)
    axes = [
        fig.add_subplot(1, 2, 1, projection="3d"),
        fig.add_subplot(1, 2, 2, projection="3d"),
    ]
    for ax, side in zip(axes, ["left", "right"]):
        state = np.asarray(state_tcp[side], dtype=np.float64)
        pt = np.asarray(pt_boundary_tcp[side], dtype=np.float64)
        ax.scatter(
            state[:, 0],
            state[:, 1],
            state[:, 2],
            color="#222222",
            s=12,
            alpha=0.75,
            label="robot state",
        )
        ax.scatter(
            pt[:, 0],
            pt[:, 1],
            pt[:, 2],
            color="#6f2b8c",
            s=16,
            alpha=0.8,
            label="pt denorm action[t,0]",
        )
        ax.scatter(
            state[0, 0],
            state[0, 1],
            state[0, 2],
            marker="o",
            color="#2f8f46",
            s=42,
            label="start",
        )
        ax.scatter(
            state[-1, 0],
            state[-1, 1],
            state[-1, 2],
            marker="x",
            color="#b3472f",
            s=58,
            label="end",
        )
        ax.set_title(f"{side} TCP (PT substep0 sanity check)")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
        _set_equal_3d(ax, [state, pt])
        ax.legend(loc="best")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_action_chunk_segments_3d(
    out_path: str,
    state_tcp: dict[str, np.ndarray],
    data_tcp_segments: list[dict[str, np.ndarray]],
    chunk_indices: np.ndarray,
    actor_tcp_segments: list[dict[str, np.ndarray]] | None = None,
    q_data: np.ndarray | None = None,
    q_actor: np.ndarray | None = None,
) -> None:
    """Plot each saved 10-step action chunk as an independent segment.

    In these PT files, each high-level robot state owns one forecast/action
    chunk. Adjacent chunks are not a continuous executed path, so this plot
    deliberately leaves gaps between chunks and uses the robot states only as
    point references.
    """

    has_actor = actor_tcp_segments is not None

    fig = plt.figure(figsize=(12, 5.5), dpi=150)
    axes = [
        fig.add_subplot(1, 2, 1, projection="3d"),
        fig.add_subplot(1, 2, 2, projection="3d"),
    ]
    for ax, side in zip(axes, ["left", "right"]):
        state = np.asarray(state_tcp[side], dtype=np.float64)
        arrays = [state]
        ax.scatter(
            state[:, 0],
            state[:, 1],
            state[:, 2],
            color="#222222",
            s=10,
            alpha=0.65,
            label="robot state",
        )
        ax.scatter(
            state[0, 0],
            state[0, 1],
            state[0, 2],
            marker="o",
            color="#2f8f46",
            s=54,
            label="trajectory start",
        )
        ax.scatter(
            state[-1, 0],
            state[-1, 1],
            state[-1, 2],
            marker="x",
            color="#b3472f",
            s=74,
            label="trajectory end",
        )

        for idx, data_seg in enumerate(data_tcp_segments):
            data = np.asarray(data_seg[side], dtype=np.float64)
            ax.plot(
                data[:, 0],
                data[:, 1],
                data[:, 2],
                color="#6f2b8c",
                linewidth=1.3,
                linestyle="-",
                alpha=0.42,
                label="pt action chunk" if idx == 0 else None,
            )
            arrays.append(data)

            if has_actor and idx < len(actor_tcp_segments):
                actor = np.asarray(actor_tcp_segments[idx][side], dtype=np.float64)
                ax.plot(
                    actor[:, 0],
                    actor[:, 1],
                    actor[:, 2],
                    color="#2a6fbb",
                    linewidth=1.15,
                    linestyle="--",
                    alpha=0.42,
                    label="actor chunk" if idx == 0 else None,
                )
                arrays.append(actor)

        ax.set_title(f"{side} TCP action chunks")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
        _set_equal_3d(ax, arrays)
        ax.legend(loc="best")
    handles = [
        plt.Line2D(
            [0], [0], color="#6f2b8c", linestyle="-", label="solid: pt action chunk"
        ),
        plt.Line2D(
            [0], [0], color="#2a6fbb", linestyle="--", label="dashed: actor chunk"
        ),
    ]
    if has_actor:
        fig.legend(handles=handles, loc="lower center", ncol=2)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_action_chunk_triplet_segments_3d(
    out_path: str,
    state_tcp: dict[str, np.ndarray],
    pt_tcp_segments: list[dict[str, np.ndarray]],
    gt_reconstructed_tcp_segments: list[dict[str, np.ndarray]],
    actor_tcp_segments: list[dict[str, np.ndarray]],
) -> None:
    """Plot PT, GT, and actor chunks in the global TCP trajectory view.

    This intentionally matches the earlier sanity-check plot style: robot-state
    TCP positions set the global task-scale context, while action chunks are
    drawn as independent forecast/execution segments. Adjacent chunks are not
    connected to each other.
    """

    fig = plt.figure(figsize=(12, 5.5), dpi=150)
    axes = [
        fig.add_subplot(1, 2, 1, projection="3d"),
        fig.add_subplot(1, 2, 2, projection="3d"),
    ]
    for ax, side in zip(axes, ["left", "right"]):
        state = np.asarray(state_tcp[side], dtype=np.float64)
        arrays = [state]
        ax.scatter(
            state[:, 0],
            state[:, 1],
            state[:, 2],
            color="#222222",
            s=10,
            alpha=0.65,
            label="robot state",
        )
        ax.scatter(
            state[0, 0],
            state[0, 1],
            state[0, 2],
            marker="o",
            color="#2f8f46",
            s=54,
            label="trajectory start",
        )
        ax.scatter(
            state[-1, 0],
            state[-1, 1],
            state[-1, 2],
            marker="x",
            color="#b3472f",
            s=74,
            label="trajectory end",
        )

        rows = [
            ("PT absolute", pt_tcp_segments, "#6f2b8c", "-", 1.8, 0.50),
            (
                "GT reconstructed",
                gt_reconstructed_tcp_segments,
                "#d17a22",
                "--",
                1.5,
                0.70,
            ),
            ("Actor reconstructed", actor_tcp_segments, "#2a6fbb", ":", 1.8, 0.85),
        ]
        for label, segments, color, linestyle, linewidth, alpha in rows:
            for idx, seg in enumerate(segments):
                points = np.asarray(seg[side], dtype=np.float64)
                ax.plot(
                    points[:, 0],
                    points[:, 1],
                    points[:, 2],
                    color=color,
                    linewidth=linewidth,
                    linestyle=linestyle,
                    alpha=alpha,
                    label=label if idx == 0 else None,
                )
                arrays.append(points)

        ax.set_title(f"{side} TCP action chunks")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
        _set_equal_3d(ax, arrays)
        ax.legend(loc="best")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_pt_gt_reconstruction_segments_3d(
    out_path: str,
    state_tcp: dict[str, np.ndarray],
    pt_tcp_segments: list[dict[str, np.ndarray]],
    gt_reconstructed_tcp_segments: list[dict[str, np.ndarray]],
    max_abs_error: float | None = None,
    mean_abs_error: float | None = None,
) -> None:
    """Plot the data-transform invariant: PT absolute chunks vs reconstructed GT.

    This plot intentionally excludes actor predictions. It only checks whether
    OpenPI-style Unnormalize + AbsoluteActions(mask) reconstructs the absolute
    action chunks saved in the PT file.
    """

    fig = plt.figure(figsize=(12, 5.5), dpi=150)
    axes = [
        fig.add_subplot(1, 2, 1, projection="3d"),
        fig.add_subplot(1, 2, 2, projection="3d"),
    ]
    for ax, side in zip(axes, ["left", "right"]):
        state = np.asarray(state_tcp[side], dtype=np.float64)
        arrays = [state]
        ax.scatter(
            state[:, 0],
            state[:, 1],
            state[:, 2],
            color="#222222",
            s=10,
            alpha=0.65,
            label="robot state",
        )
        ax.scatter(
            state[0, 0],
            state[0, 1],
            state[0, 2],
            marker="o",
            color="#2f8f46",
            s=54,
            label="trajectory start",
        )
        ax.scatter(
            state[-1, 0],
            state[-1, 1],
            state[-1, 2],
            marker="x",
            color="#b3472f",
            s=74,
            label="trajectory end",
        )

        rows = [
            ("PT absolute", pt_tcp_segments, "#6f2b8c", "-", 2.0, 0.48),
            (
                "GT reconstructed",
                gt_reconstructed_tcp_segments,
                "#d17a22",
                "--",
                1.7,
                0.88,
            ),
        ]
        for label, segments, color, linestyle, linewidth, alpha in rows:
            for idx, seg in enumerate(segments):
                points = np.asarray(seg[side], dtype=np.float64)
                ax.plot(
                    points[:, 0],
                    points[:, 1],
                    points[:, 2],
                    color=color,
                    linewidth=linewidth,
                    linestyle=linestyle,
                    alpha=alpha,
                    label=label if idx == 0 else None,
                )
                arrays.append(points)

        title = f"{side} TCP PT/GT reconstruction"
        if max_abs_error is not None:
            title += f"\nmax abs joint err={max_abs_error:.2e}"
        ax.set_title(title)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
        _set_equal_3d(ax, arrays)
        ax.legend(loc="best")
    if mean_abs_error is not None:
        fig.suptitle(
            f"PT absolute vs GT reconstructed, mean abs joint err={mean_abs_error:.2e}",
            y=0.98,
        )
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_action_chunk_triplet_overlay_3d(
    out_path: str,
    state_tcp: dict[str, np.ndarray],
    pt_tcp_segments: list[dict[str, np.ndarray]],
    gt_reconstructed_tcp_segments: list[dict[str, np.ndarray]],
    actor_tcp_segments: list[dict[str, np.ndarray]],
    side_limits: dict[
        str, tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    ],
) -> None:
    fig = plt.figure(figsize=(12.5, 5.8), dpi=150)
    rows = [
        ("PT absolute", pt_tcp_segments, "#5b2a86", "-"),
        ("GT reconstructed", gt_reconstructed_tcp_segments, "#d17a22", "--"),
        ("Actor reconstructed", actor_tcp_segments, "#2a6fbb", ":"),
    ]
    for col_idx, side in enumerate(["left", "right"]):
        ax = fig.add_subplot(1, 2, col_idx + 1, projection="3d")
        state = np.asarray(state_tcp[side], dtype=np.float64)
        ax.scatter(
            state[:, 0],
            state[:, 1],
            state[:, 2],
            color="#111111",
            s=5,
            alpha=0.15,
            label="robot state",
        )
        ax.scatter(
            state[0, 0],
            state[0, 1],
            state[0, 2],
            marker="o",
            facecolors="white",
            edgecolors="#111111",
            s=50,
            label="trajectory start",
        )
        ax.scatter(
            state[-1, 0],
            state[-1, 1],
            state[-1, 2],
            marker="x",
            color="#111111",
            s=62,
            label="trajectory end",
        )
        for label, segments, color, linestyle in rows:
            for idx, seg in enumerate(segments):
                points = np.asarray(seg[side], dtype=np.float64)
                ax.plot(
                    points[:, 0],
                    points[:, 1],
                    points[:, 2],
                    color=color,
                    linewidth=2.2,
                    linestyle=linestyle,
                    alpha=0.9,
                    label=label if idx == 0 else None,
                )
        ax.set_title(f"{side} TCP overlay")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
        _apply_3d_limits(ax, side_limits[side])
        ax.view_init(elev=24, azim=-58 if side == "left" else -124)
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_trajectory_3d_with_state(
    out_path: str,
    state_tcp: dict[str, np.ndarray],
    data_tcp_segments: list[dict[str, np.ndarray]],
    actor_tcp_segments: list[dict[str, np.ndarray]],
    chunk_indices: np.ndarray,
    q_data: np.ndarray,
    q_actor: np.ndarray,
) -> None:
    fig = plt.figure(figsize=(12, 5.5), dpi=150)
    axes = [
        fig.add_subplot(1, 2, 1, projection="3d"),
        fig.add_subplot(1, 2, 2, projection="3d"),
    ]
    for ax, side in zip(axes, ["left", "right"]):
        state = state_tcp[side]
        ax.scatter(
            state[:, 0],
            state[:, 1],
            state[:, 2],
            color="#222222",
            s=10,
            alpha=0.65,
            label="robot state",
        )
        ax.scatter(
            state[0, 0],
            state[0, 1],
            state[0, 2],
            marker="o",
            color="#2f8f46",
            s=54,
            label="trajectory start",
        )
        ax.scatter(
            state[-1, 0],
            state[-1, 1],
            state[-1, 2],
            marker="x",
            color="#b3472f",
            s=74,
            label="trajectory end",
        )
        arrays = [state]
        for idx, (data_seg, actor_seg) in enumerate(
            zip(data_tcp_segments, actor_tcp_segments)
        ):
            data = data_seg[side]
            actor = actor_seg[side]
            ax.plot(
                data[:, 0],
                data[:, 1],
                data[:, 2],
                color="#6f2b8c",
                linewidth=1.3,
                linestyle="-",
                alpha=0.42,
                label="pt action chunk" if idx == 0 else None,
            )
            ax.plot(
                actor[:, 0],
                actor[:, 1],
                actor[:, 2],
                color="#2a6fbb",
                linewidth=1.15,
                linestyle="--",
                alpha=0.42,
                label="actor chunk" if idx == 0 else None,
            )
            arrays.extend([data, actor])
        ax.set_title(f"{side} TCP")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
        _set_equal_3d(ax, arrays)
        ax.legend(loc="best")
    handles = [
        plt.Line2D(
            [0], [0], color="#6f2b8c", linestyle="-", label="solid: pt action chunk"
        ),
        plt.Line2D(
            [0], [0], color="#2a6fbb", linestyle="--", label="dashed: actor chunk"
        ),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _trajectory_1d(trajectory: dict, key: str, default: float = 0.0) -> np.ndarray:
    value = trajectory.get(key, None)
    if value is None:
        length = int(trajectory["actions"].shape[0])
        return np.full((length,), default, dtype=np.float64)
    if torch.is_tensor(value):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=np.float64).reshape(-1)


def evaluate_validation_trajectory(
    worker,
    trajectory: dict,
    chunk_count: int,
) -> dict[str, np.ndarray]:
    action_horizon = int(
        worker.cfg.actor.model.get(
            "action_horizon", worker.cfg.actor.model.num_action_chunks
        )
    )
    length = int(trajectory["actions"].shape[0])
    rewards_all = _trajectory_1d(trajectory, "rewards")
    terminations_all = _trajectory_1d(trajectory, "terminations")
    if "dones" in trajectory:
        terminations_all = np.maximum(
            terminations_all, _trajectory_1d(trajectory, "dones")
        )
    if "truncations" in trajectory:
        terminations_all = np.maximum(
            terminations_all, _trajectory_1d(trajectory, "truncations")
        )
    viz_cfg = worker.cfg.runner.get("offline_validation_visualization", {})
    index_mode = str(viz_cfg.get("index_mode", "terminal_window"))
    if bool(viz_cfg.get("whole_trajectory", False)):
        chunk_indices = list(range(length))
    else:
        chunk_indices = make_validation_indices(
            length,
            chunk_count,
            rewards=rewards_all,
            terminations=terminations_all,
            mode=index_mode,
        )
    mc_returns = compute_chunk_mc_returns(
        rewards_all,
        terminations_all,
        float(worker.cfg.algorithm.gamma),
        action_horizon,
    )
    batch = build_validation_batch(
        trajectory,
        chunk_indices,
        worker.cfg.algorithm.get("task_descriptions", ""),
        action_horizon,
    )
    batch = put_tensor_device(batch, worker.device)
    dtype = worker.torch_dtype
    with torch.no_grad():
        curr_obs = batch["curr_obs"]
        visual_feat = worker.build_visual_feat_fn(worker.get_visual_input(curr_obs))
        gt_actions = worker.reshape_action_fn(
            batch["actions"].to(worker.device, dtype=dtype),
            "validation.actions",
        )
        ref_action = worker.get_ref_action(
            curr_obs, gt_actions, "validation.ref_action"
        )
        actor_actions, actor_aux = worker.model(
            forward_type=ForwardType.TD3,
            mode="actor",
            visual_feat=visual_feat,
            robot_state=worker.get_robot_state(curr_obs),
            ref_action=ref_action,
            ref_action_dropout_p=0.0,
            use_target=False,
            compute_recon_loss=False,
        )
        critic_rl_state = actor_aux.get(
            "critic_rl_state", actor_aux["rl_state"]
        ).detach()
        gt_q1, gt_q2 = worker.model(
            forward_type=ForwardType.TD3_Q,
            rl_state=critic_rl_state,
            action=gt_actions,
        )
        actor_q1, actor_q2 = worker.model(
            forward_type=ForwardType.TD3_Q,
            rl_state=critic_rl_state,
            action=actor_actions,
        )
        critic_q_data_q1 = gt_q1.reshape(-1).detach().float().cpu().numpy()
        critic_q_data_q2 = gt_q2.reshape(-1).detach().float().cpu().numpy()
        critic_q_actor_q1 = actor_q1.reshape(-1).detach().float().cpu().numpy()
        critic_q_actor_q2 = actor_q2.reshape(-1).detach().float().cpu().numpy()
        critic_q_data = (
            torch.minimum(gt_q1, gt_q2).reshape(-1).detach().float().cpu().numpy()
        )
        critic_q_actor = (
            torch.minimum(actor_q1, actor_q2).reshape(-1).detach().float().cpu().numpy()
        )
        mse_tensor = torch.nn.functional.mse_loss(
            actor_actions, gt_actions, reduction="none"
        )
        mse_matrix = mse_tensor.detach().float().cpu().numpy()
        mse = (
            mse_tensor.reshape(mse_tensor.shape[0], -1)
            .mean(dim=-1)
            .detach()
            .float()
            .cpu()
            .numpy()
        )
    states = torch.stack(
        [trajectory["curr_obs"]["states"][idx, 0] for idx in chunk_indices], dim=0
    )
    rewards = torch.stack(
        [trajectory["rewards"][idx, 0] for idx in chunk_indices], dim=0
    )
    terminations = torch.stack(
        [trajectory["terminations"][idx, 0] for idx in chunk_indices], dim=0
    )
    return {
        "chunk_indices": np.asarray(chunk_indices, dtype=np.int64),
        "start_states": states.detach().float().cpu().numpy(),
        "gt_actions": gt_actions.detach().float().cpu().numpy(),
        "actor_actions": actor_actions.detach().float().cpu().numpy(),
        "critic_q_data": critic_q_data,
        "critic_q_actor": critic_q_actor,
        "critic_q_data_q1": critic_q_data_q1,
        "critic_q_data_q2": critic_q_data_q2,
        "critic_q_actor_q1": critic_q_actor_q1,
        "critic_q_actor_q2": critic_q_actor_q2,
        "action_mse_matrix": mse_matrix,
        "mc_return": mc_returns[np.asarray(chunk_indices, dtype=np.int64)],
        "all_mc_return": mc_returns,
        "mc_discount_per_chunk": np.asarray(float(worker.cfg.algorithm.gamma)),
        "action_mse": mse,
        "sample_rewards": rewards.detach().float().cpu().numpy(),
        "sample_terminations": terminations.detach().float().cpu().numpy(),
        "critic_mc_mse": np.square(
            critic_q_data - mc_returns[np.asarray(chunk_indices, dtype=np.int64)]
        ),
    }


def save_json(path: str, payload: dict[str, Any]) -> None:
    def convert(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {k: convert(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(v) for v in value]
        return value

    with open(path, "w", encoding="utf-8") as f:
        json.dump(convert(payload), f, indent=2)


__all__ = [
    "SimpleUrdfKinematics",
    "compute_chunk_mc_returns",
    "evaluate_validation_trajectory",
    "actions_to_absolute_joint_targets",
    "load_action_norm_stats",
    "load_key_segment_start_map",
    "load_validation_trajectories",
    "plot_action_mse_heatmaps",
    "plot_boundary_trajectory_3d",
    "plot_action_chunk_segments_3d",
    "plot_action_chunk_triplet_segments_3d",
    "plot_pt_gt_reconstruction_segments_3d",
    "plot_pt_state_trajectory_3d",
    "plot_critic_timeline_with_images",
    "plot_q_values",
    "plot_trajectory_3d",
    "plot_trajectory_3d_with_state",
    "reconstruct_executed_joint_targets",
    "save_json",
    "tcp_trajectories_from_joint_targets",
]
