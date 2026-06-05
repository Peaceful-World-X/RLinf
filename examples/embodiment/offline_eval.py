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

"""Offline evaluation using the RLinf framework model loading path.

Usage:
    bash examples/embodiment/run_offline_eval.sh offline_eval_pi05
"""

import copy
import json
import os
import random
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import open_dict

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("RLINF_SKIP_ROS_CLEANUP", "1")
os.environ.setdefault("AV_LOG_LEVEL", "panic")
os.environ.setdefault("LIBDAV1D_LOG_LEVEL", "0")


def _suppress_video_logging() -> None:
    try:
        import av

        av.logging.set_level(av.logging.FATAL)
    except ImportError:
        pass


def get_episode_indices(
    dataset_path: str, num_trajectories: int, seed: int
) -> list[int]:
    episodes_file = Path(dataset_path) / "meta" / "episodes.jsonl"
    all_indices = [json.loads(l)["episode_index"] for l in open(episodes_file)]
    return random.Random(seed).sample(
        all_indices, min(num_trajectories, len(all_indices))
    )


def load_episode_data(
    ds, episode_idx: int, episode_indices: list, dataset_path: str, action_chunk: int
):
    """GT actions from parquet + images only for stride frames."""
    import pyarrow.parquet as pq

    all_actions = []
    for pf in sorted((Path(dataset_path) / "data").glob("chunk-*/episode_*.parquet")):
        tbl = pq.read_table(pf, columns=["episode_index", "action"])
        d = tbl.to_pydict()
        rows = [
            a for ep, a in zip(d["episode_index"], d["action"]) if ep == episode_idx
        ]
        if rows:
            all_actions = rows
            break
    if not all_actions:
        return [], np.array([]), ""
    all_actions = np.array(all_actions)
    T = len(all_actions)

    task_str = ""
    for l in open(Path(dataset_path) / "meta" / "episodes.jsonl"):
        row = json.loads(l)
        if row["episode_index"] == episode_idx:
            tasks = row.get("tasks", [])
            task_str = tasks[0] if tasks else ""
            break

    pos = episode_indices.index(episode_idx)
    start = ds.episode_data_index["from"][pos].item()
    stride_frames = [(t, ds[start + t]) for t in range(0, T, action_chunk)]

    return stride_frames, all_actions, task_str


def img_to_hwc(t) -> np.ndarray:
    arr = t.numpy() if hasattr(t, "numpy") else np.array(t)
    if arr.dtype != np.uint8:
        arr = (arr * 255).clip(0, 255).astype(np.uint8)
    return arr.transpose(1, 2, 0) if arr.shape[0] == 3 else arr


def frame_to_env_obs(frame: dict, task: str) -> dict:
    """Convert a LeRobot frame to env_obs matching realworld_env._wrap_obs() exactly."""
    import torch

    cam_high = img_to_hwc(frame["observation.images.cam_high"])
    cam_lw = img_to_hwc(frame["observation.images.cam_left_wrist"])
    cam_rw = img_to_hwc(frame["observation.images.cam_right_wrist"])

    main_images = torch.from_numpy(cam_high).unsqueeze(0)
    extra_views = torch.from_numpy(np.stack([cam_lw, cam_rw], axis=0)).unsqueeze(0)
    states = torch.from_numpy(frame["observation.state"].numpy()).float().unsqueeze(0)

    return {
        "main_images": main_images,
        "extra_view_images": extra_views,
        "states": states,
        "task_descriptions": [task],
    }


def _align_gt_actions_to_prediction(
    pred_chunk: np.ndarray, gt: np.ndarray
) -> np.ndarray:
    pred_dim = pred_chunk.shape[-1]
    gt_dim = gt.shape[-1]
    if gt_dim < pred_dim:
        raise ValueError(
            f"Ground-truth action dim {gt_dim} is smaller than prediction dim {pred_dim}."
        )
    if gt_dim > pred_dim:
        return gt[..., :pred_dim]
    return gt


def eval_episode(
    model,
    stride_frames: list,
    all_actions: np.ndarray,
    task: str,
    action_chunk: int = 50,
) -> np.ndarray:
    import torch

    T = len(all_actions)
    errors = []
    with torch.no_grad():
        for t, frame in stride_frames:
            env_obs = frame_to_env_obs(frame, task)
            actions, _ = model.predict_action_batch(
                env_obs=env_obs, mode="eval", compute_values=False
            )
            pred_chunk = actions[0].detach().cpu().numpy()
            if pred_chunk.ndim == 1:
                pred_chunk = pred_chunk.reshape(-1, all_actions.shape[-1])
            H = pred_chunk.shape[0]
            gt = all_actions[t : min(t + H, T)]
            if len(gt) < H:
                gt = np.pad(gt, ((0, H - len(gt)), (0, 0)), mode="edge")
            gt = _align_gt_actions_to_prediction(pred_chunk, gt)
            errors.append(np.abs(pred_chunk - gt))

    return np.stack(errors, axis=0)  # [num_chunks, H, action_dim]


def plot_chunk_heatmaps(all_errors, episode_indices, output_dir, sample_frames=5):
    os.makedirs(output_dir, exist_ok=True)

    for ep_errors, ep_idx in zip(all_errors, episode_indices):
        T, H, D = ep_errors.shape
        frame_indices = np.linspace(0, T - 1, min(sample_frames, T), dtype=int)

        fig, axes = plt.subplots(
            1, len(frame_indices), figsize=(4 * len(frame_indices), D * 0.5 + 2)
        )
        if len(frame_indices) == 1:
            axes = [axes]

        vmax = ep_errors[frame_indices].max()
        for ax, fi in zip(axes, frame_indices):
            im = ax.imshow(
                ep_errors[fi].T,
                aspect="auto",
                vmin=0,
                vmax=vmax,
                cmap="hot_r",
                origin="upper",
            )
            ax.set_title(f"frame {fi}")
            ax.set_xlabel("chunk step")
            ax.set_ylabel("action dim")
            ax.set_yticks(range(D))
        fig.colorbar(im, ax=axes[-1], label="|pred - gt|")
        fig.suptitle(f"Episode {ep_idx} — action chunk abs error")
        plt.tight_layout()
        path = os.path.join(output_dir, f"episode_{ep_idx:06d}_chunk_error.png")
        plt.savefig(path, dpi=100)
        plt.close(fig)
        print(f"  Saved: {path}")

    all_concat = np.concatenate(all_errors, axis=0)
    mean_err = all_concat.mean(axis=0)
    H, D = mean_err.shape
    fig, ax = plt.subplots(figsize=(max(6, H // 4), D * 0.5 + 2))
    im = ax.imshow(mean_err.T, aspect="auto", vmin=0, cmap="hot_r", origin="upper")
    ax.set_xlabel("chunk step")
    ax.set_ylabel("action dim")
    ax.set_yticks(range(D))
    fig.colorbar(im, ax=ax, label="|pred - gt|")
    fig.suptitle("Mean action chunk abs error (all episodes & frames)")
    plt.tight_layout()
    path = os.path.join(output_dir, "summary_chunk_error.png")
    plt.savefig(path, dpi=100)
    plt.close(fig)
    print(f"  Saved summary: {path}")


@hydra.main(version_base="1.1", config_path="config", config_name="offline_eval_pi05")
def main(cfg) -> None:
    from rlinf.models import get_model

    _suppress_video_logging()
    dataset_path = cfg.dataset_path
    output_dir = cfg.get("output_dir", "logs/offline_eval")

    rollout_model_config = copy.deepcopy(cfg.actor.model)
    with open_dict(rollout_model_config):
        rollout_model_config.precision = cfg.rollout.model.precision
        rollout_model_config.model_path = cfg.rollout.model.model_path
    print(
        f"Loading model: {rollout_model_config.openpi.config_name} from {rollout_model_config.model_path}"
    )
    model = get_model(rollout_model_config)
    model.eval()

    import torch

    if torch.cuda.is_available():
        model = model.cuda()

    episode_indices = get_episode_indices(
        dataset_path, cfg.get("num_trajectories", 5), cfg.get("seed", 42)
    )
    print(f"Evaluating episodes: {episode_indices}")

    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    print("Loading dataset...")
    ds = LeRobotDataset(
        dataset_path,
        episodes=episode_indices,
        video_backend=cfg.get("video_backend", "pyav"),
    )

    all_errors = []
    action_chunk = cfg.actor.model.num_action_chunks
    for ep_idx in episode_indices:
        print(f"  Episode {ep_idx}...")
        stride_frames, all_actions, task = load_episode_data(
            ds, ep_idx, episode_indices, dataset_path, action_chunk
        )
        if not stride_frames:
            continue
        errors = eval_episode(
            model, stride_frames, all_actions, task, action_chunk=action_chunk
        )
        all_errors.append(errors)
        print(
            f"    chunks={len(stride_frames)}, action_horizon={errors.shape[1]}, MSE={np.mean(errors**2):.6f}"
        )

    all_flat = np.concatenate([e.reshape(-1) for e in all_errors])
    print(f"\nOverall action MSE: {np.mean(all_flat**2):.6f}")

    all_concat = np.concatenate(all_errors, axis=0)
    per_dim_mse = np.mean(all_concat**2, axis=(0, 1))
    print("Per-dimension MSE:")
    for i, v in enumerate(per_dim_mse):
        print(f"  dim {i:2d}: {v:.6f}")

    plot_chunk_heatmaps(
        all_errors,
        episode_indices,
        output_dir,
        sample_frames=cfg.get("sample_frames", 5),
    )
    print("Done.")


if __name__ == "__main__":
    main()
