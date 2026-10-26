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

"""Prepare UR7 data and supervise the official RECAP value SFT workflow."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
os.environ.setdefault("REPO_PATH", str(REPO))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2,3")
os.environ.setdefault("HF_HOME", "/16T-2/sudu/code/.data/recap_models/.cache")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault(
    "HF_DATASETS_CACHE", "/16T-2/sudu/code/.data/recap_models/.cache/datasets"
)
LOG = logging.getLogger("ur7_recap")


def write_json(path: Path, value: object) -> None:
    """Atomically publish a JSON artifact."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_models(cfg, output: Path) -> None:
    """Fetch the pinned vision model and record the actual local asset identity."""
    vision = Path(cfg.actor.model.siglip_path)
    if not (vision / "model.safetensors").is_file():
        from huggingface_hub import snapshot_download

        snapshot_download(
            "google/siglip2-so400m-patch14-224",
            revision=cfg.task.siglip_revision,
            local_dir=str(vision),
            endpoint="https://hf-mirror.com",
            allow_patterns=[
                "config.json",
                "model.safetensors",
                "preprocessor_config.json",
            ],
        )
    assets = {}
    for name, root in (
        ("siglip2", vision),
        ("gemma3", Path(cfg.actor.model.gemma3_path)),
    ):
        assets[name] = {"path": str(root), "files": {}}
        for file in ("config.json", "model.safetensors"):
            assets[name]["files"][file] = sha256(root / file)
    assets["siglip2"]["revision"] = cfg.task.siglip_revision
    assets["rlinf_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()
    write_json(output / "assets.json", assets)


def prepare(cfg, output: Path) -> None:
    """Create disjoint LeRobot views; only videos reference the source dataset."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from process.compute_returns import process_dataset

    source = Path(cfg.task.dataset).resolve()
    info = json.loads((source / "meta/info.json").read_text())
    episodes = {
        row["episode_index"]: row for row in read_jsonl(source / "meta/episodes.jsonl")
    }
    stats = {
        row["episode_index"]: row
        for row in read_jsonl(source / "meta/episodes_stats.jsonl")
    }
    failures = set(cfg.task.failure_episodes)
    test = int(cfg.task.test_episode)
    if test in failures or not failures.issubset(episodes) or test not in episodes:
        raise ValueError(
            "Expected a successful held-out test and existing failure episode IDs"
        )
    candidates = sorted(set(episodes) - failures - {test})
    val = set(
        np.random.default_rng(cfg.data.seed)
        .choice(candidates, cfg.task.validation_episodes, replace=False)
        .tolist()
    )
    splits = {
        "train_success": sorted(set(candidates) - val),
        "train_failure": sorted(failures),
        "validation": sorted(val),
        "test": [test],
    }
    if sum(map(len, splits.values())) != len(episodes):
        raise ValueError("Episode split must cover the source exactly once")
    manifest = {
        "source": str(source),
        "splits": splits,
        "source_sha256": {},
        "frames": {},
    }
    # Hash metadata and numeric samples; videos are referenced without modification.
    for path in sorted((source / "meta").glob("*")) + sorted(
        (source / "data").rglob("*.parquet")
    ):
        if path.is_file():
            manifest["source_sha256"][str(path.relative_to(source))] = sha256(path)
    video_keys = [k for k, v in info["features"].items() if v["dtype"] == "video"]
    all_train_returns = []
    for name, original_ids in splits.items():
        root = output / "data" / name
        (root / "meta").mkdir(parents=True, exist_ok=False)
        subset_info = copy.deepcopy(info)
        subset_episodes, subset_stats = [], []
        offset = 0
        for new_id, original_id in enumerate(original_ids):
            old_args = {
                "episode_chunk": original_id // info["chunks_size"],
                "episode_index": original_id,
            }
            new_args = {
                "episode_chunk": new_id // info["chunks_size"],
                "episode_index": new_id,
            }
            table = pq.read_table(source / info["data_path"].format(**old_args))
            n = len(table)
            if n != episodes[original_id]["length"]:
                raise ValueError(
                    f"Episode {original_id}: metadata length does not match parquet"
                )
            for key, values in (
                ("episode_index", np.full(n, new_id)),
                ("index", np.arange(offset, offset + n)),
            ):
                table = table.set_column(
                    table.schema.get_field_index(key),
                    key,
                    pa.array(values, type=pa.int64()),
                )
            table = table.append_column(
                "is_success", pa.array([original_id not in failures] * n)
            )
            target = root / info["data_path"].format(**new_args)
            target.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, target)
            for key in video_keys:
                target = root / info["video_path"].format(**new_args, video_key=key)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to(
                    source / info["video_path"].format(**old_args, video_key=key)
                )
            subset_episodes.append({**episodes[original_id], "episode_index": new_id})
            item = copy.deepcopy(stats[original_id])
            item["episode_index"] = new_id
            for key, values in (
                ("episode_index", np.full(n, new_id)),
                ("index", np.arange(offset, offset + n)),
            ):
                item["stats"][key] = {
                    "min": [int(values.min())],
                    "max": [int(values.max())],
                    "mean": [float(values.mean())],
                    "std": [float(values.std())],
                    "count": [n],
                }
            subset_stats.append(item)
            offset += n
        subset_info.update(
            total_episodes=len(original_ids),
            total_frames=offset,
            total_videos=len(original_ids) * len(video_keys),
            total_chunks=math.ceil(len(original_ids) / info["chunks_size"]),
            splits={"train": f"0:{len(original_ids)}"},
        )
        subset_info["features"]["is_success"] = {
            "dtype": "bool",
            "shape": [1],
            "names": None,
        }
        write_json(root / "meta/info.json", subset_info)
        for filename, rows in (
            ("episodes.jsonl", subset_episodes),
            ("episodes_stats.jsonl", subset_stats),
        ):
            (root / "meta" / filename).write_text(
                "".join(json.dumps(row) + "\n" for row in rows)
            )
        shutil.copyfile(source / "meta/tasks.jsonl", root / "meta/tasks.jsonl")
        process_dataset(
            root,
            None,
            "rollout",
            float(cfg.data.gamma),
            float(cfg.task.failure_reward),
            num_workers=8,
            tag=cfg.data.tag,
        )
        returns = pq.read_table(root / "meta" / f"returns_{cfg.data.tag}.parquet")
        manifest["frames"][name] = len(returns)
        if name.startswith("train"):
            all_train_returns.append(returns["return"].to_numpy())
    train = np.concatenate(all_train_returns)
    cfg.data.return_min, cfg.data.return_max = float(train.min()), float(train.max())
    manifest.update(
        return_min=cfg.data.return_min,
        return_max=cfg.data.return_max,
        train_mean=float(train.mean() / abs(train.min())),
        failure_reward=float(cfg.task.failure_reward),
        seed=int(cfg.data.seed),
    )
    cfg.data.train_data_paths = [
        {
            "dataset_path": str(output / "data" / name),
            "type": "rollout",
            "weight": weight,
        }
        for name, weight in (("train_success", 0.95), ("train_failure", 0.05))
    ]
    cfg.data.eval_data_paths = [
        {"dataset_path": str(output / "data/validation"), "name": "validation"}
    ]
    cfg.runner.logger.log_path = str(output / "training")
    write_json(output / "split.json", manifest)
    OmegaConf.save(cfg, output / "config.yaml", resolve=True)
    LOG.info(
        "Prepared %s; return range [%s, %s]",
        manifest["frames"],
        cfg.data.return_min,
        cfg.data.return_max,
    )


def stop_owned_processes(process: subprocess.Popen, descendants: dict) -> None:
    """Terminate only processes descended from this training launch."""
    import psutil

    try:
        for child in psutil.Process(process.pid).children(recursive=True):
            descendants[child.pid] = child
    except psutil.NoSuchProcess:
        pass
    for child in reversed(list(descendants.values())):
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass
    if process.poll() is None:
        process.terminate()
    _, alive = psutil.wait_procs(list(descendants.values()), timeout=10)
    for child in alive:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def train(
    cfg, output: Path, phase: str, resume: Path | None = None, monitor: bool = False
) -> dict:
    """Run the official entry point and select checkpoints from its event logs."""
    import psutil

    sys.modules.setdefault("tensorflow", None)
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    phase_dir = output / phase
    phase_dir.mkdir(exist_ok=True)
    runtime = copy.deepcopy(cfg)
    runtime.runner.logger.log_path = str(phase_dir)
    if resume is not None:
        runtime.runner.resume_dir = str(resume)
    OmegaConf.save(runtime, phase_dir / "runtime.yaml", resolve=True)
    env = os.environ.copy()
    env.update(
        CUDA_VISIBLE_DEVICES="2,3",
        PYTHONPATH=str(REPO),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        RAY_TMPDIR=f"/16T-2/sudu/.ray/{os.getpid()}_{phase[:1]}",
        PYTHONUNBUFFERED="1",
        PYTHONHASHSEED=str(cfg.actor.seed),
    )
    env.pop("RAY_ADDRESS", None)
    command = [
        sys.executable,
        str(Path(__file__).with_name("train_value.py")),
        "--config-path",
        str(phase_dir),
        "--config-name",
        "runtime",
    ]
    started, last_report = time.time(), 0.0
    seen, descendants = set(), {}
    best, patience_best, stale = math.inf, math.inf, 0
    result = {"best_step": None, "stopped_early": False}
    events = EventAccumulator(
        str(phase_dir / "tensorboard"), size_guidance={"scalars": 0}
    )
    with (output / "train.log").open("a", buffering=1) as logfile:
        logfile.write(f"\n=== {phase}: {' '.join(command)} ===\n")
        process = subprocess.Popen(
            command,
            cwd=REPO,
            env=env,
            stdout=logfile,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            while True:
                finished = process.poll() is not None
                try:
                    for child in psutil.Process(process.pid).children(recursive=True):
                        descendants[child.pid] = child
                except psutil.NoSuchProcess:
                    pass
                if (phase_dir / "tensorboard").exists():
                    events.Reload()
                    tags = events.Tags()["scalars"]
                    if "eval/validation/loss" in tags:
                        for event in events.Scalars("eval/validation/loss"):
                            step = (
                                event.step + 1
                            )  # SFTRunner logs the zero-based loop index.
                            if step in seen:
                                continue
                            weight = (
                                phase_dir
                                / cfg.runner.logger.experiment_name
                                / f"checkpoints/global_step_{step}/actor/model_state_dict/full_weights.pt"
                            )
                            if not weight.is_file() or not math.isfinite(event.value):
                                raise RuntimeError(
                                    f"Invalid checkpoint or validation loss at step {step}"
                                )
                            seen.add(step)
                            if event.value < best:
                                best = event.value
                                result.update(
                                    best_step=step,
                                    best_validation_ce=best,
                                    checkpoint=str(weight),
                                )
                                if monitor:
                                    shutil.copyfile(weight, output / "best.pt.tmp")
                                    (output / "best.pt.tmp").replace(output / "best.pt")
                                    result["checkpoint"] = str(output / "best.pt")
                            if event.value < patience_best - cfg.task.min_delta:
                                patience_best, stale = event.value, 0
                            else:
                                stale += 1
                            result.update(
                                last_validated_step=step,
                                elapsed_seconds=time.time() - started,
                            )
                            write_json(phase_dir / "progress.json", result)
                            LOG.info(
                                "%s step %d/%d: validation CE %.6f, best step %s",
                                phase,
                                step,
                                cfg.runner.max_steps,
                                event.value,
                                result["best_step"],
                            )
                            if (
                                monitor
                                and step >= cfg.task.min_steps
                                and stale >= cfg.task.patience
                            ):
                                result["stopped_early"] = True
                            if monitor:
                                for old in weight.parents[2].parent.glob(
                                    "global_step_*"
                                ):
                                    if (
                                        int(old.name.removeprefix("global_step_"))
                                        < step
                                    ):
                                        shutil.rmtree(old)
                    if time.time() - last_report > 60 and "train/loss" in tags:
                        point = events.Scalars("train/loss")[-1]
                        elapsed = time.time() - started
                        LOG.info(
                            "%s step %d: loss %.6f, elapsed %.1f min, estimated remaining %.1f min",
                            phase,
                            point.step + 1,
                            point.value,
                            elapsed / 60,
                            elapsed
                            / max(point.step + 1, 1)
                            * (cfg.runner.max_steps - point.step - 1)
                            / 60,
                        )
                        last_report = time.time()
                if finished or result["stopped_early"]:
                    break
                time.sleep(10)
            if process.poll() not in (None, 0) and not result["stopped_early"]:
                raise RuntimeError(
                    f"Official {phase} training exited {process.returncode}; see {output / 'train.log'}"
                )
            if not seen:
                raise RuntimeError(f"{phase} produced no validated checkpoint")
        finally:
            stop_owned_processes(process, descendants)
    write_json(phase_dir / "progress.json", result)
    return result


def make_dataset(cfg, root: Path):
    from rlinf.data.datasets.recap.value_dataset import ValueDataset

    return ValueDataset(
        str(root),
        robot_type="ur7",
        model_type="pi05",
        action_dim=32,
        action_horizon=1,
        return_min=cfg.data.return_min,
        return_max=cfg.data.return_max,
        tag=cfg.data.tag,
    )


def evaluate(cfg, output: Path) -> None:
    """Strictly reload the selected checkpoint and evaluate each frame once."""
    import torch
    from scipy.stats import spearmanr
    from torch.utils.data import DataLoader

    from rlinf.models.embodiment.value_model.recap import get_model
    from rlinf.models.embodiment.value_model.recap.data_collator import (
        ValueDataCollator,
    )
    from rlinf.models.embodiment.value_model.recap.processing import ValueProcessor

    rebuilt = not (output / "data").exists()
    if rebuilt:
        manifest = json.loads((output / "split.json").read_text())
        for name, digest in manifest["source_sha256"].items():
            if sha256(Path(manifest["source"]) / name) != digest:
                raise RuntimeError(f"Source dataset changed since training: {name}")
        prepare(cfg, output)
    torch.set_num_threads(4)
    model = get_model(cfg.actor.model)
    state = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True, assign=True)
    del state
    # Match FSDP's parameter, buffer, and input casts; autocast alone differs.
    model.eval().to("cuda", dtype=torch.bfloat16)
    processor = ValueProcessor(
        tokenizer_name_or_path=cfg.actor.model.tokenizer_path,
        max_token_len=cfg.actor.model.max_token_len,
        do_augment=False,
    )
    collator = ValueDataCollator(
        processor, max_length=cfg.actor.model.max_token_len, train=False
    )
    manifest = json.loads((output / "split.json").read_text())
    report = {
        "checkpoint_sha256": sha256(output / "best.pt"),
        "train_mean_baseline": manifest["train_mean"],
        "normalization": {
            "return_min": cfg.data.return_min,
            "return_max": cfg.data.return_max,
        },
        "sets": {},
    }
    curves = {}

    def device_tree(value):
        if isinstance(value, dict):
            return {k: device_tree(v) for k, v in value.items()}
        if isinstance(value, torch.Tensor):
            dtype = torch.bfloat16 if value.is_floating_point() else value.dtype
            return value.to("cuda", dtype=dtype)
        return value

    for name in ("validation", "train_failure", "test"):
        dataset = make_dataset(cfg, output / "data" / name)
        loader = DataLoader(
            dataset, batch_size=2, shuffle=False, num_workers=2, collate_fn=collator
        )
        predictions, targets = [], []
        with torch.inference_mode():
            for batch in loader:
                pred = (
                    model(observation=device_tree(batch["observation"]))
                    .predicted_values.float()
                    .cpu()
                    .numpy()
                    .reshape(-1)
                )
                predictions.extend(pred.tolist())
                targets.extend(batch["target_values"].numpy().reshape(-1).tolist())
        p, y = np.asarray(predictions), np.asarray(targets)
        if (
            len(p) != len(dataset)
            or not np.isfinite(p).all()
            or np.any((p < -1.00001) | (p > 0.00001))
        ):
            raise RuntimeError(f"Invalid predictions for {name}")
        baseline = manifest["train_mean"]
        corr = float(spearmanr(p, y).statistic)
        report["sets"][name] = {
            "frames": len(p),
            "mae": float(np.abs(p - y).mean()),
            "mse": float(np.square(p - y).mean()),
            "spearman": corr if math.isfinite(corr) else None,
            "baseline_mae": float(np.abs(baseline - y).mean()),
            "baseline_mse": float(np.square(baseline - y).mean()),
        }
        curves[name] = {"prediction": predictions, "target": targets}
        if name == "train_failure":
            offset = 0
            lengths = read_jsonl(output / "data/train_failure/meta/episodes.jsonl")
            for original_id, item in zip(
                manifest["splits"][name], lengths, strict=True
            ):
                end = offset + item["length"]
                report["sets"][f"training_episode_{original_id}"] = {
                    "frames": end - offset,
                    "mae": float(np.abs(p[offset:end] - y[offset:end]).mean()),
                    "predicted_endpoint": float(p[end - 1]),
                    "target_endpoint": float(y[end - 1]),
                }
                offset = end
        LOG.info("Independent %s: %s", name, report["sets"][name])
    report["passes_baseline"] = (
        report["sets"]["validation"]["mae"]
        < report["sets"]["validation"]["baseline_mae"]
    )
    report["failure_scope"] = (
        "Training-fit diagnostic only; both failures were used for training."
    )
    write_json(output / "predictions.json", curves)
    report["video"] = video(cfg, output, curves["test"], manifest["train_mean"])
    write_json(output / "evaluation.json", report)
    training = json.loads((output / "training_result.json").read_text())
    validation, test = report["sets"]["validation"], report["sets"]["test"]
    verdict = "PASS" if report["passes_baseline"] else "TARGET NOT MET"
    (output / "report.md").write_text(
        f"# RLinf RECAP UR7: {verdict}\n\n"
        f"Best checkpoint: step {training['best_step']}; "
        f"last validated step: {training['last_validated_step']}.\n\n"
        f"Validation: {validation['frames']} frames, MAE {validation['mae']:.6f}, "
        f"MSE {validation['mse']:.6f}; mean baseline MAE {validation['baseline_mae']:.6f}.\n\n"
        f"Held-out episode {cfg.task.test_episode}: {test['frames']} frames, "
        f"MAE {test['mae']:.6f}, MSE {test['mse']:.6f}.\n\n"
        "Weights: `best.pt`. Full checkpoint keys were strictly validated in a fresh process.\n\n"
        f"Video: `{report['video']['filename']}`. "
        "Predictions are unsmoothed normalized returns, not success probabilities.\n\n"
        "Both failure episodes were used for training; failure results measure training fit only.\n\n"
        "Original pi0.5 weights are not used by this independent official RECAP value model.\n\n"
        "Re-evaluate from the RLinf repository:\n\n```bash\n"
        ".venv/bin/python examples/offline_rl/advantage_labeling/recap/ur7_value.py "
        f"evaluate --output-dir {output}\n```\n"
    )
    if rebuilt:
        shutil.rmtree(output / "data")


def video(cfg, output: Path, curve: dict, baseline: float) -> dict:
    """Render synchronized source cameras and unmodified predicted returns."""
    import av
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter

    source = Path(cfg.task.dataset)
    info = json.loads((source / "meta/info.json").read_text())
    episode = int(cfg.task.test_episode)
    cameras = []
    for key in ("observation.images.base", "observation.images.wrist"):
        path = source / info["video_path"].format(
            episode_chunk=episode // info["chunks_size"],
            episode_index=episode,
            video_key=key,
        )
        with av.open(str(path)) as container:
            cameras.append(
                [
                    frame.to_ndarray(format="rgb24")
                    for frame in container.decode(video=0)
                ]
            )
    n, fps = len(curve["prediction"]), info["fps"]
    if any(len(frames) != n for frames in cameras):
        raise ValueError("Source camera frame counts must match predicted frame count")
    fig = plt.figure(figsize=(12.8, 10), dpi=100, facecolor="white")
    axes = [
        fig.add_axes([0.02, 0.46, 0.47, 0.45]),
        fig.add_axes([0.51, 0.46, 0.47, 0.45]),
    ]
    images = []
    for ax, frames, label in zip(axes, cameras, ("Base", "Wrist")):
        images.append(ax.imshow(frames[0]))
        ax.set_title(label, fontsize=15)
        ax.set_axis_off()
    ax = fig.add_axes([0.09, 0.09, 0.86, 0.31])
    times = np.arange(n) / fps
    ax.plot(
        times,
        curve["prediction"],
        color="#2076b4",
        label="Predicted value",
        linewidth=2,
    )
    ax.plot(
        times, curve["target"], color="#d34c34", label="Normalized return", linewidth=2
    )
    ax.axhline(baseline, color="#328455", linestyle="--", label="Train-mean baseline")
    cursor = ax.axvline(0, color="#333333", linewidth=1)
    ax.set(xlabel="Time (s)", ylabel="Normalized return", xlim=(0, max(times[-1], 0.1)))
    ax.grid(alpha=0.2)
    ax.legend(loc="lower right", fontsize=11)
    best_step = json.loads((output / "training_result.json").read_text())["best_step"]
    fig.suptitle(
        f"RLinf RECAP | UR7 | Step {best_step} | Episode {episode:06d}",
        fontsize=20,
        y=0.97,
    )
    status = fig.text(0.09, 0.43, "", fontsize=13)
    destination = output / f"episode_{episode:06d}.mp4"
    writer = FFMpegWriter(
        fps=fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-crf", "18"]
    )
    with writer.saving(fig, str(destination), dpi=100):
        for i in range(n):
            for artist, frames in zip(images, cameras):
                artist.set_data(frames[i])
            cursor.set_xdata([times[i], times[i]])
            status.set_text(
                f"Frame {i + 1}/{n} | {times[i]:.1f} s | Value {curve['prediction'][i]:.4f} | Return {curve['target'][i]:.4f}"
            )
            writer.grab_frame()
    plt.close(fig)
    with av.open(str(destination)) as container:
        decoded = list(container.decode(video=0))
    if len(decoded) != n or decoded[0].width != 1280 or decoded[0].height != 1000:
        raise RuntimeError("Rendered video dimensions or frame count are incorrect")
    for i in (0, n // 2, n - 1):
        if decoded[i].to_ndarray(format="rgb24").std() < 10:
            raise RuntimeError(f"Rendered frame {i} is blank")
    probe = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_frames",
                "-show_entries",
                "stream=codec_name,width,height,r_frame_rate,nb_read_frames,duration,pix_fmt",
                "-of",
                "json",
                str(destination),
            ],
            text=True,
        )
    )["streams"][0]
    numerator, denominator = map(int, probe["r_frame_rate"].split("/"))
    if (
        int(probe["nb_read_frames"]) != n
        or numerator / denominator != fps
        or abs(float(probe["duration"]) - n / fps) > 1 / fps
        or probe["codec_name"] != "h264"
        or probe["pix_fmt"] != "yuv420p"
    ):
        raise RuntimeError(f"Unexpected encoded video properties: {probe}")
    LOG.info("Validated video: %s (%d frames, %s FPS)", destination, n, fps)
    return {"filename": destination.name, **probe}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("all", "prepare", "train", "evaluate"))
    parser.add_argument("--config", default="recap_ur7_value")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    logging.getLogger("tensorboard").setLevel(logging.WARNING)
    if args.command in ("train", "evaluate"):
        if args.output_dir is None:
            parser.error("--output-dir is required for train/evaluate")
        output = args.output_dir.resolve()
        cfg = OmegaConf.load(output / "config.yaml")
    else:
        from hydra import compose, initialize_config_dir

        with initialize_config_dir(
            config_dir=str(REPO / "examples/offline_rl/config"), version_base=None
        ):
            cfg = compose(config_name=args.config)
        OmegaConf.set_struct(cfg, False)
        stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
        output = (args.output_dir or Path(cfg.task.output_root) / stamp).resolve()
        output.mkdir(parents=True, exist_ok=False)
        LOG.info("Run directory: %s", output)
        prepare(cfg, output)
    if args.command == "prepare":
        return
    if args.command in ("all", "train"):
        ensure_models(cfg, output)
        smoke = copy.deepcopy(cfg)
        (
            smoke.runner.max_steps,
            smoke.runner.val_check_interval,
            smoke.runner.save_interval,
        ) = 2, 2, 2
        smoke.data.eval_data_paths[0].max_samples = 16
        try:
            first = train(smoke, output, "smoke")
        except RuntimeError:
            if "out of memory" not in (output / "train.log").read_text().lower():
                raise
            cfg.actor.micro_batch_size = smoke.actor.micro_batch_size = 1
            OmegaConf.save(cfg, output / "config.yaml", resolve=True)
            first = train(smoke, output, "smoke_micro1")
        smoke.runner.max_steps = 3
        resumed = train(
            smoke, output, "resume", resume=Path(first["checkpoint"]).parents[2]
        )
        if resumed["last_validated_step"] != 3 or sha256(
            Path(first["checkpoint"])
        ) == sha256(Path(resumed["checkpoint"])):
            raise RuntimeError("Checkpoint resume did not advance and update weights")
        result = train(cfg, output, "training", monitor=True)
        result["checkpoint"] = str(output / "best.pt")
        write_json(output / "training_result.json", result)
        env = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "2",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
        subprocess.run(
            [sys.executable, __file__, "evaluate", "--output-dir", str(output)],
            env=env,
            check=True,
        )
        manifest = json.loads((output / "split.json").read_text())
        for name, digest in manifest["source_sha256"].items():
            if sha256(Path(manifest["source"]) / name) != digest:
                raise RuntimeError(f"Source dataset changed during training: {name}")
        for name in ("smoke", "smoke_micro1", "resume", "data"):
            if (output / name).exists():
                shutil.rmtree(output / name)
        shutil.rmtree(
            output / "training" / cfg.runner.logger.experiment_name / "checkpoints"
        )
        LOG.info("Complete: %s", output)
    else:
        evaluate(cfg, output)


if __name__ == "__main__":
    main()
