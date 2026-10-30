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

"""Run the validated G1 RLT training monitor and OpenPI inference commands."""

import argparse
import csv
import io
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from rlinf.models.embodiment.openpi.checkpoint import resolve_full_weights
from rlinf.utils.logging import get_logger
from toolkits import g1_rlt_features as features
from toolkits.g1_rlt_actor_evaluate import load_actor

REPO = Path(__file__).resolve().parents[1]
logger = get_logger()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    tmp.replace(path)


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M")


def timeline(start, description):
    path = REPO / "TIME.csv"
    lines = (
        path.read_bytes().decode("utf-8-sig").splitlines(keepends=True)
        if path.exists()
        else ["start,end,summary\n"]
    )
    stream = io.StringIO(newline="")
    csv.writer(stream, lineterminator="\n").writerow([start, now(), description])
    for i, line in enumerate(lines):
        row = next(csv.reader([line]))
        if len(row) == 3 and row[0] == start and row[2] == description:
            lines[i] = stream.getvalue()
            break
    else:
        lines.append(stream.getvalue())
    path.write_bytes(("\ufeff" + "".join(lines)).encode())


def config(path):
    with initialize_config_dir(
        version_base="1.1", config_dir=str(path.resolve().parent)
    ):
        return compose(config_name=path.stem)


def prepare(args, cfg):
    """Reuse the fixed feature cache, verify its inputs, and preserve split metadata."""
    train = Path(cfg.algorithm.replay_buffer.load_path)
    validation = args.validation_replay or train.parent / "validation_replay"
    manifest = json.loads((train / "feature_manifest.json").read_text())
    source = Path(manifest["source_cache"]) / "source_manifest.json"
    if features.sha256(source) != manifest["source_manifest_sha256"]:
        raise ValueError("Source labels/split changed; rebuild the feature cache first")
    source_metadata = json.loads(source.read_text())
    split = source_metadata["split"]
    outcomes = {e["episode_index"]: e["outcome"] for e in source_metadata["episodes"]}
    if set(split["train"]) & set(split["validation"]):
        raise ValueError("Training and validation episodes overlap")
    weights = resolve_full_weights(args.stage1_checkpoint)
    if weights is None or weights.resolve() != Path(manifest["checkpoint"]).resolve():
        raise ValueError("Stage 1 checkpoint differs from the fixed feature cache")
    if features.sha256(weights) != manifest["checkpoint_sha256"]:
        raise ValueError("Stage 1 weights changed; rebuild the feature cache first")
    signature = features.feature_contract_sha256(manifest)
    if cfg.get("rlt_input_signature", signature) != signature:
        raise ValueError("Actor configuration and feature coordinates differ")
    data = None
    for role, source_dir, name in [
        ("train", train, "feature_replay"),
        ("validation", validation, "validation_replay"),
    ]:
        source_dir = source_dir.resolve()
        metadata = json.loads((source_dir / "feature_manifest.json").read_text())
        if (
            features.feature_contract_sha256(metadata) != signature
            or metadata["role"] != role
        ):
            raise ValueError("Train/validation feature contracts differ")
        trajectories = features.load_raw_replay(source_dir)
        if sorted(features.episode_id(t) for t in trajectories) != sorted(split[role]):
            raise ValueError(f"{role} episode IDs do not match the source split")
        coordinates = features.G1ActionCoordinates(metadata["action_transform"])
        for t in trajectories:
            original = coordinates.normalize(
                t.forward_inputs["stage1_ref_chunk_physical"], t.curr_obs["states"]
            )
            if not torch.equal(t.curr_obs["ref_chunk"], original):
                raise ValueError(
                    "Actor inputs must contain original VLA references, not human answers"
                )
            outcome = outcomes[features.episode_id(t)]
            if outcome not in ("success", "failure"):
                raise ValueError(
                    "BC needs reviewed success/failure labels for every episode"
                )
            successful = outcome == "success"
            if bool((t.rewards > 0).any()) != successful:
                raise ValueError(
                    "Source outcome disagrees with the cached terminal reward"
                )
            t.forward_inputs["bc_success"] = torch.full_like(
                t.rewards, successful, dtype=torch.bool
            )
        destination = args.output / name
        if destination.resolve() == source_dir:
            raise ValueError(
                "Use a new output directory, separate from the source replay"
            )
        destination.mkdir()
        features.save_replay(trajectories, destination, metadata["seed"])
        write_json(
            destination / "feature_manifest.json",
            {
                **metadata,
                "split": split,
                "bc_success_source": str(source),
                "bc_success_definition": "Reviewed insertion outcome is success; human steps use executed actions, autonomous steps use VLA reference. Failed attempts remain in critic replay.",
            },
        )
        if role == "validation":
            obs = {
                k: torch.cat([t.curr_obs[k][:, 0] for t in trajectories])
                for k in ("z_rl", "proprio", "ref_chunk")
            }
            data = (
                obs,
                torch.cat([t.actions.reshape(-1, 10, 16) for t in trajectories]),
                torch.cat(
                    [
                        t.intervene_flags.reshape(-1, 10, 16).any(-1)
                        for t in trajectories
                    ]
                ),
            )
    cfg.algorithm.replay_buffer.load_path = str(args.output / "feature_replay")
    cfg.runner.logger.log_path = str(args.output / "train_run")
    OmegaConf.update(cfg, "rlt_input_signature", signature, force_add=True)
    OmegaConf.save(cfg, args.output / "config.yaml")
    write_json(args.output / "action_transform.json", manifest["action_transform"])
    write_json(args.output / "split.json", split)
    return data


@torch.inference_mode()
def evaluate(cfg, checkpoint, data, scale):
    """Use the same human-action selection criterion as the six matched runs."""
    model = load_actor(cfg.actor.model, checkpoint, "cpu")
    obs, truth, human = data
    prediction = model.sac_forward(obs, deterministic=True)[0].reshape_as(truth)
    repeat = model.sac_forward(obs, deterministic=True)[0].reshape_as(truth)
    if not torch.isfinite(prediction).all() or not torch.equal(prediction, repeat):
        raise ValueError("Actor predictions must be finite and repeatable")
    q = model.sac_q_forward(obs, prediction)
    if q.shape != (len(truth), 2) or not torch.isfinite(q).all():
        raise ValueError("Expected finite twin-Q predictions")
    result = {"checkpoint_sha256": features.sha256(checkpoint), "strict_load": True}
    for name, mask in [
        ("human", human),
        ("policy", ~human),
        ("all", torch.ones_like(human)),
    ]:
        error = (prediction - truth)[mask]
        if not error.numel():
            raise ValueError(f"Validation has no {name} steps")
        physical = error * scale
        result[name] = {
            "steps": int(mask.sum()),
            "normalized_mse": float(error.square().mean()),
            "joint_mae": float(physical[:, :14].abs().mean()),
            "gripper_mae": float(physical[:, 14:].abs().mean()),
        }
    result["selection_score"] = result["human"]["normalized_mse"]
    return result


def train(args):
    """Monitor the existing official four-GPU trainer, saving the validation best."""
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = config(args.config)
    if args.steps is not None:
        cfg.runner.max_steps = cfg.runner.max_epochs = args.steps
    if args.save_every is not None:
        cfg.runner.save_interval = args.save_every
    cfg.runner.resume_dir = str(args.resume_dir.resolve()) if args.resume_dir else None
    cfg.runner.ckpt_path = None
    if str(cfg.cluster.component_placement.actor) != "0-3":
        raise ValueError("This G1 training command expects actor placement 0-3")
    data = prepare(args, cfg)
    scale = torch.tensor(
        json.loads((args.output / "action_transform.json").read_text())["scale"]
    )
    limit = int(cfg.runner.max_steps)
    run = args.output / "train_run"
    run.mkdir()
    ray_tmp = Path(f"/tmp/g1_rlt_{os.getpid()}")
    env = os.environ.copy()
    env["RAY_TMPDIR"] = str(ray_tmp)
    command = [
        sys.executable,
        "-B",
        str(REPO / "examples/embodiment/train_offline_rl.py"),
        "--config-path",
        str(args.output),
        "--config-name",
        "config",
    ]
    start, clock = now(), time.monotonic()
    tested, best, exited_at = {}, None, None
    with (args.output / "training.log").open("w") as log:
        child = subprocess.Popen(
            command,
            cwd=REPO,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        write_json(
            args.output / "launch.json",
            {"pid": child.pid, "command": command, "started_at": start},
        )
        try:
            while True:
                for path in sorted(
                    (run / "train/checkpoints").glob("global_step_*"),
                    key=lambda p: int(p.name.split("_")[-1]),
                ):
                    step = int(path.name.split("_")[-1])
                    learners = [
                        path / f"actor/sac_components/learner_rank_{r}.pt"
                        for r in range(4)
                    ]
                    if (
                        step in tested
                        or not all(p.is_file() for p in learners)
                        or time.time() - path.stat().st_mtime < 2
                    ):
                        continue
                    counters = [
                        torch.load(p, weights_only=True, map_location="cpu")
                        for p in learners
                    ]
                    actor_steps = int(counters[0]["actor_update_step"])
                    if not all(
                        c["update_step"] == step
                        and c["actor_update_step"] == actor_steps
                        for c in counters
                    ):
                        raise ValueError("The four learner update counters disagree")
                    ratio = int(cfg.algorithm.critic_actor_ratio)
                    if actor_steps != (step + ratio - 1) // ratio:
                        raise ValueError(
                            "Actor update count differs from the configured critic:actor ratio"
                        )
                    weight = path / "actor/model_state_dict/full_weights.pt"
                    score = evaluate(cfg, weight, data, scale)
                    score.update(
                        step=step,
                        actor_updates=actor_steps,
                        source_checkpoint=str(path),
                    )
                    tested[step] = score
                    write_json(args.output / f"evaluation_{step:05d}.json", score)
                    if (
                        best is None
                        or score["selection_score"] < tested[best]["selection_score"]
                    ):
                        best = step
                        shutil.copy2(weight, args.output / "best_actor_critic.pt")
                        write_json(
                            args.output / "selection.json",
                            {
                                **score,
                                "checkpoint": str(args.output / "best_actor_critic.pt"),
                            },
                        )
                    logger.info(
                        "critic=%d actor=%d human_MSE=%.8g best=%d",
                        step,
                        actor_steps,
                        score["selection_score"],
                        best,
                    )
                    for previous in (run / "train/checkpoints").glob("global_step_*"):
                        if (
                            int(previous.name.split("_")[-1]) in tested
                            and previous != path
                        ):
                            shutil.rmtree(previous)
                last = max(tested, default=0)
                elapsed = time.monotonic() - clock
                write_json(
                    args.output / "progress.json",
                    {
                        "critic": last,
                        "best_critic": best,
                        "elapsed_seconds": elapsed,
                        "remaining_seconds_estimate": elapsed / last * (limit - last)
                        if last
                        else None,
                    },
                )
                timeline(start, "四卡训练人工优先actor并按验证误差选择权重")
                if child.poll() is not None:
                    if child.returncode:
                        raise RuntimeError(
                            f"Trainer exited {child.returncode}; inspect {args.output / 'training.log'}"
                        )
                    if last == limit:
                        break
                    exited_at = exited_at or time.monotonic()
                    if time.monotonic() - exited_at > 45:
                        raise RuntimeError(
                            "Trainer exited without a complete final checkpoint"
                        )
                time.sleep(3)
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
            child.wait(timeout=30)
    final = run / f"train/checkpoints/global_step_{limit}"
    (args.output / "final_actor_critic.pt").symlink_to(
        final / "actor/model_state_dict/full_weights.pt"
    )
    selected = evaluate(cfg, args.output / "best_actor_critic.pt", data, scale)
    if (
        selected["checkpoint_sha256"] != tested[best]["checkpoint_sha256"]
        or selected["selection_score"] != tested[best]["selection_score"]
    ):
        raise ValueError("Best checkpoint changed after reloading")
    result = {
        "status": "completed",
        "critic_updates": limit,
        "actor_updates": tested[limit]["actor_updates"],
        "best_critic": best,
        "best_actor": tested[best]["actor_updates"],
        "best_human_mse": selected["selection_score"],
        "elapsed_seconds": time.monotonic() - clock,
    }
    write_json(args.output / "result.json", result)
    write_json(args.output / "progress.json", result)
    if ray_tmp.exists():
        shutil.rmtree(ray_tmp)
    logger.info("Best actor/critic: %s", args.output / "best_actor_critic.pt")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["stage1", "train", "serve"])
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "examples/embodiment/config/g1_rlt_stage2_offline_ac_mlp.yaml",
    )
    parser.add_argument(
        "--stage1-config",
        type=Path,
        default=REPO / "examples/sft/config/g1_rlt_stage1_sft_openpi_pi05.yaml",
    )
    parser.add_argument(
        "--stage1-checkpoint",
        type=Path,
        default=Path(
            "/16T-2/sudu/code/.result2/RLT/RLinf/zd_20261026/stage1/train/checkpoints/global_step_5000"
        ),
    )
    parser.add_argument("--validation-replay", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--port", type=int, default=8014)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--save-every", type=int)
    parser.add_argument("--resume-dir", type=Path)
    args = parser.parse_args()
    if (args.steps is not None and args.steps <= 0) or (
        args.save_every is not None and args.save_every <= 0
    ):
        parser.error("--steps and --save-every must be positive")
    torch.set_num_threads(2)
    os.environ.setdefault("PYTHONPATH", str(REPO))
    os.environ.setdefault("EMBODIED_PATH", str(REPO / "examples/embodiment"))
    for key, value in {
        "RAY_ADDRESS": "local",
        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "JAX_PLATFORMS": "cpu",
        "NUMPY_MADVISE_HUGEPAGE": "0",
        "PYTHONUNBUFFERED": "1",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    }.items():
        os.environ.setdefault(key, value)
    if args.phase == "stage1":
        if resolve_full_weights(args.stage1_checkpoint) is not None:
            logger.info("Reuse fixed Stage 1 checkpoint: %s", args.stage1_checkpoint)
            return
        if args.stage1_checkpoint.name != "global_step_5000":
            parser.error(
                "Stage 1 output must end with train/checkpoints/global_step_5000"
            )
        subprocess.run(
            [
                sys.executable,
                "-B",
                str(REPO / "examples/sft/train_vla_sft.py"),
                "--config-path",
                str(args.stage1_config.resolve().parent),
                "--config-name",
                args.stage1_config.stem,
                f"runner.logger.log_path={args.stage1_checkpoint.resolve().parents[2]}",
                "runner.max_steps=5000",
                "cluster.component_placement.actor=0-3",
                "actor.global_batch_size=4",
            ],
            cwd=REPO,
            check=True,
        )
        return
    if args.output is None:
        parser.error("--output is required for train/serve")
    args.output = args.output.resolve()
    if args.phase == "train":
        train(args)
        return
    if args.checkpoint is None:
        parser.error("--checkpoint is required for serve")
    if not (args.output / "config.yaml").exists():
        args.output.mkdir(parents=True, exist_ok=True)
        prepare(args, config(args.config))
    command = [
        sys.executable,
        "-B",
        str(REPO / "toolkits/g1_rlt_actor_evaluate.py"),
        "--config",
        str(args.output / "config.yaml"),
        "--checkpoint",
        str(args.checkpoint.resolve()),
        "--replay",
        str(args.output / "validation_replay"),
        "--serve",
        "--device",
        "cuda:0",
        "--port",
        str(args.port),
    ]
    os.execve(sys.executable, command, os.environ.copy())


if __name__ == "__main__":
    main()
