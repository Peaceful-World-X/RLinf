# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run official RLinf SFT in checkpoint blocks; evaluate between GPU processes."""

import csv
import datetime as dt
import fcntl
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from rlinf.utils.logging import get_logger

LOGGER = get_logger()

ROOT = Path("/mnt/20T-1/sudu/.data/RLinf/g1_rlt/20261027_1042")
WORK = Path(__file__).resolve().parents[1]
PYTHON = str(WORK / "env/bin/python")
CONFIG = "g1_rlt_stage1_sft_openpi_pi05"
BASE = "/mnt/20T-1/sudu/.result/zh_g1_zip_tie_1677_sft_b256_40k_v1_2w_rlinf_torch"
TIME = WORK / "TIME.csv"
STATE = ROOT / "stage1_progress.json"
CHECKPOINTS = ROOT / "stage1/checkpoints"
MILESTONES = (5000, 10000, 15000, 20000)
ENV = os.environ.copy()
ENV.update(
    CUDA_VISIBLE_DEVICES="1",
    PYTHONPATH=str(WORK),
    EMBODIED_PATH=str(WORK / "examples/embodiment"),
    JAX_PLATFORMS="cpu",
    NUMPY_MADVISE_HUGEPAGE="0",
    OMP_NUM_THREADS="4",
    MKL_NUM_THREADS="4",
    OPENBLAS_NUM_THREADS="4",
    MALLOC_ARENA_MAX="1",
    MALLOC_TRIM_THRESHOLD_="131072",
    PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
    OPENPI_DATA_HOME="/mnt/20T-1/sudu/.data/.cache/openpi",
    HF_HOME="/mnt/20T-1/sudu/.data/.cache/huggingface",
)
# This job is single-node: keep communication local and connect peers at init.
ENV.update(
    NCCL_SOCKET_IFNAME="lo",
    GLOO_SOCKET_IFNAME="lo",
    NCCL_RUNTIME_CONNECT="0",
    NCCL_DEBUG="INFO",
    NCCL_DEBUG_SUBSYS="INIT,NET",
)


def now():
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M")


def save_state(state):
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(STATE)


def record(start, message):
    with TIME.open("a", encoding="utf-8", newline="") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        csv.writer(stream).writerow([start, now(), message])
        stream.flush()
    LOGGER.info("%s %s", now(), message)


def run(args, log_name, state, phase):
    env = ENV.copy()
    env["RAY_TMPDIR"] = "/mnt/20T-1/sudu/.data/.cache/r1"
    log = ROOT / log_name
    with log.open("a") as stream:
        process = subprocess.Popen(
            [PYTHON, "-u", *args],
            cwd=WORK,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
        state.update(
            status="running",
            phase=phase,
            child_pid=process.pid,
            current_log=str(log),
            updated=now(),
        )
        save_state(state)
        exit_code = process.wait()
    if exit_code:
        raise RuntimeError(f"{phase} exited {exit_code}; inspect {log}")
    state.pop("child_pid", None)
    save_state(state)


def validate(checkpoint, step, state):
    output = ROOT / f"stage1_validation_{step:05d}.json"
    if not output.exists():
        run(
            [
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc_per_node=1",
                "toolkits/evaluate_rlt.py",
                "--config",
                CONFIG,
                "--checkpoint",
                str(checkpoint),
                "--output",
                str(output),
                "--samples",
                "512",
            ],
            f"stage1_validation_{step:05d}.log",
            state,
            f"validation_{step}",
        )
        run(
            [
                "toolkits/evaluate_rlt.py",
                "--merge-only",
                "--checkpoint",
                str(checkpoint),
                "--output",
                str(output),
            ],
            f"stage1_validation_{step:05d}.log",
            state,
            f"merge_validation_{step}",
        )
    result = json.loads(output.read_text())
    if result["samples"] != 512 or not result["finite"]:
        raise RuntimeError("Incomplete or nonfinite validation")
    return result


def prune(state):
    keep = {*MILESTONES, 8500, state["completed_step"], state["best_step"]}
    for path in CHECKPOINTS.glob("global_step_*"):
        if (
            not path.is_symlink()
            and int(path.name.removeprefix("global_step_")) not in keep
        ):
            shutil.rmtree(path)


def train_segment(previous, target, state):
    checkpoint = CHECKPOINTS / f"global_step_{target}"
    marker = checkpoint / "training_complete.json"
    if marker.exists():
        return
    if checkpoint.exists():
        raise RuntimeError(f"Unconfirmed partial checkpoint: {checkpoint}")
    if shutil.disk_usage(ROOT).free < 30 * 1024**3:
        raise RuntimeError("Less than 30 GiB free for the next official checkpoint")
    args = [
        "examples/sft/train_vla_sft.py",
        "--config-name",
        CONFIG,
        f"runner.max_steps={target}",
        "runner.save_interval=5000",
        f"+data.resume_step={previous}",
    ]
    if previous:
        args.append(f"runner.resume_dir={CHECKPOINTS / f'global_step_{previous}'}")
    run(
        args,
        f"stage1_train_{previous:05d}_{target:05d}.log",
        state,
        f"train_to_{target}",
    )
    marker.write_text(
        json.dumps({"step": target, "official_entry_exit_code": 0}) + "\n"
    )


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    lock = (ROOT / "stage1_controller.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = (
        json.loads(STATE.read_text())
        if STATE.exists()
        else {
            "completed_step": 0,
            "best_step": 0,
            "best": None,
            "anchor": None,
            "bad_checks": 0,
            "started": now(),
            "training_seconds": 0.0,
            "minimum_steps": 5000,
            "maximum_steps": 20000,
            "validation_interval": 500,
            "relative_improvement": 0.005,
            "patience": 5,
            "freeze_vla": True,
        }
    )
    state.update(
        minimum_steps=20000,
        maximum_steps=20000,
        early_stopping_enabled=False,
        validation_interval=5000,
        checkpoint_milestones=list(MILESTONES),
    )
    if state.get("status") == "complete" and state["completed_step"] >= 20000:
        return
    state.pop("stop_reason", None)
    try:
        smoke = json.loads((ROOT / "stage1_frozen_smoke.json").read_text())
        assert (
            smoke["checkpoint_saved"]
            and smoke["resume_update_completed"]
            and not smoke["synthetic_observations_only"]
        )
        assert all(r["vla_unchanged"] for group in smoke["steps"] for r in group)
        if "initial_validation" not in state:
            initial = ROOT / "stage1_initial_validation.json"
            if initial.exists():
                result = json.loads(initial.read_text())
            else:
                result = validate(BASE, 0, state)
            state["initial_validation"] = {
                key: result[key]
                for key in (
                    "rlt_mse",
                    "vla_loss",
                    "per_sample_mean_feature_mse",
                    "samples",
                )
            }
            save_state(state)
        while state["completed_step"] < state["maximum_steps"]:
            previous = state["completed_step"]
            # Validate a completed checkpoint before resuming from the next milestone.
            pending = sorted(
                int(p.name.removeprefix("global_step_"))
                for p in CHECKPOINTS.glob("global_step_*")
                if (p / "training_complete.json").exists()
                and previous < int(p.name.removeprefix("global_step_")) <= 20000
            )
            target = (
                pending[0]
                if pending
                else next(step for step in MILESTONES if step > previous)
            )
            checkpoint = CHECKPOINTS / f"global_step_{target}"
            started = now()
            clock = time.monotonic()
            if previous == 0:
                train_segment(0, 100, state)
                previous = 100
            train_segment(previous, target, state)
            result = validate(checkpoint, target, state)
            score = float(result["rlt_mse"])
            if (
                float(result["vla_loss"])
                > float(state["initial_validation"]["vla_loss"]) * 1.05
            ):
                raise RuntimeError(
                    "Frozen VLA validation loss changed by more than 5%; inspect before continuing"
                )
            eligible = state["best"] is None or score < state["best"]
            if eligible:
                state.update(best=score, best_step=target)
                pointer = CHECKPOINTS / "best_model"
                tmp = CHECKPOINTS / ".best_model.tmp"
                tmp.unlink(missing_ok=True)
                tmp.symlink_to(checkpoint.name, target_is_directory=True)
                tmp.replace(pointer)
            if state["anchor"] is None or score < state["anchor"] * (
                1 - state["relative_improvement"]
            ):
                state.update(anchor=score, bad_checks=0)
            else:
                state["bad_checks"] += 1
            state["training_seconds"] += time.monotonic() - clock
            state.update(
                completed_step=target,
                last_validation=result,
                last_validation_path=str(ROOT / f"stage1_validation_{target:05d}.json"),
                updated=now(),
            )
            state["hours_to_maximum_estimate"] = (
                state["training_seconds"]
                / target
                * (state["maximum_steps"] - target)
                / 3600
            )
            state["hours_to_minimum_estimate"] = (
                state["training_seconds"]
                / target
                * max(0, state["minimum_steps"] - target)
                / 3600
            )
            save_state(state)
            with (ROOT / "stage1_validation.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(
                        {
                            "step": target,
                            "rlt_mse": score,
                            "vla_loss": result["vla_loss"],
                            "mean_feature_mse": result["per_sample_mean_feature_mse"],
                            "best_step": state["best_step"],
                            "bad_checks": state["bad_checks"],
                        }
                    )
                    + "\n"
                )
            record(
                started,
                f"远程：冻结 VLA 的 RLT 训练完成 {target} 步，验证 MSE {score:.6f}，当前最优 {state['best_step']} 步。",
            )
            run(
                [
                    "toolkits/plot_rlt_loss.py",
                    "--log-dir",
                    str(ROOT / "tensorboard"),
                    "--output-dir",
                    str(ROOT / "stage1/plots"),
                ],
                "stage1_plot.log",
                state,
                "plot",
            )
            prune(state)
            # The requested budget is exactly 20,000 updates; retain best-model selection.
        state.setdefault("stop_reason", "budget_reached")
        state.update(
            status="complete",
            phase="stage1_validated",
            updated=now(),
            best_checkpoint=str(CHECKPOINTS / f"global_step_{state['best_step']}"),
        )
        save_state(state)
        record(
            state["started"],
            f"远程：RLT 特征训练及周期验证完成，选定 {state['best_step']} 步权重，进入 actor 特征缓存准备。",
        )
    except BaseException as error:
        state.update(status="failed", error=repr(error), updated=now())
        save_state(state)
        record(
            state.get("started", now()),
            f"远程：RLT 自动流程中断，当前已验证至 {state['completed_step']} 步；原因 {str(error)[:180]}。",
        )
        raise


if __name__ == "__main__":
    main()
