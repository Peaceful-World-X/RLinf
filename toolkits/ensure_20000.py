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

"""Wait for the active controller, then finish the requested 20k RLT budget."""

import csv
import datetime as dt
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from zoneinfo import ZoneInfo

from rlinf.utils.logging import get_logger

LOGGER = get_logger()

ROOT = Path("/mnt/20T-1/sudu/.data/RLinf/g1_rlt/20261027_1042")
REPO = Path(__file__).resolve().parents[1]
OUT = Path("/mnt/20T-1/sudu/.result/g1_rlt/20261027_1042/rlt_20000")
JOINT = Path("/mnt/20T-1/sudu/.result/g1_rlt_joint14/20261028_1530")
STATE = ROOT / "stage1_progress.json"
PROGRESS = ROOT / "target_progress.json"
PYTHON = str(REPO / "env/bin/python")
MILESTONES = (5000, 10000, 15000, 20000)


def now():
    return dt.datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M")


def write(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2) + "\n")
    temp.replace(path)


def record(message):
    with (REPO / "TIME.csv").open("a", newline="") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        csv.writer(stream).writerow([now(), now(), message])
    LOGGER.info("%s %s", now(), message)


def controller_alive(pid):
    try:
        p = Path("/proc") / str(pid)
        cmd = p.joinpath("cmdline").read_bytes().replace(b"\0", b" ").decode()
        return str(REPO / "toolkits/run_stage1.py") in cmd and p.joinpath(
            "stat"
        ).read_text().split(") ", 1)[1].split()[0] not in ("Z", "X")
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return False


def any_live_controller():
    for p in Path("/proc").iterdir():
        if p.name.isdigit():
            try:
                if controller_alive(int(p.name)):
                    return int(p.name)
            except PermissionError:
                pass
    return None


def snapshot(state, pid):
    log = Path(state.get("current_log", ""))
    step = state["completed_step"]
    loss = None
    if log.is_file():
        with log.open("rb") as stream:
            stream.seek(max(0, log.stat().st_size - 131072))
            tail = stream.read().decode(errors="replace")
        entries = re.findall(
            r"(\d+)/(\d+)\s+\[[^\r\n]*?train/rlt_loss=([\d.eE+-]+)", tail
        )
        if entries:
            step = int(entries[-1][0])
            loss = float(entries[-1][2])
    for checkpoint in (ROOT / "stage1/checkpoints").glob("global_step_*"):
        if (checkpoint / "training_complete.json").exists():
            step = max(step, int(checkpoint.name.removeprefix("global_step_")))
    result = {
        "updated": now(),
        "status": "running",
        "target_step": 20000,
        "controller_pid": pid,
        "controller_alive": controller_alive(pid) if pid else False,
        "child_pid": state.get("child_pid"),
        "phase": state.get("phase"),
        "optimizer_step": step,
        "last_logged_rlt_loss": loss,
        "last_validated_step": state["completed_step"],
        "best_step": state["best_step"],
        "best_mse": state["best"],
        "physical_gpus": [1],
        "current_log": str(log),
        "fixed_actor_rlt_step": 8500,
        "joint_actor_completed": (JOINT / "delivery_audit.json").is_file(),
        "checkpoint_milestones": list(MILESTONES),
    }
    write(PROGRESS, result)


def finalize(state):
    import torch
    from omegaconf import OmegaConf

    from rlinf.models.embodiment.openpi.checkpoint import _normalize_state_dict
    from rlinf.utils.ckpt_convertor.openpi._core import as_state_dict

    torch.set_num_threads(4)
    assert state["completed_step"] == 20000 and state["status"] == "complete"
    validation = json.loads((ROOT / "stage1_validation_20000.json").read_text())
    assert validation["samples"] == 512 and validation["finite"]
    assert validation["vla_loss"] <= state["initial_validation"]["vla_loss"] * 1.05
    checkpoint = ROOT / "stage1/checkpoints/global_step_20000"
    assert (
        json.loads((checkpoint / "training_complete.json").read_text())["step"] == 20000
    )
    identity = json.loads((ROOT / "frozen_base_identity.json").read_text())
    tensors = _normalize_state_dict(
        as_state_dict(
            torch.load(
                checkpoint / "actor/model_state_dict/full_weights.pt",
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
        )
    )
    counts = {"base": 0, "rlt": 0}
    hashes = {"base": hashlib.sha256(), "rlt": hashlib.sha256()}
    for name, value in sorted(tensors.items()):
        assert torch.isfinite(value).all(), name
        group = "rlt" if name.startswith("rlt_module.") else "base"
        counts[group] += 1
        digest = hashes[group]
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.contiguous().view(torch.uint8).numpy())
    assert (
        counts == identity["counts"]
        and hashes["base"].hexdigest() == identity["base_sha256"]
    )
    assert hashes["rlt"].hexdigest() != identity["rlt_smoke_sha256"]
    del tensors
    OUT.mkdir(parents=True, exist_ok=True)
    checkpoints = OUT / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    retained = sorted(
        {20000, state["best_step"]}
        | {
            step
            for step in MILESTONES
            if (
                ROOT / f"stage1/checkpoints/global_step_{step}/training_complete.json"
            ).exists()
        }
    )
    for step in retained:
        src = ROOT / f"stage1/checkpoints/global_step_{step}"
        dest = checkpoints / src.name
        if not src.is_symlink():
            shutil.move(str(src), dest)
            src.symlink_to(dest, target_is_directory=True)
    shutil.copy2(ROOT / "stage1_validation_20000.json", OUT / "validation_20000.json")
    shutil.copy2(
        ROOT / f"stage1_validation_{state['best_step']:05d}.json",
        OUT / "validation_best.json",
    )
    shutil.copy2(STATE, OUT / "training_progress.json")
    shutil.copy2(ROOT / "stage1_validation.jsonl", OUT / "validation.jsonl")
    shutil.copy2(ROOT / "episode_split.json", OUT / "episode_split.json")
    shutil.copy2(ROOT / "training_target.json", OUT / "training_target.json")
    shutil.copy2(REPO / "toolkits/run_stage1.py", OUT / "run_stage1.py")
    logs = OUT / "logs"
    logs.mkdir(exist_ok=True)
    for pattern in (
        "stage1_train_*.log",
        "stage1_validation_*.log",
        "stage1_controller*.log",
    ):
        for log in ROOT.glob(pattern):
            shutil.copy2(log, logs / log.name)
    shutil.copytree(ROOT / "tensorboard", logs / "tensorboard", dirs_exist_ok=True)
    for step in retained:
        shutil.copy2(
            ROOT / f"stage1_validation_{step:05d}.json", OUT / f"validation_{step}.json"
        )
    (OUT / "configs").mkdir(exist_ok=True)
    shutil.copy2(
        REPO / "examples/sft/config/g1_rlt_stage1_sft_openpi_pi05.yaml",
        OUT / "configs/stage1.yaml",
    )
    feature = OmegaConf.load(
        "/mnt/20T-1/sudu/.result/g1_rlt/20261027_1042/configs/feature_config.yaml"
    )
    feature.model_path = str(checkpoints / "global_step_20000")
    OmegaConf.save(feature, OUT / "configs/feature_20000.yaml")
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONPATH": str(REPO),
        "OMP_NUM_THREADS": "4",
    }
    with (OUT / "plot.log").open("w") as log:
        subprocess.run(
            [
                PYTHON,
                "toolkits/plot_rlt_loss.py",
                "--log-dir",
                str(ROOT / "tensorboard"),
                "--output-dir",
                str(OUT / "plots"),
                "--run-dir",
                str(ROOT),
                "--config",
                str(OUT / "configs/stage1.yaml"),
            ],
            cwd=REPO,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    plot = json.loads((OUT / "plots/rlt_loss.json").read_text())
    assert plot["last_completed_step"] == 20000 and plot["logged_points"] == 20000, (
        "Incomplete full-training loss history"
    )
    actor = json.loads((JOINT / "report.json").read_text())
    audit = json.loads((JOINT / "delivery_audit.json").read_text())
    assert audit["status"] == "passed" and Path(actor["actor_checkpoint"]).is_file()
    report = {
        "status": "validated",
        "completed_step": 20000,
        "best_step": state["best_step"],
        "final_rlt_mse": validation["rlt_mse"],
        "best_rlt_mse": state["best"],
        "frozen_vla_identity_verified": True,
        "finite_weights": True,
        "counts": counts,
        "base_sha256": hashes["base"].hexdigest(),
        "rlt_sha256": hashes["rlt"].hexdigest(),
        "checkpoint_20000": str(
            checkpoints / "global_step_20000/actor/model_state_dict/full_weights.pt"
        ),
        "best_checkpoint": str(
            checkpoints
            / f"global_step_{state['best_step']}/actor/model_state_dict/full_weights.pt"
        ),
        "actor_checkpoint": actor["actor_checkpoint"],
        "actor_feature_checkpoint": actor["feature_checkpoint"],
        "actor_feature_step": 8500,
        "actor_action_dim": 14,
        "completed": now(),
        "retained_rlt_steps": retained,
        "unavailable_historical_milestones": [
            s for s in MILESTONES if s not in retained
        ],
        "loss_plot": str(OUT / "plots/rlt_loss.png"),
        "note": "Actor remains paired with frozen step8500; using step20000 requires rebuilding its features and training a new actor.",
    }
    write(OUT / "report.json", report)
    write(
        PROGRESS,
        {
            "status": "complete",
            "target_step": 20000,
            "optimizer_step": 20000,
            "updated": now(),
            "report": str(OUT / "report.json"),
        },
    )
    record(
        "RLT已完成20000步及固定512帧验证，冻结VLA指纹通过；8500步配套的14维actor已验证交付。"
    )
    shutil.copy2(REPO / "TIME.csv", OUT / "TIME.csv")


def main():
    lock = (ROOT / "target_controller.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (OUT / "report.json").exists() and json.loads(
        (OUT / "report.json").read_text()
    ).get("status") == "validated":
        return
    record(
        "启动20000步接续监督：等待现有RLT训练完成当前流程，必要时从已验证断点继续；仅使用物理1号卡。"
    )
    retries = 0
    while True:
        state = json.loads(STATE.read_text())
        pid = any_live_controller()
        snapshot(state, pid)
        if pid:
            time.sleep(30)
            continue
        # A released controller lock and no live training child are required for handoff.
        control = (ROOT / "stage1_controller.lock").open("a")
        try:
            fcntl.flock(control, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            control.close()
            time.sleep(30)
            continue
        child = state.get("child_pid")
        if child:
            try:
                child_alive = (Path("/proc") / str(child) / "stat").read_text().split(
                    ") ", 1
                )[1].split()[0] not in ("Z", "X")
            except (FileNotFoundError, ProcessLookupError):
                child_alive = False
            if child_alive:
                control.close()
                time.sleep(30)
                continue
        if state["completed_step"] == 20000 and state["status"] == "complete":
            control.close()
            finalize(state)
            return
        assert state["completed_step"] < 20000
        for next_dir in (ROOT / "stage1/checkpoints").glob("global_step_*"):
            if (
                int(next_dir.name.removeprefix("global_step_"))
                > state["completed_step"]
                and not (next_dir / "training_complete.json").exists()
            ):
                raise RuntimeError(
                    f"Partial unconfirmed checkpoint requires inspection: {next_dir}"
                )
        if state.get("status") == "failed":
            retries += 1
            if retries > 2:
                raise RuntimeError(
                    "RLT controller failed repeatedly; inspect logs before retrying"
                )
        state.update(
            status="continuing_to_20000",
            minimum_steps=20000,
            maximum_steps=20000,
            early_stopping_enabled=False,
        )
        state.pop("stop_reason", None)
        state.pop("error", None)
        state.pop("child_pid", None)
        write(STATE, state)
        control.close()
        env = {
            **os.environ,
            "PYTHONPATH": str(REPO),
            "CUDA_VISIBLE_DEVICES": "1",
            "RAY_ADDRESS": "local",
            "OMP_NUM_THREADS": "4",
        }
        with (ROOT / "stage1_controller_20000.log").open("a") as log:
            process = subprocess.Popen(
                [PYTHON, "-u", str(REPO / "toolkits/run_stage1.py")],
                cwd=REPO,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        write(
            ROOT / "stage1_controller_launch.json",
            {
                "pid": process.pid,
                "started": now(),
                "script": str(REPO / "toolkits/run_stage1.py"),
                "physical_training_gpus": [1],
                "target": 20000,
            },
        )
        record(
            f"从已验证{state['completed_step']}步接续RLT至20000步，平台期仅记录指标，不提前结束。"
        )
        time.sleep(30)


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        write(
            PROGRESS,
            {
                "status": "needs_attention",
                "updated": now(),
                "error": repr(error),
                "target_step": 20000,
            },
        )
        record(f"20000步监督发现需处理的问题：{error}")
        raise
