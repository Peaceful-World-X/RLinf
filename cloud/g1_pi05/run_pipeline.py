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

"""Build G1 Hydra jobs from the official examples and execute their entry points."""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml
from assets_io import oss, save_json
from loss_log import hourly_updates, publish, restore_history
from sync_runs import RunStore

HERE = Path(__file__).resolve().parent
G1 = json.loads((HERE / "configs/g1.json").read_text())
ENTRY = {
    "returns": "advantage_labeling/recap/process/compute_returns.py",
    "recap_value": "advantage_labeling/recap/train_value.py",
    "steam_value": "advantage_labeling/steam/train_steam.py",
    "recap_adv": "advantage_labeling/recap/process/compute_advantages.py",
    "steam_adv": "advantage_labeling/steam/process/compute_advantages_ensemble.py",
    "policy": "policy_optimization/cfg_rl/train_cfg.py",
}
TEMPLATES = dict(
    returns="recap_compute_returns",
    recap_value="recap_value_model_sft",
    steam_value="steam_value_model_sft",
    recap_adv="recap_compute_advantages",
    steam_adv="steam_compute_advantages_ensemble",
    policy="cfg_rl_openpi",
)


def input_specs(root):
    import numpy as np
    from openpi.shared import normalize

    stats = normalize.load(root)
    for key in ("state", "actions"):
        for field in ("mean", "std", "q01", "q99"):
            arr = np.asarray(getattr(stats[key], field))
            if arr.ndim != 1 or len(arr) < 16 or not np.isfinite(arr).all():
                raise ValueError(f"Invalid G1 normalization: {key}.{field}")
    specs = []
    for item in json.loads((root / "datasets.json").read_text()):
        path = (root / item["path"]).resolve()
        if not path.is_relative_to(root.resolve()) or item["type"] not in (
            "sft",
            "rollout",
        ):
            raise ValueError("Invalid dataset path or type")
        meta = json.loads((path / "meta/info.json").read_text())
        features = meta["features"]
        for key in ("action", "observation.state"):
            if features[key]["shape"] != [16]:
                raise ValueError(f"{path}: expected 16-D {key}")
        for camera in G1["cameras"]:
            if "observation.images." + camera not in features:
                raise ValueError(f"{path}: missing {camera}")
        if item["type"] == "rollout":
            import pyarrow as pa
            import pyarrow.parquet as pq

            for file in (path / "data").rglob("*.parquet"):
                schema = pq.read_schema(file)
                idx = schema.get_field_index("is_success")
                if idx < 0 or not (
                    pa.types.is_boolean(schema.field(idx).type)
                    or pa.types.is_integer(schema.field(idx).type)
                ):
                    raise ValueError(
                        f"{file}: rollout requires scalar boolean or 0/1 is_success"
                    )
                flags = (
                    pq.read_table(file, columns=["is_success"])
                    .column(0)
                    .unique()
                    .to_pylist()
                )
                if any(v not in (True, False) for v in flags):
                    raise ValueError(f"{file}: invalid is_success values")
        weight = float(item.get("weight", 1))
        if (
            not math.isfinite(weight)
            or weight <= 0
            or any(x["dataset_path"] == str(path) for x in specs)
        ):
            raise ValueError("Invalid weight or repeated dataset")
        specs.append(dict(dataset_path=str(path), type=item["type"], weight=weight))
    if not specs:
        raise ValueError("Empty training set")
    return specs


def configure(repo, home, job, alg, smoke, n):
    root = job / "input"
    if smoke:
        from make_smoke_data import generate

        generate(root, G1["cameras"])
    data = input_specs(root)
    base = job / "pi05"
    base.mkdir(exist_ok=True)
    for filename in ("config.json", "model.safetensors"):
        f = (home / "models/pi05" / filename).resolve(strict=True)
        target = base / filename
        if target.exists() and not target.samefile(f):
            raise ValueError(f"Unexpected policy base file: {target}")
        if not target.exists():
            target.symlink_to(f)
    (base / "g1").mkdir(exist_ok=True)
    shutil.copyfile(root / "norm_stats.json", base / "g1/norm_stats.json")
    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

    pi = get_openpi_config(
        G1["policy_config"], model_path=str(base), repo_id=data[0]["dataset_path"]
    )
    if not pi.data.create(pi.assets_dirs, pi.model).norm_stats:
        raise ValueError("OpenPI did not load G1 norm stats")
    value_steps = 1 if smoke else int(os.environ["VALUE_STEPS"])
    policy_steps = 1 if smoke else int(os.environ["POLICY_STEPS"])
    if min(value_steps, policy_steps, n) <= 0:
        raise ValueError("Step budgets and GPU count must be positive")
    order = (["returns"] if alg == "recap" else []) + [
        alg + "_value",
        alg + "_adv",
        "policy",
    ]
    configs = job / "configs"
    configs.mkdir(exist_ok=True)
    tag = f"{alg}_g1"
    gemma = str(home / "models/gemma")
    vision = str(home / "models" / (alg + "_vision"))
    for stage in order:
        cfg = yaml.safe_load(
            (repo / "config" / (TEMPLATES[stage] + ".yaml")).read_text()
        )
        cfg["data"]["train_data_paths"] = data
        if "defaults" in cfg and "_self_" not in cfg["defaults"]:
            cfg["defaults"].append("_self_")
        if stage == "returns":
            cfg["data"].update(
                gamma=G1["gamma"],
                failure_reward=G1["failure_reward"],
                tag=G1["return_tag"],
                num_workers=2 if smoke else 8,
            )
        elif stage.endswith("_adv"):
            checkpoint = (
                job
                / "results/value/checkpoints"
                / f"global_step_{value_steps}"
                / "actor"
            )
            adv = cfg["advantage"]
            adv.update(
                value_checkpoint=str(checkpoint),
                batch_size=2 if smoke else 16,
                num_dataloader_workers_per_gpu=1 if smoke else 4,
                tag=tag,
            )
            if alg == "recap":
                adv.update(
                    value_checkpoint=str(checkpoint / "model_state_dict"),
                    returns_tag=G1["return_tag"],
                    positive_quantile=G1["recap_quantile"],
                )
                adv["model"].update(
                    siglip_path=vision,
                    gemma3_path=gemma,
                    tokenizer_path=gemma,
                    action_dim=16,
                    action_horizon=10,
                )
                cfg["data"]["train_data_paths"] = [
                    dict(x, robot_type="g1") for x in data
                ]
            else:
                cfg["data"].update(k=G1["steam_k"], camera_keys=G1["cameras"])
        else:
            policy = stage == "policy"
            steps = policy_steps if policy else value_steps
            runner, actor = cfg["runner"], cfg["actor"]
            interval = int(os.getenv("SAVE_INTERVAL", "1000"))
            if interval < 1:
                raise ValueError("SAVE_INTERVAL must be positive")
            actor["fsdp_config"].update(
                save_full_model_weights=True, sharding_strategy="full_shard"
            )
            stage_name = "policy" if policy else "value"
            runner.update(
                max_epochs=-1,
                max_steps=steps,
                save_interval=min(steps, interval),
                val_check_interval=-1,
                publish_completion_marker=True,
                loss_log_path=str(job / f"{stage_name}-loss.csv"),
            )
            runner["logger"].update(
                log_path=str(job / "results"),
                experiment_name=stage_name,
                logger_backends=[],
            )
            micro = int(
                os.getenv(
                    "POLICY_MICRO_BATCH" if policy else "VALUE_MICRO_BATCH",
                    "1" if smoke else "2",
                )
            )
            batch = (
                n * micro
                if smoke
                else int(os.getenv("POLICY_BATCH" if policy else "VALUE_BATCH", "256"))
            )
            if micro < 1 or batch < n * micro or batch % (n * micro):
                raise ValueError(
                    "global batch must be divisible by GPU_COUNT × micro batch"
                )
            actor.update(micro_batch_size=micro, global_batch_size=batch)
            opt = actor["optim"]
            opt.update(
                total_training_steps=steps,
                lr_warmup_steps=min(opt["lr_warmup_steps"], steps // 5),
            )
            model = actor["model"]
            if policy:
                cfg["data"].update(advantage_tag=tag, num_workers=1 if smoke else 4)
                model.update(model_path=str(base), num_action_chunks=50, action_dim=16)
                model["openpi"].update(
                    config_name=G1["policy_config"], num_images_in_input=3
                )
            else:
                cfg["data"].update(
                    eval_data_paths=[], train_num_workers=1 if smoke else 4
                )
                if alg == "recap":
                    cfg["data"].update(
                        tag=G1["return_tag"],
                        action_dim=16,
                        robot_type="g1",
                        train_data_paths=[
                            dict(x, robot_type="g1", model_type="pi05") for x in data
                        ],
                    )
                    model.update(
                        siglip_path=vision,
                        gemma3_path=gemma,
                        tokenizer_path=gemma,
                        action_dim=16,
                    )
                else:
                    cfg["data"].update(camera_keys=G1["cameras"], k=G1["steam_k"])
                    model.update(
                        vision_repo_id=vision,
                        language_repo_id=gemma,
                        tokenizer_path=gemma,
                    )
        (configs / (stage + ".yaml")).write_text(yaml.safe_dump(cfg, sort_keys=False))
    return order, value_steps, policy_steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    import ray
    import torch
    import transformers
    from openpi.models_pytorch import pi0_pytorch  # noqa: F401
    from safetensors import safe_open
    from torchcodec.decoders import VideoDecoder  # noqa: F401
    from transformers import AutoTokenizer

    home = Path(os.environ["RELEASE_DIR"])
    repo = Path(os.environ["REPO_PATH"]) / "examples/offline_rl"
    alg, mode = os.environ["ALGORITHM"], os.environ["JOB_MODE"]
    n = int(os.environ["GPU_COUNT"])
    job = home / "jobs" / alg / os.environ["RUN_ID"]
    job.mkdir(parents=True, exist_ok=True)
    print(
        "torch",
        torch.__version__,
        "ray",
        ray.__version__,
        "transformers",
        transformers.__version__,
        "CUDA",
        torch.version.cuda,
        "visible GPUs",
        torch.cuda.device_count(),
        flush=True,
    )
    if not args.prepare_only:
        if torch.cuda.device_count() != n:
            raise RuntimeError("GPU count differs from AIStudio resource request")
        for i in range(n):
            torch.ones(1, device=f"cuda:{i}").sum().item()
        subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc_per_node={n}",
                str(HERE / "check_devices.py"),
            ],
            check=True,
        )
    if mode == "env-only":
        for model in ("pi05", "gemma", alg + "_vision"):
            with safe_open(
                str(home / "models" / model / "model.safetensors"), framework="pt"
            ) as f:
                if not f.keys():
                    raise ValueError("Empty model: " + model)
        AutoTokenizer.from_pretrained(home / "models/gemma", local_files_only=True)
        from openpi.models.tokenizer import PaligemmaTokenizer

        PaligemmaTokenizer(max_len=200).tokenize("Thread the cable tie.")
        print("Verified offline models and tokenizers.", flush=True)
        return
    order, vs, ps = configure(repo, home, job, alg, mode == "smoke", n)
    if args.prepare_only:
        print(job / "configs")
        return
    identity = {
        k: os.environ.get(k)
        for k in (
            "RLINF_GIT_COMMIT",
            "GPU_COUNT",
            "JOB_MODE",
            "ALGORITHM",
            "RUN_ID",
            "DATA_SHA256",
            "POLICY_STEPS",
            "VALUE_STEPS",
            "SAVE_INTERVAL",
            "VALUE_MICRO_BATCH",
            "POLICY_MICRO_BATCH",
            "POLICY_BATCH",
            "VALUE_BATCH",
        )
    }
    from assets_io import sha

    identity["manifest"] = sha(os.environ["RELEASE_MANIFEST"])
    store = RunStore(job, identity)
    store.begin()
    for key in ("value", "policy"):
        checkpoint = store.restore(key) if os.getenv("RESUME") == "1" else 0
        if not checkpoint:
            restore_history(job, key, [])
        if checkpoint:
            cfg_path = (
                job
                / "configs"
                / ((alg + "_value" if key == "value" else key) + ".yaml")
            )
            cfg = yaml.safe_load(cfg_path.read_text())
            cfg["runner"]["resume_dir"] = str(
                job / "results" / key / "checkpoints" / f"global_step_{checkpoint}"
            )
            cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    save_json(job / "loss-ready.json", os.environ["ATTEMPT"])
    with hourly_updates(job, store.uri):
        run_stages(repo, job, n, order, vs, ps, store)
    publish(job, store.uri)
    result = job / "result.json"
    save_json(result, dict(algorithm=alg, mode=mode, policy_step=ps, value_step=vs))
    oss("cp", result, store.uri + "/result.json", "-f")


def run_stages(repo, job, n, order, vs, ps, store):
    for config in sorted((job / "configs").glob("*.yaml")):
        oss("cp", config, store.uri + "/configs/" + config.name, "-f")
    for stage in order:
        key = (
            "policy"
            if stage == "policy"
            else "value"
            if stage.endswith("_value")
            else None
        )
        step = ps if key == "policy" else vs
        if key and (key, step) in store.sent:
            continue
        args = [sys.executable]
        if stage.endswith("_adv"):
            args += [
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc_per_node={n}",
            ]
        args += [
            str(repo / ENTRY[stage]),
            "--config-path",
            str(job / "configs"),
            "--config-name",
            stage,
        ]
        print("Starting", stage, flush=True)
        with subprocess.Popen(args, cwd=job) as process:
            try:
                while True:
                    try:
                        code = process.wait(timeout=90)
                        break
                    except subprocess.TimeoutExpired:
                        if key:
                            store.sync(key)
            except BaseException:
                process.terminate()
                raise
        if key:
            store.sync(key)
        if code:
            raise subprocess.CalledProcessError(code, args)
        if key and (key, step) not in store.sent:
            raise RuntimeError(f"Missing complete {key} checkpoint at {step}")
    oss(
        "cp",
        job / "input/norm_stats.json",
        store.uri + "/policy/g1/norm_stats.json",
        "-f",
    )


if __name__ == "__main__":
    main()
