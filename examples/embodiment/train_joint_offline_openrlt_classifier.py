#!/usr/bin/env python3
# Copyright 2026 The GIGA Authors.
#
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _find_repo_root(start: Path) -> Path:
    for path in [start, *start.parents]:
        if (path / "rlinf").is_dir() and (path / "examples").is_dir():
            return path
    raise RuntimeError(f"Could not find RLinf repo root from {start}")


REPO_ROOT = _find_repo_root(Path(__file__).resolve().parent)


@dataclass(frozen=True)
class StageConfig:
    config: str
    enabled: bool = True
    output_subdir: str = ""
    extra_args: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class JointTrainConfig:
    run_name: str
    output_root: str | None = None
    timestamped_run_dir: bool = True
    co_locate_outputs: bool = True
    run_mode: str = "sequential"
    stop_on_failure: bool = True
    openrlt: StageConfig = field(
        default_factory=lambda: StageConfig(
            config="offline_train_on_online122_residual_q010_new",
            output_subdir="openrlt",
        )
    )
    classifier: StageConfig = field(
        default_factory=lambda: StageConfig(
            config="intervention_classifier_online122_residual_q010_new",
            output_subdir="intervention_classifier",
        )
    )


@dataclass(frozen=True)
class StageCommand:
    name: str
    cmd: list[str]
    output_dir: Path
    log_path: Path


def _stage_from_dict(raw: dict[str, Any], default_subdir: str) -> StageConfig:
    return StageConfig(
        config=str(raw["config"]),
        enabled=bool(raw.get("enabled", True)),
        output_subdir=str(raw.get("output_subdir", default_subdir)),
        extra_args=[str(item) for item in raw.get("extra_args", [])],
    )


def load_joint_config(path: Path) -> JointTrainConfig:
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise TypeError(f"Expected mapping in {path}, got {type(raw).__name__}")
    if "run_name" not in raw:
        raise KeyError("Joint config must define `run_name`.")
    return JointTrainConfig(
        run_name=str(raw["run_name"]),
        output_root=raw.get("output_root"),
        timestamped_run_dir=bool(raw.get("timestamped_run_dir", True)),
        co_locate_outputs=bool(raw.get("co_locate_outputs", True)),
        run_mode=str(raw.get("run_mode", "sequential")),
        stop_on_failure=bool(raw.get("stop_on_failure", True)),
        openrlt=_stage_from_dict(raw.get("openrlt", {}), "openrlt"),
        classifier=_stage_from_dict(
            raw.get("classifier", {}), "intervention_classifier"
        ),
    )


def resolve_config_path(config: str, config_dir: Path) -> Path:
    value = Path(os.path.expanduser(str(config)))
    if value.suffix in {".yaml", ".yml"} or value.parent != Path("."):
        path = value
    else:
        path = config_dir / f"{config}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    return path


def resolve_run_dir(
    cfg: JointTrainConfig, repo_root: Path, now: float | None = None
) -> Path:
    output_root = (
        Path(os.path.expanduser(cfg.output_root))
        if cfg.output_root
        else repo_root / "logs"
    )
    if cfg.timestamped_run_dir:
        stamp = time.strftime("%Y%m%d-%H:%M:%S", time.localtime(now or time.time()))
        return output_root / f"{stamp}-{cfg.run_name}"
    return output_root / cfg.run_name


def _output_override_args(
    cfg: JointTrainConfig, stage: StageConfig, run_dir: Path
) -> tuple[Path, list[str]]:
    if cfg.co_locate_outputs:
        output_dir = run_dir / stage.output_subdir
        return output_dir, ["--output-dir", str(output_dir)]
    return run_dir / stage.output_subdir, []


def build_stage_commands(
    cfg: JointTrainConfig,
    repo_root: Path,
    config_dir: Path,
    run_dir: Path,
) -> list[StageCommand]:
    stages: list[StageCommand] = []
    embodied = repo_root / "examples" / "embodiment"
    if cfg.openrlt.enabled:
        config_path = resolve_config_path(cfg.openrlt.config, config_dir)
        output_dir, output_args = _output_override_args(cfg, cfg.openrlt, run_dir)
        stages.append(
            StageCommand(
                name="openrlt",
                cmd=[
                    "bash",
                    str(embodied / "run_offline_train_openrlt.sh"),
                    str(config_path),
                    *cfg.openrlt.extra_args,
                    *output_args,
                ],
                output_dir=output_dir,
                log_path=run_dir / "openrlt.log",
            )
        )
    if cfg.classifier.enabled:
        config_path = resolve_config_path(cfg.classifier.config, config_dir)
        output_dir, output_args = _output_override_args(cfg, cfg.classifier, run_dir)
        stages.append(
            StageCommand(
                name="classifier",
                cmd=[
                    "bash",
                    str(embodied / "run_intervention_classifier_train.sh"),
                    str(config_path),
                    *cfg.classifier.extra_args,
                    *output_args,
                ],
                output_dir=output_dir,
                log_path=run_dir / "classifier.log",
            )
        )
    if not stages:
        raise ValueError("At least one joint training stage must be enabled.")
    return stages


def _stage_to_dict(stage: StageCommand) -> dict[str, Any]:
    return {
        "name": stage.name,
        "cmd": stage.cmd,
        "cmd_string": shlex.join(stage.cmd),
        "output_dir": str(stage.output_dir),
        "log_path": str(stage.log_path),
    }


def expected_outputs(run_dir: Path) -> dict[str, str]:
    return {
        "openrlt_final_checkpoint": str(
            run_dir / "openrlt" / "checkpoints" / "final" / "actor_critic.pt"
        ),
        "openrlt_standalone_checkpoint": str(
            run_dir / "openrlt" / "checkpoints" / "final" / "actor_critic_standalone.pt"
        ),
        "classifier_best_checkpoint": str(
            run_dir / "intervention_classifier" / "best_intervention_classifier.pt"
        ),
        "classifier_final_checkpoint": str(
            run_dir / "intervention_classifier" / "final_intervention_classifier.pt"
        ),
    }


def write_manifest(
    run_dir: Path,
    config_path: Path,
    cfg: JointTrainConfig,
    stages: list[StageCommand],
    dry_run: bool,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "config_path": str(config_path),
        "run_dir": str(run_dir),
        "dry_run": bool(dry_run),
        "joint_config": asdict(cfg),
        "stages": [_stage_to_dict(stage) for stage in stages],
        "results": results,
        "expected_outputs": expected_outputs(run_dir),
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def run_stage(stage: StageCommand, repo_root: Path) -> dict[str, Any]:
    stage.output_dir.mkdir(parents=True, exist_ok=True)
    stage.log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{repo_root}:{env.get('PYTHONPATH', '')}"
    start = time.time()
    with stage.log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + shlex.join(stage.cmd) + "\n")
        log.flush()
        proc = subprocess.Popen(
            stage.cmd,
            cwd=str(repo_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
        rc = proc.wait()
    return {
        "name": stage.name,
        "returncode": int(rc),
        "elapsed_sec": float(time.time() - start),
        "log_path": str(stage.log_path),
        "output_dir": str(stage.output_dir),
    }


def run_joint_training(
    config_path: Path,
    cfg: JointTrainConfig,
    repo_root: Path,
    dry_run: bool,
) -> int:
    if cfg.run_mode != "sequential":
        raise ValueError("Only run_mode='sequential' is supported in this version.")
    config_dir = repo_root / "examples" / "embodiment" / "config"
    run_dir = resolve_run_dir(cfg, repo_root)
    stages = build_stage_commands(cfg, repo_root, config_dir, run_dir)
    if dry_run:
        write_manifest(run_dir, config_path, cfg, stages, dry_run=True, results=[])
        print(json.dumps({"dry_run_manifest": str(run_dir / "manifest.json")}))
        for stage in stages:
            print(f"[dry-run] {stage.name}: {shlex.join(stage.cmd)}")
        return 0

    results: list[dict[str, Any]] = []
    for stage in stages:
        result = run_stage(stage, repo_root)
        results.append(result)
        write_manifest(
            run_dir, config_path, cfg, stages, dry_run=False, results=results
        )
        if result["returncode"] != 0 and cfg.stop_on_failure:
            return int(result["returncode"])
    write_manifest(run_dir, config_path, cfg, stages, dry_run=False, results=results)
    print(json.dumps({"joint_manifest": str(run_dir / "manifest.json")}, indent=2))
    return 0 if all(item["returncode"] == 0 for item in results) else 1


def replace_stage_enabled(
    cfg: JointTrainConfig, stage_name: str, enabled: bool
) -> JointTrainConfig:
    openrlt = cfg.openrlt
    classifier = cfg.classifier
    if stage_name == "openrlt":
        openrlt = StageConfig(
            config=openrlt.config,
            enabled=enabled,
            output_subdir=openrlt.output_subdir,
            extra_args=list(openrlt.extra_args),
        )
    elif stage_name == "classifier":
        classifier = StageConfig(
            config=classifier.config,
            enabled=enabled,
            output_subdir=classifier.output_subdir,
            extra_args=list(classifier.extra_args),
        )
    else:
        raise ValueError(f"Unknown stage: {stage_name}")
    return JointTrainConfig(
        run_name=cfg.run_name,
        output_root=cfg.output_root,
        timestamped_run_dir=cfg.timestamped_run_dir,
        co_locate_outputs=cfg.co_locate_outputs,
        run_mode=cfg.run_mode,
        stop_on_failure=cfg.stop_on_failure,
        openrlt=openrlt,
        classifier=classifier,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-openrlt", action="store_true")
    parser.add_argument("--skip-classifier", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    cfg = load_joint_config(config_path)
    if args.skip_openrlt:
        cfg = replace_stage_enabled(cfg, "openrlt", False)
    if args.skip_classifier:
        cfg = replace_stage_enabled(cfg, "classifier", False)
    return run_joint_training(config_path, cfg, REPO_ROOT, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
