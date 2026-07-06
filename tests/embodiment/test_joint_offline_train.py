# Copyright 2026 The GIGA Authors.
#
from pathlib import Path

import yaml

from examples.embodiment.train_joint_offline_openrlt_classifier import (
    StageConfig,
    build_stage_commands,
    load_joint_config,
    resolve_config_path,
    resolve_run_dir,
    write_manifest,
)


def test_resolve_config_path_accepts_config_name(tmp_path: Path):
    config_dir = tmp_path / "examples" / "embodiment" / "config"
    config_dir.mkdir(parents=True)
    expected = config_dir / "offline_train_on_online122_residual_q010_new.yaml"
    expected.write_text("x: 1\n", encoding="utf-8")

    actual = resolve_config_path(
        "offline_train_on_online122_residual_q010_new", config_dir
    )

    assert actual == expected


def test_resolve_config_path_accepts_yaml_path(tmp_path: Path):
    config_file = tmp_path / "custom.yaml"
    config_file.write_text("x: 1\n", encoding="utf-8")

    actual = resolve_config_path(str(config_file), tmp_path / "unused")

    assert actual == config_file


def test_load_joint_config_parses_nested_stage_configs(tmp_path: Path):
    cfg_file = tmp_path / "joint.yaml"
    cfg_file.write_text(
        yaml.safe_dump(
            {
                "run_name": "joint_test",
                "output_root": str(tmp_path / "logs"),
                "timestamped_run_dir": False,
                "co_locate_outputs": True,
                "run_mode": "sequential",
                "stop_on_failure": True,
                "openrlt": {
                    "enabled": True,
                    "config": "openrlt_cfg",
                    "output_subdir": "openrlt",
                    "extra_args": ["--max-steps", "2"],
                },
                "classifier": {
                    "enabled": True,
                    "config": "classifier_cfg",
                    "output_subdir": "classifier",
                    "extra_args": ["--max-steps", "2"],
                },
            }
        ),
        encoding="utf-8",
    )

    cfg = load_joint_config(cfg_file)

    assert cfg.run_name == "joint_test"
    assert cfg.openrlt == StageConfig(
        enabled=True,
        config="openrlt_cfg",
        output_subdir="openrlt",
        extra_args=["--max-steps", "2"],
    )
    assert cfg.classifier.output_subdir == "classifier"


def test_build_stage_commands_uses_separate_output_dirs(tmp_path: Path):
    repo = tmp_path / "RLinf"
    config_dir = repo / "examples" / "embodiment" / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "openrlt.yaml").write_text("x: 1\n", encoding="utf-8")
    (config_dir / "classifier.yaml").write_text("x: 1\n", encoding="utf-8")
    run_dir = tmp_path / "logs" / "joint"

    cfg_file = tmp_path / "joint.yaml"
    cfg_file.write_text(
        yaml.safe_dump(
            {
                "run_name": "joint",
                "output_root": str(tmp_path / "logs"),
                "timestamped_run_dir": False,
                "co_locate_outputs": True,
                "openrlt": {"config": "openrlt", "output_subdir": "openrlt"},
                "classifier": {
                    "config": "classifier",
                    "output_subdir": "intervention_classifier",
                },
            }
        ),
        encoding="utf-8",
    )
    cfg = load_joint_config(cfg_file)

    stages = build_stage_commands(cfg, repo, config_dir, run_dir)

    assert [stage.name for stage in stages] == ["openrlt", "classifier"]
    assert str(run_dir / "openrlt") in stages[0].cmd
    assert str(run_dir / "intervention_classifier") in stages[1].cmd
    assert stages[0].output_dir != stages[1].output_dir
    assert "run_offline_train_openrlt.sh" in " ".join(stages[0].cmd)
    assert "run_intervention_classifier_train.sh" in " ".join(stages[1].cmd)


def test_build_stage_commands_respects_disabled_stage(tmp_path: Path):
    repo = tmp_path / "RLinf"
    config_dir = repo / "examples" / "embodiment" / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "classifier.yaml").write_text("x: 1\n", encoding="utf-8")

    cfg_file = tmp_path / "joint.yaml"
    cfg_file.write_text(
        yaml.safe_dump(
            {
                "run_name": "joint",
                "output_root": str(tmp_path / "logs"),
                "timestamped_run_dir": False,
                "openrlt": {"enabled": False, "config": "openrlt"},
                "classifier": {"enabled": True, "config": "classifier"},
            }
        ),
        encoding="utf-8",
    )
    cfg = load_joint_config(cfg_file)

    stages = build_stage_commands(cfg, repo, config_dir, tmp_path / "run")

    assert [stage.name for stage in stages] == ["classifier"]


def test_dry_run_manifest_contains_independent_checkpoint_paths(tmp_path: Path):
    repo = tmp_path / "RLinf"
    config_dir = repo / "examples" / "embodiment" / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "openrlt.yaml").write_text("x: 1\n", encoding="utf-8")
    (config_dir / "classifier.yaml").write_text("x: 1\n", encoding="utf-8")
    cfg_file = tmp_path / "joint.yaml"
    cfg_file.write_text(
        yaml.safe_dump(
            {
                "run_name": "joint",
                "output_root": str(tmp_path / "logs"),
                "timestamped_run_dir": False,
                "co_locate_outputs": True,
                "openrlt": {"config": "openrlt", "output_subdir": "openrlt"},
                "classifier": {
                    "config": "classifier",
                    "output_subdir": "intervention_classifier",
                },
            }
        ),
        encoding="utf-8",
    )

    cfg = load_joint_config(cfg_file)
    run_dir = resolve_run_dir(cfg, repo, now=0)
    stages = build_stage_commands(cfg, repo, config_dir, run_dir)
    manifest = write_manifest(run_dir, cfg_file, cfg, stages, dry_run=True, results=[])

    assert manifest["dry_run"] is True
    assert manifest["expected_outputs"]["openrlt_final_checkpoint"].endswith(
        "openrlt/checkpoints/final/actor_critic.pt"
    )
    assert manifest["expected_outputs"]["classifier_best_checkpoint"].endswith(
        "intervention_classifier/best_intervention_classifier.pt"
    )
    assert manifest["stages"][0]["output_dir"] != manifest["stages"][1]["output_dir"]
    assert (run_dir / "manifest.json").exists()
