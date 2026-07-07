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

import subprocess
import sys
from pathlib import Path

import torch

from examples.embodiment import train_openrlt_right_arm_offline as train


def _write_trajectory(path: Path, num_chunks: int) -> None:
    actions = torch.arange(num_chunks * 210, dtype=torch.float32).reshape(
        num_chunks, 1, 210
    )
    states = torch.zeros(num_chunks, 1, 14, dtype=torch.float32)
    states[..., 7:14] = 0.1
    traj = {
        "actions": actions,
        "rewards": torch.zeros(num_chunks, 1, 1, dtype=torch.float32),
        "dones": torch.zeros(num_chunks, 1, 1, dtype=torch.bool),
        "curr_obs": {"states": states},
        "next_obs": {"states": states + 0.01},
        "forward_inputs": {
            "action": actions,
            "executed_env_action_absolute": actions,
        },
    }
    torch.save(traj, path)


def _cfg(tmp_path, data_dir, feature_cache, **overrides):
    kwargs = {
        "data_dir": str(data_dir),
        "feature_cache": str(feature_cache),
        "norm_stats_path": str(tmp_path / "norm.json"),
        "output_dir": str(tmp_path / "out"),
        "urdf_path": str(tmp_path / "robot.urdf"),
        "tcp_radius": 0.0,
        "warm_up_chunks": 0,
        "generate_feature_cache_if_missing": False,
    }
    kwargs.update(overrides)
    return train.TrainConfig(**kwargs)


def test_load_dataset_skips_per_trajectory_warm_up_chunks(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "SimpleUrdfKinematics", lambda _: object())
    monkeypatch.setattr(
        train, "right_tcp", lambda q, kin, offset: torch.zeros(1, 3).numpy()
    )

    data_dir = tmp_path / "demos"
    data_dir.mkdir()
    traj_path = data_dir / "trajectory_0.pt"
    _write_trajectory(traj_path, num_chunks=5)
    feature_cache = tmp_path / "features.pt"
    torch.save(
        {
            "rltoken": torch.arange(5 * 2048, dtype=torch.float32).reshape(5, 2048),
            "metas": [
                {
                    "path": str(traj_path),
                    "rel_chunk": rel,
                    "traj_index": 0,
                    "success": False,
                }
                for rel in range(5)
            ],
        },
        feature_cache,
    )

    data = train.load_dataset(_cfg(tmp_path, data_dir, feature_cache, warm_up_chunks=3))

    assert data["rel"].tolist() == [3, 4]
    assert [row["rel"] for row in data["rows"]] == [3, 4]
    assert data["skipped_warmup"] == 3


def test_load_dataset_uses_full_chunk_horizon_for_reward_and_done(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(train, "SimpleUrdfKinematics", lambda _: object())
    monkeypatch.setattr(
        train, "right_tcp", lambda q, kin, offset: torch.zeros(1, 3).numpy()
    )

    data_dir = tmp_path / "demos"
    data_dir.mkdir()
    traj_path = data_dir / "trajectory_0.pt"
    _write_trajectory(traj_path, num_chunks=4)
    traj = torch.load(traj_path, map_location="cpu", weights_only=False)
    traj["rewards"] = torch.zeros(4, 1, 15, dtype=torch.float32)
    traj["dones"] = torch.zeros(4, 1, 15, dtype=torch.bool)
    traj["rewards"][3, 0, 2] = 1.0
    traj["dones"][3, 0, 2] = True
    torch.save(traj, traj_path)

    feature_cache = tmp_path / "features.pt"
    torch.save(
        {
            "rltoken": torch.arange(4 * 2048, dtype=torch.float32).reshape(4, 2048),
            "metas": [
                {
                    "path": str(traj_path),
                    "rel_chunk": rel,
                    "traj_index": 0,
                    "success": False,
                }
                for rel in range(4)
            ],
        },
        feature_cache,
    )

    data = train.load_dataset(_cfg(tmp_path, data_dir, feature_cache))
    train.build_n_step_arrays(data, gamma=0.9, n_step=4)

    assert data["reward"].reshape(-1).tolist() == [0.0, 0.0, 0.0, 1.0]
    assert data["done"].reshape(-1).tolist() == [0.0, 0.0, 0.0, 1.0]
    assert torch.allclose(
        data["n_return"].reshape(-1), torch.tensor([0.9**3, 0.9**2, 0.9, 1.0])
    )
    assert data["n_done"].reshape(-1).tolist() == [1.0, 1.0, 1.0, 1.0]


def test_ensure_feature_cache_generates_missing_cache(tmp_path):
    data_dir = tmp_path / "demos"
    data_dir.mkdir()
    feature_cache = tmp_path / "features.pt"
    cfg = _cfg(
        tmp_path,
        data_dir,
        feature_cache,
        generate_feature_cache_if_missing=True,
    )
    calls = []

    def fake_generator(cfg_arg):
        calls.append(cfg_arg.feature_cache)
        torch.save(
            {"rltoken": torch.zeros(0, 2048), "metas": []}, cfg_arg.feature_cache
        )

    train.ensure_feature_cache(cfg, generator=fake_generator)

    assert calls == [str(feature_cache)]
    assert feature_cache.exists()


def test_train_parser_accepts_chunk_len_argument():
    proc = subprocess.run(
        [
            sys.executable,
            str(Path(train.__file__)),
            "--chunk-len",
            "15",
            "--help",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert proc.returncode == 0, proc.stdout
    assert "--chunk-len" in proc.stdout


def test_build_feature_model_cfg_reads_collection_actor_model(tmp_path):
    collection_cfg = tmp_path / "collect.yaml"
    collection_cfg.write_text(
        """
rollout:
  model:
    precision: ${actor.model.precision}
actor:
  model:
    model_path: /old/model
    num_action_chunks: 15
    action_dim: 14
    openpi:
      config_name: pi05_aloha_cubeinsert
      action_chunk: 15
      action_env_dim: 14
""",
        encoding="utf-8",
    )
    cfg = _cfg(
        tmp_path,
        tmp_path / "demos",
        tmp_path / "features.pt",
        model_path="/new/model",
        rl_token_path="/new/rltoken",
        feature_cache_model_config=str(collection_cfg),
    )

    model_cfg = train.build_feature_model_cfg(cfg)

    assert model_cfg.model_type == "openpi_rl_token"
    assert model_cfg.model_path == "/new/model"
    assert model_cfg.rl_token_path == "/new/rltoken"
    assert model_cfg.is_lora is False
    assert model_cfg.openpi.config_name == "pi05_aloha_cubeinsert"
    assert model_cfg.num_action_chunks == 15
    assert model_cfg.action_dim == 14
