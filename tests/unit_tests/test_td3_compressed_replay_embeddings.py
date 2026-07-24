# Copyright 2026 The GIGA Authors.
#
from pathlib import Path

import torch

from rlinf.data.embodied_io_struct import Trajectory
from rlinf.data.replay_buffer import TrajectoryReplayBuffer
from rlinf.models.embodiment.openpi.rl_token_policy import (
    REPLAY_ACTOR_TOKEN_KEY,
    REPLAY_CRITIC_TOKEN_1_KEY,
    REPLAY_CRITIC_TOKEN_2_KEY,
)
from rlinf.workers.actor.fsdp_td3_policy_worker import (
    compress_trajectory_replay_embeddings,
)
from rlinf.workers.env.env_worker import extract_replay_transition_features

EMBEDDING_KEYS = (
    REPLAY_ACTOR_TOKEN_KEY,
    REPLAY_CRITIC_TOKEN_1_KEY,
    REPLAY_CRITIC_TOKEN_2_KEY,
)


def _legacy_trajectory() -> Trajectory:
    trajectory = Trajectory(max_episode_length=4, model_weights_id="legacy")
    trajectory.actions = torch.zeros(2, 1, 3)
    trajectory.rewards = torch.zeros(2, 1, 1)
    trajectory.terminations = torch.zeros(2, 1, 1, dtype=torch.bool)
    trajectory.truncations = torch.zeros(2, 1, 1, dtype=torch.bool)
    trajectory.dones = torch.zeros(2, 1, 1, dtype=torch.bool)
    trajectory.curr_obs = {
        "visual_latent": torch.arange(48, dtype=torch.float32).reshape(2, 1, 3, 8),
        "states": torch.zeros(2, 1, 6),
    }
    trajectory.next_obs = {
        "visual_latent": torch.arange(48, 96, dtype=torch.float32).reshape(2, 1, 3, 8),
        "states": torch.ones(2, 1, 6),
    }
    trajectory.forward_inputs = {
        "visual_latent": trajectory.curr_obs["visual_latent"].clone(),
        "action": torch.zeros(2, 1, 3),
    }
    return trajectory


def _encode(prefix: torch.Tensor) -> dict[str, torch.Tensor]:
    pooled = prefix.sum(dim=1)
    return {
        REPLAY_ACTOR_TOKEN_KEY: pooled,
        REPLAY_CRITIC_TOKEN_1_KEY: pooled + 1,
        REPLAY_CRITIC_TOKEN_2_KEY: pooled + 2,
    }


def test_extract_replay_transition_features_prefers_compressed_embeddings():
    forward_inputs = {
        REPLAY_ACTOR_TOKEN_KEY: torch.zeros(1, 8),
        REPLAY_CRITIC_TOKEN_1_KEY: torch.ones(1, 8),
        REPLAY_CRITIC_TOKEN_2_KEY: torch.full((1, 8), 2.0),
        "ref_action": torch.ones(1, 3),
    }

    features = extract_replay_transition_features(forward_inputs)

    assert set(features) == {*EMBEDDING_KEYS, "ref_action"}
    assert "visual_latent" not in features


def test_legacy_replay_is_compressed_in_memory_without_rewriting_source(tmp_path: Path):
    source = TrajectoryReplayBuffer(
        enable_cache=True,
        cache_size=1,
        sample_window_size=1,
        auto_save=True,
        auto_save_path=str(tmp_path),
        save_executor_workers=1,
        cache_obs_keys=["visual_latent", "states"],
    )
    source.add_trajectories([_legacy_trajectory()])
    source.close(wait=True)

    trajectory_file = next(tmp_path.glob("trajectory_*.pt"))
    persisted_before = torch.load(trajectory_file, map_location="cpu")
    assert "visual_latent" in persisted_before["curr_obs"]

    loaded = TrajectoryReplayBuffer(
        enable_cache=True,
        cache_size=1,
        sample_window_size=1,
        auto_save=False,
        cache_obs_keys=[*EMBEDDING_KEYS, "states"],
    )
    loaded.set_trajectory_transform(
        lambda trajectory: compress_trajectory_replay_embeddings(
            trajectory, _encode, batch_size=1
        )
    )
    loaded.load_checkpoint(str(tmp_path))
    flat = loaded._flat_trajectory_cache.get(0)

    assert "visual_latent" not in flat["curr_obs"]
    assert "visual_latent" not in flat["next_obs"]
    assert "visual_latent" not in flat["forward_inputs"]
    assert set(EMBEDDING_KEYS).issubset(flat["curr_obs"])
    assert set(EMBEDDING_KEYS).issubset(flat["next_obs"])

    persisted_after = torch.load(trajectory_file, map_location="cpu")
    assert "visual_latent" in persisted_after["curr_obs"]


def test_new_replay_persists_only_compressed_embeddings(tmp_path: Path):
    buffer = TrajectoryReplayBuffer(
        enable_cache=True,
        cache_size=1,
        sample_window_size=1,
        auto_save=True,
        auto_save_path=str(tmp_path),
        save_executor_workers=1,
        cache_obs_keys=[*EMBEDDING_KEYS, "states"],
    )
    buffer.set_trajectory_transform(
        lambda trajectory: compress_trajectory_replay_embeddings(
            trajectory, _encode, batch_size=1
        )
    )
    trajectory = _legacy_trajectory()
    trajectory.forward_inputs.update(
        {key: torch.zeros(2, 1, 8) for key in (*EMBEDDING_KEYS, "rl_token")}
    )
    buffer.add_trajectories([trajectory])
    buffer.close(wait=True)

    trajectory_file = next(tmp_path.glob("trajectory_*.pt"))
    persisted = torch.load(trajectory_file, map_location="cpu")
    assert "visual_latent" not in persisted["curr_obs"]
    assert "visual_latent" not in persisted["next_obs"]
    assert "visual_latent" not in persisted["forward_inputs"]
    assert set(EMBEDDING_KEYS).issubset(persisted["curr_obs"])
    assert not set(EMBEDDING_KEYS).intersection(persisted["forward_inputs"])
    assert "rl_token" not in persisted["forward_inputs"]


def test_resaved_index_preserves_preloaded_trajectory_source_paths(tmp_path: Path):
    source_path = tmp_path / "source"
    run_path = tmp_path / "run"
    source = TrajectoryReplayBuffer(
        enable_cache=True,
        cache_size=2,
        sample_window_size=2,
        auto_save=True,
        auto_save_path=str(source_path),
        save_executor_workers=1,
    )
    source.add_trajectories([_legacy_trajectory()])
    source.close(wait=True)

    run = TrajectoryReplayBuffer(
        enable_cache=True,
        cache_size=2,
        sample_window_size=2,
        auto_save=True,
        auto_save_path=str(run_path),
        save_executor_workers=1,
    )
    run.set_trajectory_transform(
        lambda trajectory: compress_trajectory_replay_embeddings(
            trajectory, _encode, batch_size=1
        )
    )
    run.load_checkpoint(str(source_path))
    run.add_trajectories([_legacy_trajectory()])
    run.close(wait=True)

    assert not list(run_path.glob("trajectory_0_*.pt"))
    resumed = TrajectoryReplayBuffer(
        enable_cache=True,
        cache_size=2,
        sample_window_size=2,
        auto_save=False,
    )
    resumed.set_trajectory_transform(
        lambda trajectory: compress_trajectory_replay_embeddings(
            trajectory, _encode, batch_size=1
        )
    )
    resumed.load_checkpoint(str(run_path))

    assert len(resumed) == 2
    assert resumed._trajectory_file_path[0] == str(source_path)
    assert resumed._trajectory_file_path[1] == str(run_path)
    for trajectory_id in (0, 1):
        cached = resumed._flat_trajectory_cache.get(trajectory_id)
        assert "visual_latent" not in cached["curr_obs"]
        assert set(EMBEDDING_KEYS).issubset(cached["curr_obs"])


def test_delete_preloaded_trajectory_preserves_source_file(tmp_path: Path):
    source_path = tmp_path / "source"
    run_path = tmp_path / "run"
    source = TrajectoryReplayBuffer(
        enable_cache=True,
        cache_size=1,
        sample_window_size=1,
        auto_save=True,
        auto_save_path=str(source_path),
        save_executor_workers=1,
    )
    source.add_trajectories([_legacy_trajectory()])
    source.close(wait=True)
    source_file = next(source_path.glob("trajectory_*.pt"))

    run = TrajectoryReplayBuffer(
        enable_cache=True,
        cache_size=1,
        sample_window_size=1,
        auto_save=True,
        auto_save_path=str(run_path),
        save_executor_workers=1,
    )
    run.load_checkpoint(str(source_path))
    result = run.delete_last_trajectory()
    run.close(wait=True)

    assert result["deleted"]
    assert not result["file_deleted"]
    assert source_file.is_file()


def test_online_config_enables_compressed_cache_size_ten():
    config_path = (
        Path(__file__).parents[2]
        / "examples/embodiment/config/realworld_piper_gigacl_charge_insert_online.yaml"
    )
    config_text = config_path.read_text()

    assert 'replay_embedding_mode: "frozen_image_last_linear"' in config_text
    assert "cache_size: 10" in config_text
    assert "sample_window_size: 10" in config_text
    assert (
        "preload_checkpoint_path: /home/focal/shared_disk/users/kwj/RLinf/logs/"
        in config_text
    )
    assert "replay_buffer/rank_0" in config_text
    assert "- visual_latent" not in config_text
    for key in EMBEDDING_KEYS:
        assert f"- {key}" in config_text
