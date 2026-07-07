# Copyright 2026 The GIGA Authors.
#
"""Tests for real-time (no-cache) prefix-token inference in offline OpenRLT."""

from pathlib import Path

import torch

from examples.embodiment.train_openrlt_right_arm_offline import (
    RealtimeFeatureProvider,
    TrainConfig,
    _collate_env_obs,
    load_dataset,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def make_cfg(tmp_path: Path, **overrides) -> TrainConfig:
    cfg = TrainConfig(
        data_dir=str(tmp_path / "data"),
        feature_cache=str(tmp_path / "features.pt"),
        norm_stats_path=str(tmp_path / "norm_stats.json"),
        output_dir=str(tmp_path / "out"),
        urdf_path=str(REPO_ROOT / "assets/piper_local_assets/piper.urdf"),
        rl_token_source="image_last_linear",
        prefix_feature_type="full_prefix",
        num_image_tokens=2,
        actor_train_prefix_token_linear=True,
        critic_train_prefix_token_linear=True,
        realtime_prefix_features=True,
        z_dim=4,
        proprio_dim=6,
        chunk_len=2,
        action_dim=6,
        hidden_dim=8,
        num_layers=1,
        tcp_radius=0.0,
        warm_up_chunks=0,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


class FakeVLA:
    """Returns a prefix whose every token equals the row's first state value.

    Lets tests verify that a dataset row maps to the correct observation.
    """

    def __init__(self, num_image_tokens: int, z_dim: int):
        self.num_image_tokens = int(num_image_tokens)
        self.z_dim = int(z_dim)

    def _build_prefix_cache_from_obs(self, obs):
        states = obs["states"]
        bsz = states.shape[0]
        seq_len = self.num_image_tokens + 1
        base = states[:, 0].reshape(bsz, 1, 1).float()
        prefix = base.expand(bsz, seq_len, self.z_dim).clone()
        return prefix, None, None


def write_traj(path: Path, tag: int, num_chunks: int = 3) -> None:
    # states[rel, 0, 0] encodes (tag, rel) so provider outputs are identifiable.
    states = torch.zeros(num_chunks, 1, 14)
    for rel in range(num_chunks):
        states[rel, 0, 0] = float(100 * tag + rel)
    actions = torch.zeros(num_chunks, 1, 12)
    traj = {
        "curr_obs": {"states": states},
        "next_obs": {"states": states + 0.1},
        "actions": actions,
        "rewards": torch.zeros(num_chunks, 1),
        "dones": torch.zeros(num_chunks, 1, dtype=torch.bool),
        "forward_inputs": {"executed_action": actions, "ref_action": actions},
    }
    torch.save(traj, path)


def test_collate_env_obs_batches_tensors_and_flattens_lists():
    obs_a = {"states": torch.zeros(1, 14), "task_descriptions": ["a"]}
    obs_b = {"states": torch.ones(1, 14), "task_descriptions": ["b"]}

    collated = _collate_env_obs([obs_a, obs_b])

    assert collated["states"].shape == (2, 14)
    assert torch.equal(collated["states"][0], torch.zeros(14))
    assert torch.equal(collated["states"][1], torch.ones(14))
    assert collated["task_descriptions"] == ["a", "b"]


def test_realtime_provider_maps_rows_to_observations(tmp_path: Path):
    cfg = make_cfg(tmp_path, feature_cache_batch_size=2)
    path = str(tmp_path / "trajectory_0_test.pt")
    num_chunks = 3
    states = torch.zeros(num_chunks, 1, 14)
    for rel in range(num_chunks):
        states[rel, 0, 0] = float(10 + rel)
    trajectories = {path: {"curr_obs": {"states": states}}}
    rows = [{"path": path, "rel": rel} for rel in range(num_chunks)]
    provider = RealtimeFeatureProvider(
        FakeVLA(cfg.num_image_tokens, cfg.z_dim),
        rows,
        trajectories,
        cfg,
        torch.device("cpu"),
    )

    assert provider.shape == torch.Size([3, cfg.num_image_tokens + 1, cfg.z_dim])

    out = provider[torch.tensor([0, 2])]
    assert out.shape == (2, cfg.num_image_tokens + 1, cfg.z_dim)
    assert torch.allclose(out[0], torch.full_like(out[0], 10.0))
    assert torch.allclose(out[1], torch.full_like(out[1], 12.0))

    # List indexing and single-row indexing hit the same mapping.
    assert torch.allclose(provider[[1]][0], torch.full_like(out[0], 11.0))


def test_load_dataset_realtime_builds_rows_without_cache(tmp_path: Path):
    cfg = make_cfg(tmp_path)
    data_dir = Path(cfg.data_dir)
    data_dir.mkdir(parents=True)
    write_traj(data_dir / "trajectory_0_test.pt", tag=0)
    write_traj(data_dir / "trajectory_1_test.pt", tag=1)

    model = FakeVLA(cfg.num_image_tokens, cfg.z_dim)
    data = load_dataset(cfg, feature_model=model, device=torch.device("cpu"))

    provider = data["z"]
    assert isinstance(provider, RealtimeFeatureProvider)
    # 2 trajectories x 3 chunks, warm_up_chunks=0, tcp_radius=0 -> all kept.
    assert provider.shape == torch.Size([6, cfg.num_image_tokens + 1, cfg.z_dim])
    assert not Path(cfg.feature_cache).exists()  # no cache written or read

    # Every batch index resolves to a real per-row feature.
    batch = provider[torch.arange(provider.shape[0])]
    assert batch.shape == (6, cfg.num_image_tokens + 1, cfg.z_dim)
    # Row values equal the encoded (tag, rel) state; each row is internally constant.
    for i in range(batch.shape[0]):
        assert torch.allclose(batch[i], torch.full_like(batch[i], batch[i, 0, 0]))
