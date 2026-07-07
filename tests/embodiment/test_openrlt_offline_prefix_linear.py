# Copyright 2026 The GIGA Authors.
#
from pathlib import Path

import torch

from examples.embodiment.train_openrlt_right_arm_offline import (
    CachedFeatureProvider,
    ChunkActor,
    OfflinePrefixTokenEncoder,
    TrainConfig,
    TwinCritic,
    build_actor_optimizer,
    build_critic_optimizer,
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


def write_traj(path: Path, num_chunks: int = 3) -> None:
    states = torch.zeros(num_chunks, 1, 14)
    actions = torch.zeros(num_chunks, 1, 12)
    traj = {
        "curr_obs": {"states": states},
        "next_obs": {"states": states + 0.1},
        "actions": actions,
        "rewards": torch.zeros(num_chunks, 1),
        "dones": torch.zeros(num_chunks, 1, dtype=torch.bool),
        "forward_inputs": {
            "executed_action": actions,
            "ref_action": actions,
        },
    }
    torch.save(traj, path)


def test_load_dataset_reads_prefix_tokens_and_next_prefix_tokens(tmp_path: Path):
    cfg = make_cfg(tmp_path)
    data_dir = Path(cfg.data_dir)
    data_dir.mkdir(parents=True)
    traj_path = data_dir / "trajectory_0_test.pt"
    write_traj(traj_path)
    prefix_tokens = torch.arange(3 * 3 * 4, dtype=torch.float32).reshape(3, 3, 4)
    torch.save(
        {
            "prefix_tokens": prefix_tokens,
            "metas": [
                {
                    "path": str(traj_path),
                    "rel_chunk": rel,
                    "traj_index": 0,
                    "success": False,
                }
                for rel in range(3)
            ],
        },
        cfg.feature_cache,
    )

    data = load_dataset(cfg)

    assert isinstance(data["z"], CachedFeatureProvider)
    assert data["z"].shape == (3, 3, 4)
    assert torch.equal(data["z"][torch.arange(3)], prefix_tokens)
    next_z = data["z"][data["next_idx"]]
    assert torch.equal(next_z[0], prefix_tokens[1])
    assert torch.equal(next_z[-1], prefix_tokens[-1])


def test_cached_feature_provider_indexes_via_feature_idx_mapping():
    z_all = torch.arange(5 * 3 * 2, dtype=torch.float32).reshape(5, 3, 2)
    # Row 0 -> cache row 2, row 1 -> cache row 0, row 2 -> cache row 2 (duplicate).
    feature_idx = torch.tensor([2, 0, 2], dtype=torch.long)
    provider = CachedFeatureProvider(z_all, feature_idx)

    assert provider.shape == torch.Size([3, 3, 2])
    assert torch.equal(provider[torch.tensor([0, 1])], z_all[torch.tensor([2, 0])])
    assert torch.equal(provider[[2]], z_all[2:3])
    assert torch.equal(provider[1], z_all[0:1])


def test_actor_optimizer_includes_actor_prefix_linear_when_enabled(tmp_path: Path):
    cfg = make_cfg(tmp_path)
    actor_prefix = OfflinePrefixTokenEncoder(cfg, role="actor")
    actor = ChunkActor(cfg, prefix_encoder=actor_prefix)

    opt = build_actor_optimizer(actor, cfg)
    opt_params = {id(param) for group in opt.param_groups for param in group["params"]}

    assert id(actor_prefix.actor_prefix_token_linear.weight) in opt_params


def test_critic_optimizer_includes_critic_prefix_linears_when_enabled(tmp_path: Path):
    cfg = make_cfg(tmp_path)
    critic_prefix = OfflinePrefixTokenEncoder(cfg, role="critic")
    critic = TwinCritic(cfg, prefix_encoder=critic_prefix)

    opt = build_critic_optimizer(critic, cfg)
    opt_params = {id(param) for group in opt.param_groups for param in group["params"]}

    assert id(critic_prefix.critic_prefix_token_linear_1.weight) in opt_params
    assert id(critic_prefix.critic_prefix_token_linear_2.weight) in opt_params


def test_actor_updates_prefix_linear_from_prefix_token_batch(tmp_path: Path):
    cfg = make_cfg(tmp_path, actor_residual_ref=False)
    actor_prefix = OfflinePrefixTokenEncoder(cfg, role="actor")
    actor = ChunkActor(cfg, prefix_encoder=actor_prefix)
    opt = build_actor_optimizer(actor, cfg)
    before = actor_prefix.actor_prefix_token_linear.weight.detach().clone()

    prefix_tokens = torch.randn(5, 3, 4)
    state = torch.randn(5, 6)
    ref = torch.randn(5, 2, 6)
    target = torch.randn(5, 2, 6)
    loss = torch.nn.functional.mse_loss(actor.mean(prefix_tokens, state, ref), target)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()

    assert not torch.equal(actor_prefix.actor_prefix_token_linear.weight, before)
