# Copyright 2026 The GIGA Authors.
#
import torch
from omegaconf import OmegaConf

from rlinf.workers.actor.fsdp_td3_policy_worker import EmbodiedTD3FSDPPolicy


class _TinyActorCritic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.actor_prefix_token_linear = torch.nn.Linear(2, 1, bias=False)
        self.critic_prefix_token_linear_1 = torch.nn.Linear(2, 1, bias=False)
        self.critic_prefix_token_linear_2 = torch.nn.Linear(2, 1, bias=False)
        self.actor_head = torch.nn.Linear(1, 1, bias=False)
        self.critic_head_1 = torch.nn.Linear(1, 1, bias=False)
        self.critic_head_2 = torch.nn.Linear(1, 1, bias=False)


def _make_policy_with_frozen_prefix_linears():
    policy = object.__new__(EmbodiedTD3FSDPPolicy)
    policy.cfg = OmegaConf.create(
        {
            "runner": {"load_optimizer_on_actor_critic_resume": False},
            "algorithm": {},
            "actor": {
                "model": {
                    "replay_embedding_mode": "frozen_image_last_linear",
                    "actor_train_prefix_token_linear": False,
                    "critic_train_prefix_token_linear": False,
                },
                "optim": {"lr": 1.0e-4},
                "critic_optim": {"lr": 1.0e-4},
            },
        }
    )
    policy.device = torch.device("cpu")
    policy.model = _TinyActorCritic()
    policy.target_model = _TinyActorCritic()
    policy.actor_optimizer = torch.optim.Adam(
        policy.model.actor_head.parameters(), lr=1.0e-4
    )
    policy.critic_optimizer = torch.optim.Adam(
        list(policy.model.critic_head_1.parameters())
        + list(policy.model.critic_head_2.parameters()),
        lr=1.0e-4,
    )
    policy.lr_scheduler = None
    policy.qf_lr_scheduler = None
    policy.update_step = 0
    policy.log_on_first_rank = lambda *args, **kwargs: None
    return policy


def test_actor_critic_resume_can_skip_optimizer_when_prefix_linear_is_frozen(tmp_path):
    source = _TinyActorCritic()
    with torch.no_grad():
        source.actor_prefix_token_linear.weight.fill_(7.0)
        source.critic_prefix_token_linear_1.weight.fill_(11.0)
        source.critic_prefix_token_linear_2.weight.fill_(13.0)

    actor_optimizer_with_prefix = torch.optim.Adam(
        list(source.actor_prefix_token_linear.parameters())
        + list(source.actor_head.parameters()),
        lr=1.0e-3,
    )
    critic_optimizer_with_prefix = torch.optim.Adam(
        list(source.critic_prefix_token_linear_1.parameters())
        + list(source.critic_prefix_token_linear_2.parameters())
        + list(source.critic_head_1.parameters())
        + list(source.critic_head_2.parameters()),
        lr=1.0e-3,
    )
    torch.save(
        {
            "format": "rlinf_td3_actor_critic_only",
            "update_step": 123,
            "model": source.state_dict(),
            "target_model": source.state_dict(),
            "actor_optimizer": actor_optimizer_with_prefix.state_dict(),
            "critic_optimizer": critic_optimizer_with_prefix.state_dict(),
            "actor_lr_scheduler": None,
            "critic_lr_scheduler": None,
        },
        tmp_path / "actor_critic.pt",
    )

    policy = _make_policy_with_frozen_prefix_linears()

    policy.load_checkpoint(str(tmp_path))

    assert torch.equal(
        policy.model.actor_prefix_token_linear.weight,
        torch.full_like(policy.model.actor_prefix_token_linear.weight, 7.0),
    )
    assert torch.equal(
        policy.model.critic_prefix_token_linear_1.weight,
        torch.full_like(policy.model.critic_prefix_token_linear_1.weight, 11.0),
    )
    assert torch.equal(
        policy.model.critic_prefix_token_linear_2.weight,
        torch.full_like(policy.model.critic_prefix_token_linear_2.weight, 13.0),
    )
    assert policy.update_step == 123


def test_actor_critic_checkpoint_saves_frozen_replay_projections(tmp_path):
    policy = _make_policy_with_frozen_prefix_linears()
    policy._compressed_replay_enabled = True

    policy._save_actor_critic_checkpoint(str(tmp_path), step=50)

    payload = torch.load(tmp_path / "actor_critic.pt", map_location="cpu")
    expected = {
        "actor_prefix_token_linear.weight",
        "critic_prefix_token_linear_1.weight",
        "critic_prefix_token_linear_2.weight",
    }
    assert expected.issubset(payload["model"])
    assert expected.issubset(payload["target_model"])


def test_compressed_replay_syncs_target_projections_from_online():
    policy = _make_policy_with_frozen_prefix_linears()
    policy._compressed_replay_enabled = True
    with torch.no_grad():
        policy.model.actor_prefix_token_linear.weight.fill_(7.0)
        policy.model.critic_prefix_token_linear_1.weight.fill_(11.0)
        policy.model.critic_prefix_token_linear_2.weight.fill_(13.0)
        policy.target_model.actor_prefix_token_linear.weight.fill_(1.0)
        policy.target_model.critic_prefix_token_linear_1.weight.fill_(2.0)
        policy.target_model.critic_prefix_token_linear_2.weight.fill_(3.0)

    policy._sync_target_replay_projections_from_online()

    for name in (
        "actor_prefix_token_linear",
        "critic_prefix_token_linear_1",
        "critic_prefix_token_linear_2",
    ):
        assert torch.equal(
            getattr(policy.target_model, name).weight,
            getattr(policy.model, name).weight,
        )
