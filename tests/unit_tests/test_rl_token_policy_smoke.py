"""Smoke test for OpenPiRLTokenPolicy — no GPU, no environment, no real PI0 weights."""

from unittest.mock import patch

import torch

from rlinf.models.embodiment.openpi.rl_token_policy import (
    OpenPiRLTokenConfig,
    OpenPiRLTokenPolicy,
)
from rlinf.models.embodiment.base_policy import ForwardType


B = 2
L = 32          # prefix token sequence length (mocked)
HIDDEN_DIM = 64  # small for speed; production uses 2048
RL_TOKEN_DIM = 16
ACTION_HORIZON = 3
ACTION_DIM = 7
ROBOT_STATE_DIM = 8


def _make_policy():
    cfg = OpenPiRLTokenConfig(
        hidden_dim=HIDDEN_DIM,
        rl_token_dim=RL_TOKEN_DIM,
        rl_token_encoder_layers=1,
        rl_token_decoder_layers=1,
        rl_token_num_heads=4,
        rl_token_max_seq_len=64,
        actor_hidden_dims=(32,),
        critic_hidden_dims=(32,),
        action_horizon=ACTION_HORIZON,
        action_dim=ACTION_DIM,
        robot_state_dim=ROBOT_STATE_DIM,
    )
    return OpenPiRLTokenPolicy(cfg)


def _fake_prefix():
    return torch.randn(B, L, HIDDEN_DIM)


def _expected_rl_state_dim():
    return RL_TOKEN_DIM + ROBOT_STATE_DIM + ACTION_HORIZON * ACTION_DIM


def test_actor_forward():
    policy = _make_policy()
    policy.eval()
    fake_prefix = _fake_prefix()

    actions, aux = policy.td3_forward(
        mode="actor",
        visual_feat=fake_prefix,
        robot_state=torch.randn(B, ROBOT_STATE_DIM),
        ref_action=torch.randn(B, ACTION_HORIZON, ACTION_DIM),
    )
    assert actions.shape == (B, ACTION_HORIZON, ACTION_DIM), actions.shape
    assert aux["rl_state"].shape == (B, _expected_rl_state_dim()), aux["rl_state"].shape
    assert aux["rl_token"].shape == (B, RL_TOKEN_DIM), aux["rl_token"].shape
    assert aux["prefix_output"].shape == (B, L, HIDDEN_DIM), aux["prefix_output"].shape


def test_critic_forward():
    policy = _make_policy()
    policy.eval()
    rl_state = torch.randn(B, _expected_rl_state_dim())
    action = torch.randn(B, ACTION_HORIZON, ACTION_DIM)

    q1, q2 = policy.td3_forward(mode="critic", rl_state=rl_state, action=action)
    assert q1.shape == (B, 1), q1.shape
    assert q2.shape == (B, 1), q2.shape


def test_target_actor_forward():
    policy = _make_policy()
    policy.eval()
    actions, aux = policy.target_actor_forward(
        visual_feat=_fake_prefix(),
        robot_state=torch.randn(B, ROBOT_STATE_DIM),
        ref_action=torch.randn(B, ACTION_HORIZON, ACTION_DIM),
    )
    assert actions.shape == (B, ACTION_HORIZON, ACTION_DIM)
    assert aux["rl_state"].shape == (B, _expected_rl_state_dim())


def test_target_critic_forward():
    policy = _make_policy()
    policy.eval()
    tq1, tq2 = policy.target_critic_forward(
        rl_state=torch.randn(B, _expected_rl_state_dim()),
        action=torch.randn(B, ACTION_HORIZON, ACTION_DIM),
    )
    assert tq1.shape == (B, 1)
    assert tq2.shape == (B, 1)


def test_recon_loss():
    policy = _make_policy()
    policy.eval()
    prefix = _fake_prefix()
    rl_token = policy.rl_token_autoencoder.encoder(prefix)
    loss = policy.compute_recon_loss(prefix, rl_token)
    assert loss.shape == (), loss.shape
    assert loss.item() >= 0.0


def test_actor_aux_rl_token_is_used_for_recon_loss():
    policy = _make_policy()
    policy.eval()
    prefix = _fake_prefix()

    _, aux = policy.td3_forward(
        mode="actor",
        visual_feat=prefix,
        robot_state=torch.randn(B, ROBOT_STATE_DIM),
        ref_action=torch.randn(B, ACTION_HORIZON, ACTION_DIM),
    )

    loss = policy.compute_recon_loss(aux["prefix_output"], aux["rl_token"])
    assert loss.shape == (), loss.shape
    assert aux["rl_state"].shape[-1] != aux["rl_token"].shape[-1]


def test_actor_forward_can_compute_recon_loss_inside_forward():
    policy = _make_policy()
    policy.eval()
    prefix = _fake_prefix()

    _, aux = policy.td3_forward(
        mode="actor",
        visual_feat=prefix,
        robot_state=torch.randn(B, ROBOT_STATE_DIM),
        ref_action=torch.randn(B, ACTION_HORIZON, ACTION_DIM),
        compute_recon_loss=True,
    )

    assert aux["recon_loss"].shape == (), aux["recon_loss"].shape
    assert aux["recon_loss"].item() >= 0.0


def test_base_policy_forward_dispatch():
    """Verify BasePolicy.forward routes TD3/TD3_Q correctly."""
    policy = _make_policy()
    policy.eval()
    fake_prefix = _fake_prefix()

    actions, aux = policy.forward(
        forward_type=ForwardType.TD3,
        mode="actor",
        visual_feat=fake_prefix,
        robot_state=torch.randn(B, ROBOT_STATE_DIM),
        ref_action=torch.randn(B, ACTION_HORIZON, ACTION_DIM),
    )
    assert actions.shape == (B, ACTION_HORIZON, ACTION_DIM)

    q1, q2 = policy.forward(
        forward_type=ForwardType.TD3_Q,
        rl_state=aux["rl_state"],
        action=actions,
    )
    assert q1.shape == (B, 1)


if __name__ == "__main__":
    test_actor_forward()
    test_critic_forward()
    test_target_actor_forward()
    test_target_critic_forward()
    test_recon_loss()
    test_actor_aux_rl_token_is_used_for_recon_loss()
    test_actor_forward_can_compute_recon_loss_inside_forward()
    test_base_policy_forward_dispatch()
    print("All smoke tests passed.")
