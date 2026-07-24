# Copyright 2026 The GIGA Authors.
#
"""Smoke test for OpenPiRLTokenPolicy — no GPU, no environment, no real PI0 weights."""

import copy

import torch

from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.openpi.rl_token_policy import (
    REPLAY_ACTOR_TOKEN_KEY,
    REPLAY_CRITIC_TOKEN_1_KEY,
    REPLAY_CRITIC_TOKEN_2_KEY,
    OpenPiRLTokenConfig,
    OpenPiRLTokenPolicy,
)

B = 2
L = 32  # prefix token sequence length (mocked)
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
        env_action_dim=ROBOT_STATE_DIM,
        robot_state_dim=ROBOT_STATE_DIM,
        controlled_action_indices=tuple(range(ROBOT_STATE_DIM)),
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


def _make_direct_token_policy(source, *, train_actor=False, train_critic=False):
    cfg = OpenPiRLTokenConfig(
        hidden_dim=HIDDEN_DIM,
        rl_token_dim=HIDDEN_DIM,
        rl_token_encoder_layers=1,
        rl_token_decoder_layers=1,
        rl_token_num_heads=4,
        rl_token_max_seq_len=64,
        num_image_tokens=4,
        prefix_feature_type="image_only",
        rl_token_source=source,
        actor_train_prefix_token_linear=train_actor,
        critic_train_prefix_token_linear=train_critic,
        actor_hidden_dims=(32,),
        critic_hidden_dims=(32,),
        action_horizon=ACTION_HORIZON,
        action_dim=ACTION_DIM,
        env_action_dim=ROBOT_STATE_DIM,
        robot_state_dim=ROBOT_STATE_DIM,
        controlled_action_indices=tuple(range(ROBOT_STATE_DIM)),
    )
    return OpenPiRLTokenPolicy(cfg)


def test_last_token_source_uses_full_prefix_last_token():
    policy = _make_direct_token_policy("last_token")
    prefix = torch.randn(B, L, HIDDEN_DIM)
    token_features = policy._select_token_features(prefix)

    actor_token = policy._encode_actor_token(token_features)
    critic_token = policy._encode_critic_tokens(token_features)

    assert torch.equal(actor_token, prefix[:, -1, :])
    assert torch.equal(critic_token, prefix[:, -1, :])


def test_image_last_linear_initializes_to_last_token():
    policy = _make_direct_token_policy("image_last_linear")
    prefix = torch.randn(B, L, HIDDEN_DIM)
    token_features = policy._select_token_features(prefix)

    assert policy.actor_prefix_token_linear.bias is None
    assert policy.critic_prefix_token_linear_1.bias is None
    assert policy.critic_prefix_token_linear_2.bias is None
    actor_weight = policy.actor_prefix_token_linear.weight.detach()
    critic_weight_1 = policy.critic_prefix_token_linear_1.weight.detach()
    critic_weight_2 = policy.critic_prefix_token_linear_2.weight.detach()
    expected = torch.zeros_like(actor_weight)
    expected[0, -1] = 1.0

    assert policy.actor_prefix_token_linear is not policy.critic_prefix_token_linear_1
    assert (
        policy.critic_prefix_token_linear_1 is not policy.critic_prefix_token_linear_2
    )
    assert torch.equal(actor_weight, expected)
    assert torch.equal(critic_weight_1, expected)
    assert torch.equal(critic_weight_2, expected)
    assert torch.equal(policy._encode_actor_token(token_features), prefix[:, -1, :])
    critic_token_1, critic_token_2 = policy._encode_critic_tokens(token_features)
    assert torch.equal(critic_token_1, prefix[:, -1, :])
    assert torch.equal(critic_token_2, prefix[:, -1, :])

    actions, aux = policy.td3_forward(
        mode="actor",
        visual_feat=prefix,
        robot_state=torch.randn(B, ROBOT_STATE_DIM),
        ref_action=torch.randn(B, ACTION_HORIZON, ACTION_DIM),
    )
    assert isinstance(aux["critic_rl_state"], tuple)
    assert len(aux["critic_rl_state"]) == 2
    q1, q2 = policy.td3_forward(
        mode="critic",
        rl_state=aux["rl_state"],
        critic_rl_state=aux["critic_rl_state"],
        action=actions,
    )
    assert q1.shape == (B, 1)
    assert q2.shape == (B, 1)


def test_precomputed_replay_embeddings_match_full_prefix_td3_forward():
    policy = _make_direct_token_policy("image_last_linear")
    policy.eval()
    with torch.no_grad():
        policy.actor_prefix_token_linear.weight.normal_()
        policy.critic_prefix_token_linear_1.weight.normal_()
        policy.critic_prefix_token_linear_2.weight.normal_()

    prefix = torch.randn(B, L, HIDDEN_DIM)
    robot_state = torch.randn(B, ROBOT_STATE_DIM)
    ref_action = torch.randn(B, ACTION_HORIZON, ACTION_DIM)

    with torch.no_grad():
        full_actions, full_aux = policy.td3_forward(
            mode="actor",
            visual_feat=prefix,
            robot_state=robot_state,
            ref_action=ref_action,
        )
        embeddings = policy.encode_replay_embeddings(prefix)
        cached_actions, cached_aux = policy.td3_forward(
            mode="actor",
            visual_feat=embeddings,
            robot_state=robot_state,
            ref_action=ref_action,
        )

        full_q = policy.td3_forward(
            mode="critic",
            rl_state=full_aux["critic_rl_state"],
            action=full_actions,
        )
        cached_q = policy.td3_forward(
            mode="critic",
            rl_state=cached_aux["critic_rl_state"],
            action=cached_actions,
        )

    assert set(embeddings) == {
        REPLAY_ACTOR_TOKEN_KEY,
        REPLAY_CRITIC_TOKEN_1_KEY,
        REPLAY_CRITIC_TOKEN_2_KEY,
    }
    assert torch.allclose(cached_actions, full_actions)
    assert torch.allclose(cached_aux["rl_token"], full_aux["rl_token"])
    assert isinstance(cached_aux["critic_rl_state"], tuple)
    for cached_state, full_state in zip(
        cached_aux["critic_rl_state"], full_aux["critic_rl_state"], strict=True
    ):
        assert torch.allclose(cached_state, full_state)
    for cached_value, full_value in zip(cached_q, full_q, strict=True):
        assert torch.allclose(cached_value, full_value)


def test_precomputed_replay_embeddings_match_target_td3_forward():
    online_policy = _make_direct_token_policy("image_last_linear")
    target_policy = copy.deepcopy(online_policy)
    online_policy.eval()
    target_policy.eval()
    prefix = torch.randn(B, L, HIDDEN_DIM)
    robot_state = torch.randn(B, ROBOT_STATE_DIM)
    ref_action = torch.randn(B, ACTION_HORIZON, ACTION_DIM)

    with torch.no_grad():
        replay_embeddings = online_policy.encode_replay_embeddings(prefix)
        full_actions, full_aux = target_policy.target_actor_forward(
            visual_feat=prefix,
            robot_state=robot_state,
            ref_action=ref_action,
        )
        cached_actions, cached_aux = target_policy.target_actor_forward(
            visual_feat=replay_embeddings,
            robot_state=robot_state,
            ref_action=ref_action,
        )

    assert torch.allclose(cached_actions, full_actions)
    for cached_state, full_state in zip(
        cached_aux["critic_rl_state"], full_aux["critic_rl_state"], strict=True
    ):
        assert torch.allclose(cached_state, full_state)


def test_compressed_replay_rollout_omits_full_prefix():
    policy = _make_direct_token_policy("image_last_linear")
    policy.eval()
    policy.replay_embedding_mode = "frozen_image_last_linear"
    prefix = torch.randn(B, L, HIDDEN_DIM)
    policy._build_prefix_cache_from_obs = lambda obs: (prefix, None, None)

    _, result = policy.predict_action_batch(
        env_obs={"states": torch.randn(B, ROBOT_STATE_DIM)}
    )

    forward_inputs = result["forward_inputs"]
    assert "visual_latent" not in forward_inputs
    assert {
        REPLAY_ACTOR_TOKEN_KEY,
        REPLAY_CRITIC_TOKEN_1_KEY,
        REPLAY_CRITIC_TOKEN_2_KEY,
    }.issubset(forward_inputs)


def test_prefix_token_linear_train_flags_are_respected():
    frozen = _make_direct_token_policy("image_last_linear")
    frozen.freeze_backbone()
    assert not frozen.actor_prefix_token_linear.weight.requires_grad
    assert not frozen.critic_prefix_token_linear_1.weight.requires_grad
    assert not frozen.critic_prefix_token_linear_2.weight.requires_grad

    trainable = _make_direct_token_policy(
        "image_last_linear", train_actor=True, train_critic=True
    )
    trainable.freeze_backbone()
    assert trainable.actor_prefix_token_linear.weight.requires_grad
    assert trainable.critic_prefix_token_linear_1.weight.requires_grad
    assert trainable.critic_prefix_token_linear_2.weight.requires_grad


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
    test_last_token_source_uses_full_prefix_last_token()
    test_image_last_linear_initializes_to_last_token()
    test_prefix_token_linear_train_flags_are_respected()
    test_base_policy_forward_dispatch()
    print("All smoke tests passed.")
