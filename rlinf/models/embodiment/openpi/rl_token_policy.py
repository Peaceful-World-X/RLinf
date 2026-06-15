# Copyright 2025 The RLinf Authors.
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

"""OpenPi RL Token Policy: frozen PI0Pytorch backbone + trainable RLTokenAutoencoder + TD3 heads."""

from __future__ import annotations

import copy
import dataclasses
import json
from typing import Any

import torch
from torch import Tensor

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.models.embodiment.modules.value_head import ValueHead
from rlinf.models.embodiment.openpi.rl_token.rl_token import (
    RLTokenAutoencoder,
    RLTokenConfig,
    reconstruction_loss,
)


@dataclasses.dataclass
class OpenPiRLTokenConfig:
    """Config for OpenPiRLTokenPolicy.

    pi0_config: passed to PI0Pytorch (can be None for smoke tests with mocked backbone)
    """

    pi0_config: Any = None
    hidden_dim: int = 2048  # must match PI0Pytorch prefix_output last dim
    rl_token_dim: int = 256
    rl_token_encoder_layers: int = 2
    rl_token_decoder_layers: int = 2
    rl_token_num_heads: int = 8
    rl_token_max_seq_len: int = 512
    rl_token_dropout: float = 0.1
    num_image_tokens: int = (
        768  # image-only tokens passed to rl_token encoder (num_images * 256)
    )
    # "full_prefix": use all prefix tokens (image + language)
    # "image_only": use only the first num_image_tokens (image tokens)
    prefix_feature_type: str = "image_only"
    robot_state_dim: int = 14  # proprioception dimension (s_p)
    actor_hidden_dims: tuple = (512, 256)
    critic_hidden_dims: tuple = (512, 256)
    action_horizon: int = 5
    action_dim: int = 7
    recon_loss_coef: float = 0.1
    actor_output_bound: float | None = None
    use_robot_state: bool = True
    critic_use_robot_state: bool | None = None
    action_space: str = "absolute"
    action_norm_stats_path: str | None = None
    action_norm_std_floor: float = 1e-6
    critic_train_rl_token_encoder: bool = False
    critic_separate_rl_token_encoder: bool = False


class OpenPiRLTokenPolicy(torch.nn.Module, BasePolicy):
    """Frozen openpi backbone + trainable RLTokenAutoencoder + actor/critic MLP heads.

    Implements BasePolicy.td3_forward() and td3_q_forward() for TD3 training.
    The PI0Pytorch backbone is frozen; only rl_token_autoencoder, actor_head,
    and critic_head_1/2 are trained.
    """

    def __init__(self, config: OpenPiRLTokenConfig):
        torch.nn.Module.__init__(self)
        self.config = config
        self.action_space = str(getattr(config, "action_space", "absolute"))

        rl_cfg = RLTokenConfig(
            hidden_dim=config.hidden_dim,
            rl_token_dim=config.rl_token_dim,
            max_seq_len=config.rl_token_max_seq_len,
            encoder_layers=config.rl_token_encoder_layers,
            decoder_layers=config.rl_token_decoder_layers,
            num_heads=config.rl_token_num_heads,
            dropout=config.rl_token_dropout,
        )

        self.rl_token_autoencoder = RLTokenAutoencoder(rl_cfg)
        self.critic_rl_token_encoder = None
        if bool(getattr(config, "critic_separate_rl_token_encoder", False)):
            self.critic_rl_token_encoder = copy.deepcopy(
                self.rl_token_autoencoder.encoder
            )
        self._setup_action_transform()
        ref_action_dim = config.action_horizon * config.action_dim
        self.use_robot_state = bool(getattr(config, "use_robot_state", True))
        critic_use_robot_state = getattr(config, "critic_use_robot_state", None)
        if critic_use_robot_state is None:
            critic_use_robot_state = self.use_robot_state
        self.critic_use_robot_state = bool(critic_use_robot_state)
        actor_robot_state_dim = config.robot_state_dim if self.use_robot_state else 0
        critic_robot_state_dim = (
            config.robot_state_dim if self.critic_use_robot_state else 0
        )
        actor_input_dim = config.rl_token_dim + actor_robot_state_dim + ref_action_dim
        self.actor_head = ValueHead(
            input_dim=actor_input_dim,
            hidden_sizes=config.actor_hidden_dims,
            output_dim=config.action_horizon * config.action_dim,
            activation="relu",
            bias_last=True,
        )
        critic_state_dim = config.rl_token_dim + critic_robot_state_dim + ref_action_dim
        critic_input_dim = critic_state_dim + ref_action_dim  # state features + a_{1:C}
        self.critic_head_1 = ValueHead(
            input_dim=critic_input_dim,
            hidden_sizes=config.critic_hidden_dims,
            output_dim=1,
            activation="relu",
            bias_last=True,
        )
        self.critic_head_2 = ValueHead(
            input_dim=critic_input_dim,
            hidden_sizes=config.critic_hidden_dims,
            output_dim=1,
            activation="relu",
            bias_last=True,
        )

        # Target networks — worker calls soft_update_target_model to keep them in sync
        self.target_rl_token_autoencoder = copy.deepcopy(self.rl_token_autoencoder)
        self.target_critic_rl_token_encoder = (
            copy.deepcopy(self.critic_rl_token_encoder)
            if self.critic_rl_token_encoder is not None
            else None
        )
        self.target_actor_head = copy.deepcopy(self.actor_head)
        self.target_critic_head_1 = copy.deepcopy(self.critic_head_1)
        self.target_critic_head_2 = copy.deepcopy(self.critic_head_2)
        target_params = (
            list(self.target_rl_token_autoencoder.parameters())
            + list(self.target_actor_head.parameters())
            + list(self.target_critic_head_1.parameters())
            + list(self.target_critic_head_2.parameters())
        )
        if self.target_critic_rl_token_encoder is not None:
            target_params += list(self.target_critic_rl_token_encoder.parameters())
        for p in target_params:
            p.requires_grad_(False)

    def _setup_action_transform(self):
        mean = torch.zeros(self.config.action_dim, dtype=torch.float32)
        std = torch.ones(self.config.action_dim, dtype=torch.float32)
        if self.action_space == "normalized_delta":
            stats_path = getattr(self.config, "action_norm_stats_path", None)
            if stats_path is None:
                raise ValueError(
                    "action_norm_stats_path is required for normalized_delta action_space"
                )
            with open(stats_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            stats = payload.get("norm_stats", payload)["actions"]
            mean = torch.tensor(
                stats["mean"][: self.config.action_dim], dtype=torch.float32
            )
            std = torch.tensor(
                stats["std"][: self.config.action_dim], dtype=torch.float32
            )
            floor = float(getattr(self.config, "action_norm_std_floor", 1e-6))
            std = torch.where(std.abs() < floor, torch.ones_like(std), std)
        mask = torch.tensor(
            [True] * 6 + [False] + [True] * 6 + [False], dtype=torch.bool
        )
        if self.config.action_dim != 14:
            mask = torch.ones(self.config.action_dim, dtype=torch.bool)
        self.register_buffer("action_norm_mean", mean, persistent=False)
        self.register_buffer("action_norm_std", std, persistent=False)
        self.register_buffer(
            "delta_action_mask", mask[: self.config.action_dim], persistent=False
        )

    def _absolute_to_training_action(
        self, actions: Tensor, state: Tensor | None
    ) -> Tensor:
        if self.action_space != "normalized_delta":
            return actions
        if state is None:
            raise ValueError("normalized_delta action transform requires robot state")
        orig_shape = actions.shape
        actions = actions.reshape(
            actions.shape[0], self.config.action_horizon, self.config.action_dim
        )
        state = state.to(device=actions.device, dtype=actions.dtype).reshape(
            actions.shape[0], self.config.action_dim
        )
        delta = actions.clone()
        mask = self.delta_action_mask.to(device=actions.device)
        delta[..., mask] = delta[..., mask] - state[:, None, :][..., mask]
        mean = self.action_norm_mean.to(
            device=actions.device, dtype=actions.dtype
        ).reshape(1, 1, -1)
        std = self.action_norm_std.to(
            device=actions.device, dtype=actions.dtype
        ).reshape(1, 1, -1)
        normalized = (delta - mean) / (std + 1e-6)
        return normalized.reshape(orig_shape)

    def _training_action_to_absolute(
        self, actions: Tensor, state: Tensor | None
    ) -> Tensor:
        if self.action_space != "normalized_delta":
            return actions
        if state is None:
            raise ValueError("normalized_delta action transform requires robot state")
        orig_shape = actions.shape
        actions = actions.reshape(
            actions.shape[0], self.config.action_horizon, self.config.action_dim
        )
        state = state.to(device=actions.device, dtype=actions.dtype).reshape(
            actions.shape[0], self.config.action_dim
        )
        mean = self.action_norm_mean.to(
            device=actions.device, dtype=actions.dtype
        ).reshape(1, 1, -1)
        std = self.action_norm_std.to(
            device=actions.device, dtype=actions.dtype
        ).reshape(1, 1, -1)
        delta = actions * (std + 1e-6) + mean
        absolute = delta.clone()
        mask = self.delta_action_mask.to(device=actions.device)
        absolute[..., mask] = absolute[..., mask] + state[:, None, :][..., mask]
        return absolute.reshape(orig_shape)

    def _zero_init_critic_outputs(self):
        for head in (self.critic_head_1, self.critic_head_2):
            last = head.mlp[-1]
            torch.nn.init.zeros_(last.weight)
            if last.bias is not None:
                torch.nn.init.zeros_(last.bias)

    # ------------------------------------------------------------------
    # BasePolicy abstract methods
    # ------------------------------------------------------------------

    def default_forward(self, **kwargs):
        return self.predict_action_batch(**kwargs)

    def predict_action_batch(self, env_obs=None, obs=None, **kwargs):
        """Rollout inference: encode prefix → actor head → actions.

        Rollout workers call policies with ``env_obs=...``. Keep ``obs`` as a
        compatibility alias for direct tests and older call sites.
        """
        obs = env_obs if env_obs is not None else obs
        if obs is None:
            raise ValueError("predict_action_batch requires `env_obs` or `obs`.")
        prefix_output, _, _ = self._build_prefix_cache_from_obs(obs)
        image_features = self._select_prefix_features(prefix_output)
        rl_token = self.rl_token_autoencoder.encoder(image_features)
        robot_state = obs.get("states", obs.get("robot_state", None))
        if robot_state is not None and not isinstance(robot_state, torch.Tensor):
            robot_state = torch.tensor(robot_state, dtype=torch.float32)
        # Get VLA reference action ã = πvla(s) for actor conditioning and BC loss
        ref_action = (
            self._get_vla_ref_action(obs)
            if hasattr(self, "_get_vla_ref_action")
            else None
        )
        if ref_action is not None:
            ref_action = self._absolute_to_training_action(ref_action, robot_state)
        if ref_action is None:
            ref_action_dim = self.config.action_horizon * self.config.action_dim
            ref_action = torch.zeros(
                rl_token.shape[0],
                ref_action_dim,
                device=rl_token.device,
                dtype=rl_token.dtype,
            )
        x = self._build_x(rl_token, robot_state, ref_action)
        training_actions = self._decode_action(x, use_target=False)
        env_actions = self._training_action_to_absolute(training_actions, robot_state)
        flat_actions = training_actions.reshape(training_actions.shape[0], -1)
        zero_scores = training_actions.new_zeros(*training_actions.shape[:2], 1)
        result = {
            "prev_logprobs": zero_scores,
            "prev_values": zero_scores,
            "forward_inputs": {
                "action": flat_actions,
                "model_action": flat_actions,
                "env_action_absolute": env_actions.reshape(
                    env_actions.shape[0], -1
                ).cpu(),
                "visual_latent": image_features.cpu(),
                "ref_action": ref_action.reshape(ref_action.shape[0], -1).cpu(),
            },
        }
        return env_actions, result

    # ------------------------------------------------------------------
    # TD3 interface (called by TD3Algorithm via BasePolicy.forward)
    # ------------------------------------------------------------------

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        return BasePolicy.forward(self, forward_type=forward_type, **kwargs)

    def td3_forward(self, mode: str = "actor", **kwargs):
        if mode == "actor":
            return self._td3_actor_forward(**kwargs)
        elif mode == "critic":
            return self._td3_critic_forward(**kwargs)
        raise ValueError(f"Unknown mode: {mode}")

    def td3_q_forward(self, rl_state: Tensor, action: Tensor, **kwargs):
        """Direct Q-value computation used during actor update."""
        return self._compute_q(rl_state, action, use_target=False)

    # ------------------------------------------------------------------
    # Target network methods (called by TD3Algorithm on unwrapped policy)
    # ------------------------------------------------------------------

    def target_actor_forward(self, visual_feat, robot_state, ref_action, **kwargs):
        # The TD3 worker calls this method on ``self.target_model``. That module's
        # ordinary heads already contain EMA target weights, so using the internal
        # target_* heads here would apply a second, stale target path.
        prefix_output = self._extract_prefix_from_visual_feat(visual_feat)
        features = self._select_prefix_features(prefix_output)
        rl_token = self.rl_token_autoencoder.encoder(features)
        x = self._build_x(rl_token, robot_state, ref_action)
        actions = self._decode_action(x, use_target=False)
        critic_rl_token = self._encode_critic_token(features, use_target=False)
        critic_state = self._build_critic_state(
            critic_rl_token, robot_state, ref_action
        )
        return actions, {
            "rl_state": x,
            "rl_token": rl_token,
            "critic_rl_token": critic_rl_token,
            "critic_rl_state": critic_state,
        }

    def target_critic_forward(self, rl_state: Tensor, action: Tensor, **kwargs):
        rl_state = kwargs.get("critic_rl_state", rl_state)
        return self._compute_q(rl_state, action, use_target=False)

    # ------------------------------------------------------------------
    # Auxiliary loss
    # ------------------------------------------------------------------

    def compute_recon_loss(self, prefix_output: Tensor, rl_token: Tensor) -> Tensor:
        recon = self.rl_token_autoencoder.decoder(rl_token, prefix_output)
        return reconstruction_loss(prefix_output, recon)

    # ------------------------------------------------------------------
    # Backbone freezing
    # ------------------------------------------------------------------

    def freeze_backbone(self):
        """Freeze all PI0Pytorch parameters; only RL heads remain trainable."""
        trainable_modules = {
            self.rl_token_autoencoder,
            self.actor_head,
            self.critic_head_1,
            self.critic_head_2,
        }
        if (
            bool(getattr(self.config, "critic_train_rl_token_encoder", False))
            and self.critic_rl_token_encoder is not None
        ):
            trainable_modules.add(self.critic_rl_token_encoder)
        for name, param in self.named_parameters():
            # Check if param belongs to any trainable module
            is_trainable = any(
                any(p is param for p in m.parameters()) for m in trainable_modules
            )
            if not is_trainable:
                param.requires_grad_(False)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _td3_actor_forward(
        self,
        visual_feat,
        robot_state: Tensor,
        ref_action: Tensor,
        ref_action_dropout_p: float = 0.0,
        use_target: bool = False,
        compute_recon_loss: bool = False,
        **kwargs,
    ):
        prefix_output = self._extract_prefix_from_visual_feat(visual_feat)
        features = self._select_prefix_features(prefix_output)
        rl_token = self._encode_actor_token(features, use_target=use_target)
        x = self._build_x(rl_token, robot_state, ref_action)
        actions = self._decode_action(x, use_target=use_target)
        critic_rl_token = self._encode_critic_token(features, use_target=use_target)
        critic_state = self._build_critic_state(
            critic_rl_token, robot_state, ref_action
        )
        aux = {
            "rl_state": x,
            "critic_rl_state": critic_state,
            "rl_token": rl_token,
            "critic_rl_token": critic_rl_token,
            "prefix_output": features,
        }
        if compute_recon_loss:
            aux["recon_loss"] = self.compute_recon_loss(features, rl_token)
        return actions, aux

    def _td3_critic_forward(
        self,
        rl_state: Tensor,
        action: Tensor,
        use_target: bool = False,
        **kwargs,
    ):
        critic_state = kwargs.get("critic_rl_state", rl_state)
        return self._compute_q(critic_state, action, use_target=use_target)

    def _decode_action(self, x: Tensor, use_target: bool) -> Tensor:
        head = self.target_actor_head if use_target else self.actor_head
        flat = head(x)
        if self.config.actor_output_bound is not None:
            bound = float(self.config.actor_output_bound)
            flat = bound * torch.tanh(flat / bound)
        return flat.reshape(
            flat.shape[0], self.config.action_horizon, self.config.action_dim
        )

    def _compute_q(self, rl_state: Tensor, action: Tensor, use_target: bool):
        if action.dim() == 3:
            action = action.reshape(action.shape[0], -1)
        critic_input = torch.cat([rl_state, action], dim=-1)
        if use_target:
            q1 = self.target_critic_head_1(critic_input)
            q2 = self.target_critic_head_2(critic_input)
        else:
            q1 = self.critic_head_1(critic_input)
            q2 = self.critic_head_2(critic_input)
        return q1, q2

    def _build_x(
        self, rl_token: Tensor, robot_state: Tensor | None, ref_action: Tensor | None
    ) -> Tensor:
        parts = [rl_token]
        if self.use_robot_state and robot_state is not None:
            parts.append(
                robot_state.to(device=rl_token.device, dtype=rl_token.dtype).reshape(
                    rl_token.shape[0], -1
                )
            )
        if ref_action is not None:
            parts.append(
                ref_action.to(device=rl_token.device, dtype=rl_token.dtype).reshape(
                    rl_token.shape[0], -1
                )
            )
        return torch.cat(parts, dim=-1)

    def _build_critic_state(
        self, rl_token: Tensor, robot_state: Tensor | None, ref_action: Tensor | None
    ) -> Tensor:
        parts = [rl_token]
        if self.critic_use_robot_state and robot_state is not None:
            parts.append(
                robot_state.to(device=rl_token.device, dtype=rl_token.dtype).reshape(
                    rl_token.shape[0], -1
                )
            )
        if ref_action is not None:
            parts.append(
                ref_action.to(device=rl_token.device, dtype=rl_token.dtype).reshape(
                    rl_token.shape[0], -1
                )
            )
        return torch.cat(parts, dim=-1)

    def _select_prefix_features(self, prefix_output: Tensor) -> Tensor:
        if self.config.prefix_feature_type == "image_only":
            return prefix_output[:, : self.config.num_image_tokens, :]
        return prefix_output  # "full_prefix"

    def _encode_actor_token(self, features: Tensor, use_target: bool = False) -> Tensor:
        encoder = (
            self.target_rl_token_autoencoder.encoder
            if use_target
            else self.rl_token_autoencoder.encoder
        )
        return encoder(features)

    def _encode_critic_token(
        self, features: Tensor, use_target: bool = False
    ) -> Tensor:
        if self.critic_rl_token_encoder is None:
            return self._encode_actor_token(features, use_target=use_target)
        if use_target:
            encoder = self.target_critic_rl_token_encoder
            if encoder is None:
                encoder = self.critic_rl_token_encoder
        else:
            encoder = self.critic_rl_token_encoder
        return encoder(features)

    def _extract_prefix_from_visual_feat(self, visual_feat) -> Tensor:
        """Extract prefix tokens from visual_feat.

        In production, visual_feat is the raw observation dict and
        _build_prefix_cache is called here. For smoke tests, visual_feat
        can be a pre-computed tensor of shape (B, L, hidden_dim).
        """
        if isinstance(visual_feat, Tensor):
            return visual_feat
        # Production path: call PI0Pytorch._build_prefix_cache
        return self._build_prefix_cache_from_obs(visual_feat)[0]

    def _build_prefix_cache_from_obs(self, obs):
        """Delegate to PI0Pytorch._build_prefix_cache. Subclasses that inherit
        PI0Pytorch will have this method available."""
        raise NotImplementedError(
            "_build_prefix_cache_from_obs must be implemented by the concrete "
            "subclass that also inherits PI0Pytorch, or mocked in tests."
        )
