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
    actor_head_type: str = "mlp"
    actor_hidden_dims: tuple = (512, 256)
    actor_transformer_dim: int = 256
    actor_transformer_layers: int = 2
    actor_transformer_heads: int = 8
    actor_transformer_ffn_dim: int = 1024
    actor_transformer_dropout: float = 0.1
    critic_hidden_dims: tuple = (512, 256)
    critic_head_type: str = "mlp"
    openrlt_z_proj_dim: int = 256
    openrlt_state_proj_dim: int = 64
    openrlt_action_proj_dim: int = 256
    openrlt_hidden_dim: int = 256
    openrlt_num_layers: int = 2
    action_horizon: int = 5
    action_dim: int = 7
    env_action_dim: int = 7
    controlled_action_indices: tuple | list | None = None
    recon_loss_coef: float = 0.1
    actor_output_bound: float | None = None
    use_robot_state: bool = True
    critic_use_robot_state: bool | None = None
    critic_use_ref_action: bool = True
    critic_use_rl_token: bool = True
    actor_ref_action_dropout_p: float = 0.0
    actor_ref_action_mask_flag: bool = False
    # Optional residual actor branch. When enabled, the OpenRLT actor head returns
    # ref_action + residual in the configured training action space. Keep this
    # off for direct-action checkpoints.
    actor_residual_ref: bool = False
    actor_residual_scale: float = 1.0
    critic_block_norm: bool = False
    critic_action_encoder_dim: int = 0
    action_space: str = "absolute"
    action_norm_stats_path: str | None = None
    action_norm_std_floor: float = 1e-6
    critic_train_rl_token_encoder: bool = False
    critic_separate_rl_token_encoder: bool = False


class TransformerActionHead(torch.nn.Module):
    """Action-chunk actor with attention over RL token, state, and ref-action steps."""

    def __init__(
        self,
        *,
        rl_token_dim: int,
        robot_state_dim: int,
        use_robot_state: bool,
        action_horizon: int,
        action_dim: int,
        model_dim: int,
        num_layers: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.rl_token_dim = int(rl_token_dim)
        self.robot_state_dim = int(robot_state_dim)
        self.use_robot_state = bool(use_robot_state)
        self.action_horizon = int(action_horizon)
        self.action_dim = int(action_dim)
        self.model_dim = int(model_dim)

        self.rl_token_proj = torch.nn.Linear(self.rl_token_dim, self.model_dim)
        self.state_proj = (
            torch.nn.Linear(self.robot_state_dim, self.model_dim)
            if self.use_robot_state
            else None
        )
        self.action_proj = torch.nn.Linear(self.action_dim, self.model_dim)
        self.type_embedding = torch.nn.Embedding(3, self.model_dim)
        self.action_pos_embedding = torch.nn.Parameter(
            torch.zeros(1, self.action_horizon, self.model_dim)
        )
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=int(num_heads),
            dim_feedforward=int(ffn_dim),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = torch.nn.TransformerEncoder(
            encoder_layer, num_layers=int(num_layers)
        )
        self.norm = torch.nn.LayerNorm(self.model_dim)
        self.action_out = torch.nn.Linear(self.model_dim, self.action_dim)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
        torch.nn.init.normal_(self.type_embedding.weight, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.action_pos_embedding, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.action_out.weight, mean=0.0, std=0.02)
        if self.action_out.bias is not None:
            torch.nn.init.zeros_(self.action_out.bias)

    def forward(self, x):
        batch_size = x.shape[0]
        offset = 0
        rl_token = x[:, offset : offset + self.rl_token_dim]
        offset += self.rl_token_dim

        tokens = [
            self.rl_token_proj(rl_token)
            + self.type_embedding.weight[0].to(device=x.device, dtype=x.dtype)
        ]

        if self.use_robot_state:
            robot_state = x[:, offset : offset + self.robot_state_dim]
            offset += self.robot_state_dim
            tokens.append(
                self.state_proj(robot_state)
                + self.type_embedding.weight[1].to(device=x.device, dtype=x.dtype)
            )

        ref_action = x[:, offset:].reshape(
            batch_size, self.action_horizon, self.action_dim
        )
        action_tokens = (
            self.action_proj(ref_action)
            + self.action_pos_embedding.to(device=x.device, dtype=x.dtype)
            + self.type_embedding.weight[2].to(device=x.device, dtype=x.dtype)
        )
        seq = torch.cat([t.unsqueeze(1) for t in tokens] + [action_tokens], dim=1)
        encoded = self.norm(self.encoder(seq))
        action_encoded = encoded[:, -self.action_horizon :, :]
        return self.action_out(action_encoded).reshape(batch_size, -1)


def _layer_norm_no_params(x: Tensor, eps: float = 1e-6) -> Tensor:
    mean = x.mean(dim=-1, keepdim=True)
    var = torch.square(x - mean).mean(dim=-1, keepdim=True)
    return (x - mean) / torch.sqrt(var + eps)


class OpenRLTMLP(torch.nn.Module):
    """Small MLP matching the openpi-RLT actor/critic trunk style."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        output_dim: int,
    ):
        super().__init__()
        dims = (
            [int(input_dim)] + [int(hidden_dim)] * int(num_layers) + [int(output_dim)]
        )
        layers: list[torch.nn.Module] = []
        for idx in range(len(dims) - 2):
            layers.append(torch.nn.Linear(dims[idx], dims[idx + 1]))
            layers.append(torch.nn.LayerNorm(dims[idx + 1]))
            layers.append(torch.nn.GELU())
        layers.append(torch.nn.Linear(dims[-2], dims[-1]))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class OpenRLTActorHead(torch.nn.Module):
    """openpi-RLT style actor: project RLT, proprio, and reference chunk separately."""

    def __init__(
        self,
        *,
        rl_token_dim: int,
        robot_state_dim: int,
        use_robot_state: bool,
        action_horizon: int,
        action_dim: int,
        z_proj_dim: int,
        state_proj_dim: int,
        action_proj_dim: int,
        hidden_dim: int,
        num_layers: int,
        has_ref_mask: bool = False,
        residual_ref: bool = False,
        residual_scale: float = 1.0,
    ):
        super().__init__()
        self.rl_token_dim = int(rl_token_dim)
        self.robot_state_dim = int(robot_state_dim)
        self.use_robot_state = bool(use_robot_state)
        self.action_horizon = int(action_horizon)
        self.action_dim = int(action_dim)
        self.ref_dim = self.action_horizon * self.action_dim
        self.has_ref_mask = bool(has_ref_mask)
        self.residual_ref = bool(residual_ref)
        self.residual_scale = float(residual_scale)
        self.z_proj = torch.nn.Linear(self.rl_token_dim, int(z_proj_dim))
        self.state_proj = (
            torch.nn.Linear(self.robot_state_dim, int(state_proj_dim))
            if self.use_robot_state
            else None
        )
        self.ref_proj = torch.nn.Linear(self.ref_dim, int(action_proj_dim))
        trunk_dim = int(z_proj_dim) + int(action_proj_dim)
        if self.use_robot_state:
            trunk_dim += int(state_proj_dim)
        self.trunk = OpenRLTMLP(
            input_dim=trunk_dim,
            hidden_dim=int(hidden_dim),
            num_layers=int(num_layers),
            output_dim=self.ref_dim,
        )
        if self.residual_ref:
            last = self.trunk.net[-1]
            if isinstance(last, torch.nn.Linear):
                torch.nn.init.zeros_(last.weight)
                torch.nn.init.zeros_(last.bias)

    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.shape[0]
        offset = 0
        rl_token = x[:, offset : offset + self.rl_token_dim]
        offset += self.rl_token_dim
        parts = [_layer_norm_no_params(self.z_proj(rl_token))]
        if self.use_robot_state:
            robot_state = x[:, offset : offset + self.robot_state_dim]
            offset += self.robot_state_dim
            parts.append(
                torch.tanh(_layer_norm_no_params(self.state_proj(robot_state)))
            )
        ref_flat = x[:, offset : offset + self.ref_dim].reshape(
            batch_size, self.ref_dim
        )
        parts.append(torch.tanh(_layer_norm_no_params(self.ref_proj(ref_flat))))
        residual_or_action = self.trunk(torch.cat(parts, dim=-1))
        if self.residual_ref:
            return ref_flat + self.residual_scale * residual_or_action
        return residual_or_action


class OpenRLTQHead(torch.nn.Module):
    """openpi-RLT style Q head.

    The critic intentionally receives the candidate action chunk rather than a
    reference action. Configs should normally set critic_use_ref_action=False.
    """

    def __init__(
        self,
        *,
        rl_token_dim: int,
        robot_state_dim: int,
        critic_use_rl_token: bool,
        critic_use_robot_state: bool,
        critic_use_ref_action: bool,
        action_horizon: int,
        action_dim: int,
        z_proj_dim: int,
        state_proj_dim: int,
        action_proj_dim: int,
        hidden_dim: int,
        num_layers: int,
    ):
        super().__init__()
        self.rl_token_dim = int(rl_token_dim)
        self.robot_state_dim = int(robot_state_dim)
        self.critic_use_rl_token = bool(critic_use_rl_token)
        self.critic_use_robot_state = bool(critic_use_robot_state)
        self.critic_use_ref_action = bool(critic_use_ref_action)
        self.action_horizon = int(action_horizon)
        self.action_dim = int(action_dim)
        self.action_flat_dim = self.action_horizon * self.action_dim
        trunk_dim = int(action_proj_dim)
        self.z_proj = (
            torch.nn.Linear(self.rl_token_dim, int(z_proj_dim))
            if self.critic_use_rl_token
            else None
        )
        if self.critic_use_rl_token:
            trunk_dim += int(z_proj_dim)
        self.state_proj = (
            torch.nn.Linear(self.robot_state_dim, int(state_proj_dim))
            if self.critic_use_robot_state
            else None
        )
        if self.critic_use_robot_state:
            trunk_dim += int(state_proj_dim)
        self.ref_proj = (
            torch.nn.Linear(self.action_flat_dim, int(action_proj_dim))
            if self.critic_use_ref_action
            else None
        )
        if self.critic_use_ref_action:
            trunk_dim += int(action_proj_dim)
        self.action_proj = torch.nn.Linear(self.action_flat_dim, int(action_proj_dim))
        self.trunk = OpenRLTMLP(
            input_dim=trunk_dim,
            hidden_dim=int(hidden_dim),
            num_layers=int(num_layers),
            output_dim=1,
        )

    def forward(self, critic_state: Tensor, action: Tensor) -> Tensor:
        batch_size = critic_state.shape[0]
        offset = 0
        parts = []
        if self.critic_use_rl_token:
            rl_token = critic_state[:, offset : offset + self.rl_token_dim]
            offset += self.rl_token_dim
            parts.append(_layer_norm_no_params(self.z_proj(rl_token)))
        if self.critic_use_robot_state:
            robot_state = critic_state[:, offset : offset + self.robot_state_dim]
            offset += self.robot_state_dim
            parts.append(
                torch.tanh(_layer_norm_no_params(self.state_proj(robot_state)))
            )
        if self.critic_use_ref_action:
            ref_flat = critic_state[:, offset : offset + self.action_flat_dim]
            offset += self.action_flat_dim
            parts.append(torch.tanh(_layer_norm_no_params(self.ref_proj(ref_flat))))
        if action.dim() == 3:
            action = action.reshape(batch_size, -1)
        parts.append(torch.tanh(_layer_norm_no_params(self.action_proj(action))))
        return self.trunk(torch.cat(parts, dim=-1))


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
        self.critic_use_ref_action = bool(
            getattr(config, "critic_use_ref_action", True)
        )
        self.critic_use_rl_token = bool(getattr(config, "critic_use_rl_token", True))
        self.actor_ref_action_mask_flag = bool(
            getattr(config, "actor_ref_action_mask_flag", False)
        )
        self.critic_block_norm = bool(getattr(config, "critic_block_norm", False))
        self.critic_action_encoder_dim = int(
            getattr(config, "critic_action_encoder_dim", 0)
        )
        actor_robot_state_dim = config.robot_state_dim if self.use_robot_state else 0
        critic_robot_state_dim = (
            config.robot_state_dim if self.critic_use_robot_state else 0
        )
        critic_ref_action_dim = ref_action_dim if self.critic_use_ref_action else 0
        actor_ref_mask_dim = 1 if self.actor_ref_action_mask_flag else 0
        actor_input_dim = (
            config.rl_token_dim
            + actor_robot_state_dim
            + ref_action_dim
            + actor_ref_mask_dim
        )
        actor_head_type = str(getattr(config, "actor_head_type", "mlp")).lower()
        if actor_head_type == "mlp":
            self.actor_head = ValueHead(
                input_dim=actor_input_dim,
                hidden_sizes=config.actor_hidden_dims,
                output_dim=config.action_horizon * config.action_dim,
                activation="relu",
                bias_last=True,
            )
        elif actor_head_type == "transformer":
            self.actor_head = TransformerActionHead(
                rl_token_dim=config.rl_token_dim,
                robot_state_dim=config.robot_state_dim,
                use_robot_state=self.use_robot_state,
                action_horizon=config.action_horizon,
                action_dim=config.action_dim,
                model_dim=config.actor_transformer_dim,
                num_layers=config.actor_transformer_layers,
                num_heads=config.actor_transformer_heads,
                ffn_dim=config.actor_transformer_ffn_dim,
                dropout=config.actor_transformer_dropout,
            )
        elif actor_head_type in {"openrlt", "openrlt_mlp"}:
            self.actor_head = OpenRLTActorHead(
                rl_token_dim=config.rl_token_dim,
                robot_state_dim=config.robot_state_dim,
                use_robot_state=self.use_robot_state,
                action_horizon=config.action_horizon,
                action_dim=config.action_dim,
                z_proj_dim=config.openrlt_z_proj_dim,
                state_proj_dim=config.openrlt_state_proj_dim,
                action_proj_dim=config.openrlt_action_proj_dim,
                hidden_dim=config.openrlt_hidden_dim,
                num_layers=config.openrlt_num_layers,
                has_ref_mask=self.actor_ref_action_mask_flag,
                residual_ref=bool(getattr(config, "actor_residual_ref", False)),
                residual_scale=float(getattr(config, "actor_residual_scale", 1.0)),
            )
        else:
            raise ValueError(f"Unsupported actor_head_type: {actor_head_type}")
        self.critic_head_type = str(getattr(config, "critic_head_type", "mlp")).lower()
        critic_rl_token_dim = config.rl_token_dim if self.critic_use_rl_token else 0
        critic_state_dim = (
            critic_rl_token_dim + critic_robot_state_dim + critic_ref_action_dim
        )
        if critic_state_dim <= 0:
            raise ValueError(
                "critic state must include at least one of rl_token, robot_state, or ref_action"
            )
        if self.critic_head_type in {"openrlt", "openrlt_mlp"}:
            self.critic_state_norm = None
            self.critic_action_norm = None
            self.critic_action_encoder = None
            q_kwargs = {
                "rl_token_dim": config.rl_token_dim,
                "robot_state_dim": config.robot_state_dim,
                "critic_use_rl_token": self.critic_use_rl_token,
                "critic_use_robot_state": self.critic_use_robot_state,
                "critic_use_ref_action": self.critic_use_ref_action,
                "action_horizon": config.action_horizon,
                "action_dim": config.action_dim,
                "z_proj_dim": config.openrlt_z_proj_dim,
                "state_proj_dim": config.openrlt_state_proj_dim,
                "action_proj_dim": config.openrlt_action_proj_dim,
                "hidden_dim": config.openrlt_hidden_dim,
                "num_layers": config.openrlt_num_layers,
            }
            self.critic_head_1 = OpenRLTQHead(**q_kwargs)
            self.critic_head_2 = OpenRLTQHead(**q_kwargs)
        else:
            critic_action_input_dim = (
                self.critic_action_encoder_dim
                if self.critic_action_encoder_dim > 0
                else ref_action_dim
            )
            critic_input_dim = critic_state_dim + critic_action_input_dim
            if self.critic_block_norm:
                self.critic_state_norm = torch.nn.LayerNorm(critic_state_dim)
                self.critic_action_norm = torch.nn.LayerNorm(ref_action_dim)
            else:
                self.critic_state_norm = None
                self.critic_action_norm = None
            self.critic_action_encoder = (
                torch.nn.Sequential(
                    torch.nn.Linear(ref_action_dim, self.critic_action_encoder_dim),
                    torch.nn.ReLU(),
                    torch.nn.LayerNorm(self.critic_action_encoder_dim),
                )
                if self.critic_action_encoder_dim > 0
                else None
            )
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

    def _controlled_action_indices(self) -> tuple[int, ...]:
        indices = getattr(self.config, "controlled_action_indices", None)
        if indices is None:
            return tuple(range(int(self.config.action_dim)))
        return tuple(int(i) for i in indices)

    def _controlled_action_index_tensor(self, device) -> Tensor:
        return torch.tensor(
            self._controlled_action_indices(), dtype=torch.long, device=device
        )

    def _full_delta_mask(self, device) -> Tensor:
        env_action_dim = int(
            getattr(self.config, "env_action_dim", self.config.action_dim)
        )
        if env_action_dim == 14:
            return torch.tensor(
                [True] * 6 + [False] + [True] * 6 + [False],
                dtype=torch.bool,
                device=device,
            )
        return torch.ones(env_action_dim, dtype=torch.bool, device=device)

    def _controlled_delta_mask(self, device) -> Tensor:
        return self._full_delta_mask(device).index_select(
            0, self._controlled_action_index_tensor(device)
        )

    def _select_controlled_state(self, state: Tensor, device, dtype) -> Tensor:
        state = state.to(device=device, dtype=dtype).reshape(state.shape[0], -1)
        if state.shape[-1] == int(self.config.action_dim):
            return state
        index = self._controlled_action_index_tensor(device)
        if state.shape[-1] <= int(index.max().item()):
            raise ValueError(
                f"robot state dim {state.shape[-1]} is too small for controlled indices "
                f"{self._controlled_action_indices()}"
            )
        return state.index_select(-1, index)

    def _setup_action_transform(self):
        action_dim = int(self.config.action_dim)
        env_action_dim = int(getattr(self.config, "env_action_dim", action_dim))
        mean = torch.zeros(action_dim, dtype=torch.float32)
        std = torch.ones(action_dim, dtype=torch.float32)
        if self.action_space == "normalized_delta":
            stats_path = getattr(self.config, "action_norm_stats_path", None)
            if stats_path is None:
                raise ValueError(
                    "action_norm_stats_path is required for normalized_delta action_space"
                )
            with open(stats_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            stats = payload.get("norm_stats", payload)["actions"]
            if (
                env_action_dim != action_dim
                and getattr(self.config, "controlled_action_indices", None) is not None
            ):
                indices = list(self._controlled_action_indices())
                mean = torch.tensor(
                    [stats["mean"][i] for i in indices], dtype=torch.float32
                )
                std = torch.tensor(
                    [stats["std"][i] for i in indices], dtype=torch.float32
                )
            else:
                mean = torch.tensor(stats["mean"][:action_dim], dtype=torch.float32)
                std = torch.tensor(stats["std"][:action_dim], dtype=torch.float32)
            floor = float(getattr(self.config, "action_norm_std_floor", 1e-6))
            std = torch.where(std.abs() < floor, torch.ones_like(std), std)
        self.register_buffer("action_norm_mean", mean, persistent=False)
        self.register_buffer("action_norm_std", std, persistent=False)
        self.register_buffer(
            "delta_action_mask",
            self._controlled_delta_mask(torch.device("cpu")),
            persistent=False,
        )

    def _absolute_to_training_action(
        self, actions: Tensor, state: Tensor | None
    ) -> Tensor:
        if self.action_space != "normalized_delta":
            if getattr(self.config, "controlled_action_indices", None) is None:
                return actions
            horizon = int(self.config.action_horizon)
            action_dim = int(self.config.action_dim)
            env_action_dim = int(getattr(self.config, "env_action_dim", action_dim))
            actions_3d = actions.reshape(actions.shape[0], horizon, env_action_dim)
            index = self._controlled_action_index_tensor(actions_3d.device)
            return actions_3d.index_select(-1, index).reshape(actions.shape[0], -1)
        if state is None:
            raise ValueError("normalized_delta action transform requires robot state")
        horizon = int(self.config.action_horizon)
        action_dim = int(self.config.action_dim)
        env_action_dim = int(getattr(self.config, "env_action_dim", action_dim))
        orig_rank = actions.dim()
        if actions.shape[-1] == action_dim:
            controlled_actions = actions.reshape(actions.shape[0], horizon, action_dim)
            controlled_state = self._select_controlled_state(
                state, controlled_actions.device, controlled_actions.dtype
            )
        else:
            full_actions = actions.reshape(actions.shape[0], horizon, env_action_dim)
            full_state = state.to(
                device=full_actions.device, dtype=full_actions.dtype
            ).reshape(full_actions.shape[0], env_action_dim)
            index = self._controlled_action_index_tensor(full_actions.device)
            controlled_actions = full_actions.index_select(-1, index)
            controlled_state = full_state.index_select(-1, index)
        delta = controlled_actions.clone()
        mask = self.delta_action_mask.to(device=delta.device)
        delta[..., mask] = delta[..., mask] - controlled_state[:, None, :][..., mask]
        mean = self.action_norm_mean.to(device=delta.device, dtype=delta.dtype).reshape(
            1, 1, -1
        )
        std = self.action_norm_std.to(device=delta.device, dtype=delta.dtype).reshape(
            1, 1, -1
        )
        normalized = (delta - mean) / (std + 1e-6)
        if orig_rank == 3 and actions.shape[-1] == action_dim:
            return normalized
        return normalized.reshape(actions.shape[0], -1)

    def _training_action_to_absolute(
        self, actions: Tensor, state: Tensor | None
    ) -> Tensor:
        if self.action_space != "normalized_delta":
            return actions
        if state is None:
            raise ValueError("normalized_delta action transform requires robot state")
        horizon = int(self.config.action_horizon)
        action_dim = int(self.config.action_dim)
        env_action_dim = int(getattr(self.config, "env_action_dim", action_dim))
        training = actions.reshape(actions.shape[0], horizon, action_dim)
        mean = self.action_norm_mean.to(
            device=training.device, dtype=training.dtype
        ).reshape(1, 1, -1)
        std = self.action_norm_std.to(
            device=training.device, dtype=training.dtype
        ).reshape(1, 1, -1)
        controlled_delta = training * (std + 1e-6) + mean
        controlled_abs = controlled_delta.clone()
        controlled_state = self._select_controlled_state(
            state, training.device, training.dtype
        )
        mask = self.delta_action_mask.to(device=training.device)
        controlled_abs[..., mask] = (
            controlled_abs[..., mask] + controlled_state[:, None, :][..., mask]
        )
        if (
            env_action_dim == action_dim
            and getattr(self.config, "controlled_action_indices", None) is None
        ):
            return controlled_abs
        full_state = state.to(device=training.device, dtype=training.dtype).reshape(
            training.shape[0], env_action_dim
        )
        full = full_state[:, None, :].expand(-1, horizon, -1).clone()
        index = self._controlled_action_index_tensor(training.device)
        full.scatter_(
            -1,
            index.reshape(1, 1, -1).expand(training.shape[0], horizon, -1),
            controlled_abs,
        )
        return full

    def _zero_init_critic_outputs(self):
        for head in (self.critic_head_1, self.critic_head_2):
            if hasattr(head, "mlp"):
                last = head.mlp[-1]
            elif hasattr(head, "trunk") and hasattr(head.trunk, "net"):
                last = head.trunk.net[-1]
            else:
                continue
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
        ref_env_action = (
            self._get_vla_ref_action(obs)
            if hasattr(self, "_get_vla_ref_action")
            else None
        )
        ref_action = None
        if ref_env_action is not None:
            ref_action = self._absolute_to_training_action(ref_env_action, robot_state)
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
        if (
            ref_env_action is not None
            and getattr(self.config, "controlled_action_indices", None) is not None
        ):
            ref_env_3d = ref_env_action.reshape(
                ref_env_action.shape[0],
                self.config.action_horizon,
                int(getattr(self.config, "env_action_dim", self.config.action_dim)),
            ).to(device=env_actions.device, dtype=env_actions.dtype)
            index = self._controlled_action_index_tensor(env_actions.device)
            env_actions = ref_env_3d.clone()
            env_actions.scatter_(
                -1,
                index.reshape(1, 1, -1).expand(
                    training_actions.shape[0], self.config.action_horizon, -1
                ),
                self._training_action_to_absolute(
                    training_actions, robot_state
                ).index_select(-1, index),
            )
        flat_actions = training_actions.reshape(training_actions.shape[0], -1)
        ref_flat = ref_action.reshape(ref_action.shape[0], -1).to(
            device=flat_actions.device, dtype=flat_actions.dtype
        )
        actor_ref_mse = torch.mean(
            (flat_actions.detach() - ref_flat.detach()) ** 2,
            dim=1,
            keepdim=True,
        )
        rollout_source = torch.ones(
            flat_actions.shape[0], 1, device=flat_actions.device, dtype=torch.long
        )
        zero_scores = training_actions.new_zeros(*training_actions.shape[:2], 1)
        forward_inputs = {
            "action": flat_actions,
            "model_action": flat_actions,
            "actor_action": flat_actions,
            "env_action_absolute": env_actions.reshape(env_actions.shape[0], -1).cpu(),
            "visual_latent": image_features.cpu(),
            "ref_action": ref_flat.cpu(),
            "actor_ref_mse": actor_ref_mse.cpu(),
            "rollout_control_source": rollout_source.cpu(),
        }
        if ref_env_action is not None:
            forward_inputs["ref_env_action_absolute"] = ref_env_action.reshape(
                ref_env_action.shape[0], -1
            ).cpu()
        result = {
            "prev_logprobs": zero_scores,
            "prev_values": zero_scores,
            "forward_inputs": forward_inputs,
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

    def target_actor_forward(
        self,
        visual_feat,
        robot_state,
        ref_action,
        ref_action_dropout_p: float = 0.0,
        **kwargs,
    ):
        # The TD3 worker calls this method on ``self.target_model``. That module's
        # ordinary heads already contain EMA target weights, so using the internal
        # target_* heads here would apply a second, stale target path.
        prefix_output = self._extract_prefix_from_visual_feat(visual_feat)
        features = self._select_prefix_features(prefix_output)
        rl_token = self.rl_token_autoencoder.encoder(features)
        ref_action_for_actor, ref_action_mask = self._maybe_mask_ref_action(
            ref_action, float(ref_action_dropout_p)
        )
        x = self._build_x(rl_token, robot_state, ref_action_for_actor, ref_action_mask)
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
            "actor_ref_action_mask": ref_action_mask,
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
        ref_action_for_actor, ref_action_mask = self._maybe_mask_ref_action(
            ref_action, float(ref_action_dropout_p)
        )
        x = self._build_x(rl_token, robot_state, ref_action_for_actor, ref_action_mask)
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
            "actor_ref_action_mask": ref_action_mask,
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
        residual_ref = bool(getattr(head, "residual_ref", False))
        if self.config.actor_output_bound is not None and not residual_ref:
            bound = float(self.config.actor_output_bound)
            flat = bound * torch.tanh(flat / bound)
        return flat.reshape(
            flat.shape[0], self.config.action_horizon, self.config.action_dim
        )

    def _compute_q(self, rl_state: Tensor, action: Tensor, use_target: bool):
        if action.dim() == 3:
            action = action.reshape(action.shape[0], -1)
        if self.critic_head_type in {"openrlt", "openrlt_mlp"}:
            if use_target:
                q1 = self.target_critic_head_1(rl_state, action)
                q2 = self.target_critic_head_2(rl_state, action)
            else:
                q1 = self.critic_head_1(rl_state, action)
                q2 = self.critic_head_2(rl_state, action)
            return q1, q2
        if self.critic_state_norm is not None:
            rl_state = self.critic_state_norm(rl_state)
        if self.critic_action_norm is not None:
            action = self.critic_action_norm(action)
        if self.critic_action_encoder is not None:
            action = self.critic_action_encoder(action)
        critic_input = torch.cat([rl_state, action], dim=-1)
        if use_target:
            q1 = self.target_critic_head_1(critic_input)
            q2 = self.target_critic_head_2(critic_input)
        else:
            q1 = self.critic_head_1(critic_input)
            q2 = self.critic_head_2(critic_input)
        return q1, q2

    def _build_x(
        self,
        rl_token: Tensor,
        robot_state: Tensor | None,
        ref_action: Tensor | None,
        ref_action_mask: Tensor | None = None,
    ) -> Tensor:
        parts = [rl_token]
        if self.use_robot_state and robot_state is not None:
            parts.append(
                self._select_controlled_state(
                    robot_state, rl_token.device, rl_token.dtype
                ).reshape(rl_token.shape[0], -1)
            )
        if ref_action is not None:
            parts.append(
                ref_action.to(device=rl_token.device, dtype=rl_token.dtype).reshape(
                    rl_token.shape[0], -1
                )
            )
        if self.actor_ref_action_mask_flag:
            if ref_action_mask is None:
                ref_action_mask = torch.ones(
                    rl_token.shape[0],
                    1,
                    device=rl_token.device,
                    dtype=rl_token.dtype,
                )
            parts.append(
                ref_action_mask.to(
                    device=rl_token.device, dtype=rl_token.dtype
                ).reshape(rl_token.shape[0], -1)
            )
        return torch.cat(parts, dim=-1)

    def _build_critic_state(
        self, rl_token: Tensor, robot_state: Tensor | None, ref_action: Tensor | None
    ) -> Tensor:
        parts = []
        if self.critic_use_rl_token:
            parts.append(rl_token)
        if self.critic_use_robot_state and robot_state is not None:
            parts.append(
                self._select_controlled_state(
                    robot_state, rl_token.device, rl_token.dtype
                ).reshape(rl_token.shape[0], -1)
            )
        if self.critic_use_ref_action and ref_action is not None:
            parts.append(
                ref_action.to(device=rl_token.device, dtype=rl_token.dtype).reshape(
                    rl_token.shape[0], -1
                )
            )
        return torch.cat(parts, dim=-1)

    def _maybe_mask_ref_action(self, ref_action: Tensor, dropout_p: float):
        if ref_action is None:
            return ref_action, None
        batch_size = ref_action.shape[0]
        keep_mask = torch.ones(
            batch_size, 1, device=ref_action.device, dtype=ref_action.dtype
        )
        if self.training and dropout_p > 0.0:
            keep_mask = (
                torch.rand(batch_size, 1, device=ref_action.device) >= dropout_p
            ).to(dtype=ref_action.dtype)
            view_shape = [batch_size] + [1] * (ref_action.dim() - 1)
            ref_action = ref_action * keep_mask.reshape(view_shape)
        return ref_action, keep_mask

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
