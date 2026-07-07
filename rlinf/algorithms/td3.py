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

import torch
import torch.nn.functional as F


class TD3Algorithm:
    """TD3 algorithm logic, decoupled from worker infrastructure.

    Handles critic loss, actor loss composition, and target network updates.
    The worker is responsible for gradient accumulation, FSDP context, and
    replay buffer management.
    """

    def __init__(self, algorithm_cfg, policy_head_cfg=None):
        self.critic_actor_ratio = int(algorithm_cfg.get("critic_actor_ratio", 2))
        self.target_update_freq = int(algorithm_cfg.get("target_update_freq", 1))
        self.target_tau = float(algorithm_cfg.get("tau", 0.005))
        self.target_policy_noise = float(algorithm_cfg.get("target_policy_noise", 0.2))
        self.target_noise_clip = float(algorithm_cfg.get("target_noise_clip", 0.5))
        action_clip = algorithm_cfg.get("target_action_clip", None)
        self.target_action_clip = None if action_clip is None else float(action_clip)
        self.bc_coef = float(algorithm_cfg.get("bc_coef", 1.0))
        self.actor_q_coef = float(algorithm_cfg.get("actor_q_coef", 1.0))
        self.actor_ref_action_dropout_p = float(
            algorithm_cfg.get("actor_ref_action_dropout_p", 0.0)
        )
        self.base_target_policy_noise = self.target_policy_noise
        self.base_target_noise_clip = self.target_noise_clip
        self.base_target_action_clip = self.target_action_clip
        self.base_bc_coef = self.bc_coef
        self.base_actor_q_coef = self.actor_q_coef
        self.critic_td_coef = float(algorithm_cfg.get("critic_td_coef", 1.0))
        self.critic_mc_return_coef = float(
            algorithm_cfg.get("critic_mc_return_coef", 0.0)
        )
        self.critic_zero_return_coef = float(
            algorithm_cfg.get("critic_zero_return_coef", 0.0)
        )
        self.critic_positive_return_coef = float(
            algorithm_cfg.get("critic_positive_return_coef", 0.0)
        )
        self.critic_return_margin_coef = float(
            algorithm_cfg.get("critic_return_margin_coef", 0.0)
        )
        self.critic_return_margin = float(
            algorithm_cfg.get("critic_return_margin", 0.25)
        )
        self.critic_success_bce_coef = float(
            algorithm_cfg.get("critic_success_bce_coef", 0.0)
        )
        self.critic_success_bce_threshold = float(
            algorithm_cfg.get("critic_success_bce_threshold", 0.25)
        )
        self.critic_success_bce_temperature = float(
            algorithm_cfg.get("critic_success_bce_temperature", 0.10)
        )
        action_diag_cfg = algorithm_cfg.get("critic_action_diagnostics", None) or {}
        self.critic_action_diagnostics_enabled = bool(
            action_diag_cfg.get("enabled", False)
        )
        self.action_diag_noisy_std = float(action_diag_cfg.get("noisy_std", 0.10))
        self.action_diag_random_low = float(action_diag_cfg.get("random_low", -1.0))
        self.action_diag_random_high = float(action_diag_cfg.get("random_high", 1.0))

        contrast_cfg = algorithm_cfg.get("critic_action_contrastive", None) or {}
        self.critic_action_contrastive_coef = float(contrast_cfg.get("coef", 0.0))
        self.critic_action_contrastive_margin = float(contrast_cfg.get("margin", 0.05))
        self.critic_action_contrastive_noisy_std = float(
            contrast_cfg.get("noisy_std", self.action_diag_noisy_std)
        )
        self.critic_action_contrastive_random_low = float(
            contrast_cfg.get("random_low", self.action_diag_random_low)
        )
        self.critic_action_contrastive_random_high = float(
            contrast_cfg.get("random_high", self.action_diag_random_high)
        )
        self.critic_action_contrastive_use_noisy = bool(
            contrast_cfg.get("use_noisy", True)
        )
        self.critic_action_contrastive_use_shuffle = bool(
            contrast_cfg.get("use_shuffle", True)
        )
        self.critic_action_contrastive_use_return_mismatch = bool(
            contrast_cfg.get("use_return_mismatch", False)
        )
        self.critic_action_contrastive_use_nearest_return_mismatch = bool(
            contrast_cfg.get("use_nearest_return_mismatch", False)
        )
        self.critic_action_contrastive_use_random = bool(
            contrast_cfg.get("use_random", False)
        )
        self.critic_action_contrastive_positive_only = bool(
            contrast_cfg.get("positive_only", True)
        )

        cql_cfg = algorithm_cfg.get("critic_cql", None) or {}
        self.critic_cql_coef = float(cql_cfg.get("coef", 0.0))
        self.critic_cql_temperature = float(cql_cfg.get("temperature", 1.0))
        self.critic_cql_num_random = int(cql_cfg.get("num_random", 2))
        self.critic_cql_random_low = float(
            cql_cfg.get("random_low", self.action_diag_random_low)
        )
        self.critic_cql_random_high = float(
            cql_cfg.get("random_high", self.action_diag_random_high)
        )
        self.critic_cql_include_shuffle = bool(cql_cfg.get("include_shuffle", True))

        quality_cfg = algorithm_cfg.get("critic_quality", None) or {}
        self.critic_quality_coef = float(quality_cfg.get("coef", 0.0))
        self.critic_quality_jerk_coef = float(quality_cfg.get("jerk_coef", 0.02))
        self.critic_quality_range_coef = float(quality_cfg.get("range_coef", 0.02))

        action_quality_cfg = (
            algorithm_cfg.get("critic_action_quality_regression", None) or {}
        )
        self.critic_action_quality_coef = float(action_quality_cfg.get("coef", 0.0))
        self.critic_action_quality_penalty_scale = float(
            action_quality_cfg.get("penalty_scale", 1.0)
        )
        self.critic_action_quality_mse_ref = float(
            action_quality_cfg.get("mse_ref", 0.01)
        )
        self.critic_action_quality_use_noisy = bool(
            action_quality_cfg.get("use_noisy", True)
        )
        self.critic_action_quality_use_random = bool(
            action_quality_cfg.get("use_random", True)
        )
        self.critic_action_quality_use_nearest = bool(
            action_quality_cfg.get("use_nearest_return_mismatch", False)
        )
        self.critic_action_quality_positive_only = bool(
            action_quality_cfg.get("positive_only", True)
        )

        local_margin_cfg = algorithm_cfg.get("critic_local_action_margin", None) or {}
        self.critic_local_action_margin_coef = float(local_margin_cfg.get("coef", 0.0))
        self.critic_local_action_margin_base = float(local_margin_cfg.get("base", 0.0))
        self.critic_local_action_margin_scale = float(
            local_margin_cfg.get("scale", 0.05)
        )
        self.critic_local_action_margin_mse_ref = float(
            local_margin_cfg.get("mse_ref", 0.0004)
        )
        self.critic_local_action_margin_noisy_std = float(
            local_margin_cfg.get("noisy_std", self.action_diag_noisy_std)
        )
        self.critic_local_action_margin_num_noisy = int(
            local_margin_cfg.get("num_noisy", 1)
        )
        self.critic_local_action_margin_use_nearest = bool(
            local_margin_cfg.get("use_nearest_return_mismatch", False)
        )
        self.critic_local_action_margin_nearest_key = str(
            local_margin_cfg.get("nearest_key", "rl_state")
        ).lower()
        self.critic_local_action_margin_positive_only = bool(
            local_margin_cfg.get("positive_only", False)
        )

        pairwise_cfg = algorithm_cfg.get("critic_pairwise_stage_margin", None) or {}
        self.critic_pairwise_stage_margin_coef = float(pairwise_cfg.get("coef", 0.0))
        self.critic_pairwise_stage_margin = float(pairwise_cfg.get("margin", 0.05))
        self.critic_pairwise_action_swap_coef = float(
            pairwise_cfg.get("action_swap_coef", 0.0)
        )
        self.critic_pairwise_action_swap_margin = float(
            pairwise_cfg.get("action_swap_margin", self.critic_pairwise_stage_margin)
        )

        grad_cfg = algorithm_cfg.get("critic_action_grad", None) or {}
        self.critic_action_grad_coef = float(grad_cfg.get("coef", 0.0))
        self.critic_action_grad_target = float(grad_cfg.get("target", 0.01))
        self.critic_action_grad_max_norm = float(grad_cfg.get("max_norm", 10.0))

        self.critic_train_representation = bool(
            algorithm_cfg.get(
                "critic_train_representation",
                algorithm_cfg.get("critic_train_rl_token_encoder", False),
            )
        )
        self.critic_target_action_source = str(
            algorithm_cfg.get("critic_target_action_source", "actor")
        ).lower()
        self.base_critic_target_action_source = self.critic_target_action_source
        self.n_step = int(algorithm_cfg.get("n_step", 1))
        self.use_n_step_target = bool(algorithm_cfg.get("use_n_step_target", True))

        stage_cfg = algorithm_cfg.get("training_stages", None) or {}
        self.coupling_start_step = int(
            stage_cfg.get("actor_critic_coupling_start_step", -1)
        )
        self.coupled_critic_target_action_source = str(
            stage_cfg.get("coupled_critic_target_action_source", "actor")
        ).lower()
        self.coupled_actor_q_coef = float(
            stage_cfg.get("coupled_actor_q_coef", max(self.base_actor_q_coef, 0.1))
        )
        self.coupled_bc_coef = float(
            stage_cfg.get("coupled_bc_coef", self.base_bc_coef)
        )
        self.coupled_target_policy_noise = float(
            stage_cfg.get(
                "coupled_target_policy_noise", max(self.base_target_policy_noise, 0.05)
            )
        )
        self.coupled_target_noise_clip = float(
            stage_cfg.get(
                "coupled_target_noise_clip", max(self.base_target_noise_clip, 0.1)
            )
        )
        coupled_action_clip = stage_cfg.get(
            "coupled_target_action_clip", self.base_target_action_clip
        )
        self.coupled_target_action_clip = (
            None if coupled_action_clip is None else float(coupled_action_clip)
        )
        self.current_stage_name = "bootstrap"

        bc_guard_cfg = algorithm_cfg.get("actor_bc_guard", None) or {}
        self.actor_bc_guard_mode = str(bc_guard_cfg.get("mode", "none")).lower()
        self.actor_bc_guard_threshold = float(bc_guard_cfg.get("threshold", 0.004))
        self.actor_bc_weighted_coef = float(bc_guard_cfg.get("weighted_bc_coef", 50.0))
        self.actor_bc_penalty_coef = float(
            bc_guard_cfg.get("hard_penalty_coef", 5000.0)
        )
        self.actor_bc_penalty_power = float(bc_guard_cfg.get("hard_penalty_power", 1.0))

    # ------------------------------------------------------------------
    # Critic
    # ------------------------------------------------------------------

    @staticmethod
    def _get_visual_input(obs):
        return obs["visual_latent"] if "visual_latent" in obs else obs

    @staticmethod
    def _detach_if_tensor(value):
        return value.detach() if torch.is_tensor(value) else value

    @classmethod
    def _detach_rl_state(cls, value):
        # `critic_rl_state` is a plain tensor for most rl_token_source values, but
        # a (q1_state, q2_state) tuple for "image_last_linear" (separate critic
        # prefix-token linears per Q head). Detach whichever shape we got.
        if isinstance(value, tuple):
            return tuple(cls._detach_if_tensor(v) for v in value)
        return cls._detach_if_tensor(value)

    @staticmethod
    def _get_robot_state(obs, device, dtype):
        robot_state = obs.get("robot_state", obs.get("states", None))
        if robot_state is None:
            return None
        return robot_state.to(device, dtype=dtype)

    @staticmethod
    def _get_ref_action(obs, fallback_action, reshape_action_fn, device, dtype, name):
        ref_action = obs.get("ref_action", None)
        if ref_action is None:
            return fallback_action
        return reshape_action_fn(ref_action.to(device, dtype=dtype), name)

    @staticmethod
    def _as_batch_column(value, batch_size: int, device, dtype, reduce: str = "sum"):
        value = value.to(device=device, dtype=dtype)
        batch_size = int(batch_size)
        if value.dim() == 0:
            return value.reshape(1, 1).expand(batch_size, 1)
        if value.numel() % batch_size == 0:
            value = value.reshape(batch_size, -1)
        elif value.dim() == 1:
            return value.reshape(-1, 1)
        else:
            value = value.reshape(value.shape[0], -1)
        if reduce == "any":
            return value.bool().any(dim=-1, keepdim=True).to(dtype)
        if reduce == "first":
            return value[:, :1]
        return value.sum(dim=-1, keepdim=True)

    @staticmethod
    def _q_min(q_pair):
        q1, q2 = q_pair
        return torch.minimum(q1, q2)

    def _make_noisy_action(self, actions):
        if self.action_diag_noisy_std <= 0.0:
            return actions
        return actions + torch.randn_like(actions) * self.action_diag_noisy_std

    def _make_random_action(self, actions, low=None, high=None):
        low = self.action_diag_random_low if low is None else low
        high = self.action_diag_random_high if high is None else high
        return torch.empty_like(actions).uniform_(float(low), float(high))

    @staticmethod
    def _make_shuffled_action(actions):
        batch_size = int(actions.shape[0])
        if batch_size <= 1:
            return actions
        perm = torch.randperm(batch_size, device=actions.device)
        if bool((perm == torch.arange(batch_size, device=actions.device)).all().item()):
            perm = torch.roll(perm, shifts=1)
        return actions.index_select(0, perm)

    @staticmethod
    def _make_return_mismatched_action(actions, positive_mask):
        if positive_mask is None:
            return actions, actions.new_zeros(actions.shape[0], 1)
        batch_size = int(actions.shape[0])
        if batch_size <= 1:
            return actions, actions.new_zeros(batch_size, 1)
        mask = positive_mask.reshape(batch_size, -1).any(dim=-1)
        pos_idx = torch.where(mask)[0]
        neg_idx = torch.where(~mask)[0]
        if pos_idx.numel() == 0 or neg_idx.numel() == 0:
            return actions, actions.new_zeros(batch_size, 1)
        perm = torch.arange(batch_size, device=actions.device)
        pos_src = neg_idx[
            torch.randint(neg_idx.numel(), (pos_idx.numel(),), device=actions.device)
        ]
        neg_src = pos_idx[
            torch.randint(pos_idx.numel(), (neg_idx.numel(),), device=actions.device)
        ]
        perm[pos_idx] = pos_src
        perm[neg_idx] = neg_src
        valid = torch.ones(batch_size, 1, device=actions.device, dtype=actions.dtype)
        return actions.index_select(0, perm), valid

    @staticmethod
    def _make_nearest_return_mismatched_action(actions, match_state, positive_mask):
        if positive_mask is None:
            return actions, actions.new_zeros(actions.shape[0], 1)
        batch_size = int(actions.shape[0])
        if batch_size <= 1:
            return actions, actions.new_zeros(batch_size, 1)
        mask = positive_mask.reshape(batch_size, -1).any(dim=-1)
        pos_idx = torch.where(mask)[0]
        neg_idx = torch.where(~mask)[0]
        if pos_idx.numel() == 0 or neg_idx.numel() == 0:
            return actions, actions.new_zeros(batch_size, 1)

        if match_state is None:
            match_state = actions
        state = match_state.detach().reshape(batch_size, -1).float()
        state = F.normalize(state, dim=-1)
        dist = torch.cdist(state, state, p=2)
        invalid = mask[:, None] == mask[None, :]
        dist = dist.masked_fill(invalid, float("inf"))
        nearest = dist.argmin(dim=1)
        valid = torch.isfinite(dist.min(dim=1).values).reshape(batch_size, 1)
        valid = valid.to(device=actions.device, dtype=actions.dtype)
        return actions.index_select(0, nearest), valid

    @staticmethod
    def _masked_mean(value, mask):
        if mask is None:
            return value.mean()
        mask = mask.to(device=value.device, dtype=value.dtype)
        while mask.dim() < value.dim():
            mask = mask.unsqueeze(-1)
        return (value * mask).sum() / mask.sum().clamp_min(1.0)

    def _critic_q_min(self, model, forward_type, rl_state, action):
        q1, q2 = model(
            forward_type=forward_type,
            mode="critic",
            rl_state=rl_state,
            action=action,
            use_target=False,
        )
        return torch.minimum(q1, q2)

    def _add_action_diagnostics(
        self,
        aux,
        model,
        forward_type,
        curr_rl_state,
        actions,
        q_min_data,
        positive_mask=None,
    ):
        if not self.critic_action_diagnostics_enabled:
            return
        with torch.no_grad():
            noisy = self._make_noisy_action(actions)
            shuffled = self._make_shuffled_action(actions)
            mismatch, mismatch_valid = self._make_return_mismatched_action(
                actions, positive_mask
            )
            nearest_mismatch, nearest_valid = (
                self._make_nearest_return_mismatched_action(
                    actions, curr_rl_state, positive_mask
                )
            )
            random_action = self._make_random_action(actions)
            q_noisy = self._critic_q_min(model, forward_type, curr_rl_state, noisy)
            q_shuffle = self._critic_q_min(model, forward_type, curr_rl_state, shuffled)
            q_mismatch = self._critic_q_min(
                model, forward_type, curr_rl_state, mismatch
            )
            q_nearest_mismatch = self._critic_q_min(
                model, forward_type, curr_rl_state, nearest_mismatch
            )
            q_random = self._critic_q_min(
                model, forward_type, curr_rl_state, random_action
            )
            mismatch_valid_sum = mismatch_valid.sum().clamp_min(1.0)
            nearest_valid_sum = nearest_valid.sum().clamp_min(1.0)
            mismatch_gap = (
                (q_min_data - q_mismatch) * mismatch_valid
            ).sum() / mismatch_valid_sum
            nearest_gap = (
                (q_min_data - q_nearest_mismatch) * nearest_valid
            ).sum() / nearest_valid_sum
            mismatch_rank = (
                (q_min_data > q_mismatch).to(q_min_data.dtype) * mismatch_valid
            ).sum() / mismatch_valid_sum
            nearest_rank = (
                (q_min_data > q_nearest_mismatch).to(q_min_data.dtype) * nearest_valid
            ).sum() / nearest_valid_sum
            mismatch_mse = (
                torch.square(mismatch - actions)
                .reshape(actions.shape[0], -1)
                .mean(dim=-1, keepdim=True)
                * mismatch_valid
            ).sum() / mismatch_valid_sum
            nearest_mse = (
                torch.square(nearest_mismatch - actions)
                .reshape(actions.shape[0], -1)
                .mean(dim=-1, keepdim=True)
                * nearest_valid
            ).sum() / nearest_valid_sum
            aux.update(
                {
                    "action_q_data_mean": float(
                        q_min_data.detach().float().mean().item()
                    ),
                    "action_q_noisy_mean": float(
                        q_noisy.detach().float().mean().item()
                    ),
                    "action_q_shuffle_mean": float(
                        q_shuffle.detach().float().mean().item()
                    ),
                    "action_q_return_mismatch_mean": float(
                        ((q_mismatch * mismatch_valid).sum() / mismatch_valid_sum)
                        .detach()
                        .float()
                        .item()
                    ),
                    "action_q_nearest_return_mismatch_mean": float(
                        ((q_nearest_mismatch * nearest_valid).sum() / nearest_valid_sum)
                        .detach()
                        .float()
                        .item()
                    ),
                    "action_q_random_mean": float(
                        q_random.detach().float().mean().item()
                    ),
                    "action_gap_data_noisy": float(
                        (q_min_data - q_noisy).detach().float().mean().item()
                    ),
                    "action_gap_data_shuffle": float(
                        (q_min_data - q_shuffle).detach().float().mean().item()
                    ),
                    "action_gap_data_return_mismatch": float(
                        mismatch_gap.detach().float().item()
                    ),
                    "action_gap_data_nearest_return_mismatch": float(
                        nearest_gap.detach().float().item()
                    ),
                    "action_gap_data_random": float(
                        (q_min_data - q_random).detach().float().mean().item()
                    ),
                    "action_rank_data_gt_noisy": float(
                        (q_min_data > q_noisy).detach().float().mean().item()
                    ),
                    "action_rank_data_gt_shuffle": float(
                        (q_min_data > q_shuffle).detach().float().mean().item()
                    ),
                    "action_rank_data_gt_return_mismatch": float(
                        mismatch_rank.detach().float().item()
                    ),
                    "action_rank_data_gt_nearest_return_mismatch": float(
                        nearest_rank.detach().float().item()
                    ),
                    "action_rank_data_gt_random": float(
                        (q_min_data > q_random).detach().float().mean().item()
                    ),
                    "action_mse_noisy": float(
                        torch.square(noisy - actions).detach().float().mean().item()
                    ),
                    "action_mse_shuffle": float(
                        torch.square(shuffled - actions).detach().float().mean().item()
                    ),
                    "action_mse_return_mismatch": float(
                        mismatch_mse.detach().float().item()
                    ),
                    "action_mse_nearest_return_mismatch": float(
                        nearest_mse.detach().float().item()
                    ),
                    "action_return_mismatch_valid_frac": float(
                        mismatch_valid.detach().float().mean().item()
                    ),
                    "action_nearest_return_mismatch_valid_frac": float(
                        nearest_valid.detach().float().mean().item()
                    ),
                    "action_mse_random": float(
                        torch.square(random_action - actions)
                        .detach()
                        .float()
                        .mean()
                        .item()
                    ),
                    "action_abs_mean": float(
                        actions.detach().float().abs().mean().item()
                    ),
                    "action_abs_std": float(actions.detach().float().std().item()),
                    "rl_state_abs_mean": float(
                        curr_rl_state.detach().float().abs().mean().item()
                    ),
                    "rl_state_abs_std": float(
                        curr_rl_state.detach().float().std().item()
                    ),
                }
            )
            if positive_mask is not None:
                success_mask = (
                    positive_mask.to(device=actions.device, dtype=actions.dtype)
                    .reshape(actions.shape[0], -1)
                    .any(dim=-1, keepdim=True)
                    .to(dtype=actions.dtype)
                )
                success_nearest_mask = success_mask * nearest_valid
                success_mask = success_mask * mismatch_valid
                success_denom = success_mask.sum().clamp_min(1.0)
                success_nearest_denom = success_nearest_mask.sum().clamp_min(1.0)
                success_mismatch_gap = (
                    (q_min_data - q_mismatch) * success_mask
                ).sum() / success_denom
                success_nearest_gap = (
                    (q_min_data - q_nearest_mismatch) * success_nearest_mask
                ).sum() / success_nearest_denom
                success_mismatch_rank = (
                    (q_min_data > q_mismatch).to(q_min_data.dtype) * success_mask
                ).sum() / success_denom
                success_nearest_rank = (
                    (q_min_data > q_nearest_mismatch).to(q_min_data.dtype)
                    * success_nearest_mask
                ).sum() / success_nearest_denom
                aux.update(
                    {
                        "success_action_gap_data_return_mismatch": float(
                            success_mismatch_gap.detach().float().item()
                        ),
                        "success_action_rank_data_gt_return_mismatch": float(
                            success_mismatch_rank.detach().float().item()
                        ),
                        "success_action_gap_data_nearest_return_mismatch": float(
                            success_nearest_gap.detach().float().item()
                        ),
                        "success_action_rank_data_gt_nearest_return_mismatch": float(
                            success_nearest_rank.detach().float().item()
                        ),
                        "success_action_q_return_mismatch_mean": float(
                            ((q_mismatch * success_mask).sum() / success_denom)
                            .detach()
                            .float()
                            .item()
                        ),
                        "success_action_q_nearest_return_mismatch_mean": float(
                            (
                                (q_nearest_mismatch * success_nearest_mask).sum()
                                / success_nearest_denom
                            )
                            .detach()
                            .float()
                            .item()
                        ),
                        "success_action_return_mismatch_frac": float(
                            success_mask.detach().float().mean().item()
                        ),
                        "success_action_nearest_return_mismatch_frac": float(
                            success_nearest_mask.detach().float().mean().item()
                        ),
                    }
                )

    def _action_contrastive_loss(
        self,
        model,
        forward_type,
        curr_rl_state,
        actions,
        q_min_data,
        positive_mask,
        nearest_match_state=None,
    ):
        if self.critic_action_contrastive_coef <= 0.0:
            return actions.new_zeros(()), {}
        bad_actions = []
        if self.critic_action_contrastive_use_noisy:
            bad_actions.append(
                actions
                + torch.randn_like(actions) * self.critic_action_contrastive_noisy_std
            )
        if self.critic_action_contrastive_use_shuffle:
            bad_actions.append(self._make_shuffled_action(actions))
        if self.critic_action_contrastive_use_return_mismatch:
            mismatch, mismatch_valid = self._make_return_mismatched_action(
                actions, positive_mask
            )
            if float(mismatch_valid.detach().sum().item()) > 0.0:
                bad_actions.append((mismatch, mismatch_valid))
        if self.critic_action_contrastive_use_nearest_return_mismatch:
            mismatch, mismatch_valid = self._make_nearest_return_mismatched_action(
                actions, curr_rl_state, positive_mask
            )
            if float(mismatch_valid.detach().sum().item()) > 0.0:
                bad_actions.append((mismatch, mismatch_valid))
        if self.critic_action_contrastive_use_random:
            bad_actions.append(
                self._make_random_action(
                    actions,
                    low=self.critic_action_contrastive_random_low,
                    high=self.critic_action_contrastive_random_high,
                )
            )
        if not bad_actions:
            return actions.new_zeros(()), {}

        mask = positive_mask if self.critic_action_contrastive_positive_only else None
        margin = q_min_data.new_tensor(float(self.critic_action_contrastive_margin))
        losses = []
        gaps = []
        for bad_item in bad_actions:
            bad_mask = None
            if isinstance(bad_item, tuple):
                bad_action, bad_mask = bad_item
            else:
                bad_action = bad_item
            q_bad = self._critic_q_min(model, forward_type, curr_rl_state, bad_action)
            this_mask = bad_mask if mask is None else mask
            if bad_mask is not None and mask is not None:
                this_mask = bad_mask * mask
            losses.append(
                self._masked_mean(torch.relu(margin + q_bad - q_min_data), this_mask)
            )
            gaps.append(self._masked_mean(q_min_data - q_bad, this_mask))
        loss = torch.stack(losses).mean()
        gap = torch.stack(gaps).mean()
        return loss, {
            "action_contrastive_loss": float(loss.detach().float().item()),
            "action_contrastive_gap": float(gap.detach().float().item()),
            "action_contrastive_coef": float(self.critic_action_contrastive_coef),
            "action_contrastive_positive_frac": float(
                positive_mask.detach().float().mean().item()
                if positive_mask is not None
                else 1.0
            ),
        }

    def _cql_loss(self, model, forward_type, curr_rl_state, actions, q_min_data):
        if self.critic_cql_coef <= 0.0:
            return actions.new_zeros(()), {}
        q_candidates = []
        for _ in range(max(1, int(self.critic_cql_num_random))):
            q_candidates.append(
                self._critic_q_min(
                    model,
                    forward_type,
                    curr_rl_state,
                    self._make_random_action(
                        actions,
                        low=self.critic_cql_random_low,
                        high=self.critic_cql_random_high,
                    ),
                )
            )
        if self.critic_cql_include_shuffle:
            q_candidates.append(
                self._critic_q_min(
                    model,
                    forward_type,
                    curr_rl_state,
                    self._make_shuffled_action(actions),
                )
            )
        q_stack = torch.stack(q_candidates + [q_min_data], dim=0)
        temperature = max(float(self.critic_cql_temperature), 1e-6)
        conservative = temperature * torch.logsumexp(q_stack / temperature, dim=0)
        loss = (conservative - q_min_data).mean()
        return loss, {
            "cql_loss": float(loss.detach().float().item()),
            "cql_coef": float(self.critic_cql_coef),
            "cql_candidate_q_mean": float(
                torch.stack(q_candidates).detach().float().mean().item()
            ),
        }

    def _quality_target_loss(self, q1, q2, actions, positive_mask):
        if self.critic_quality_coef <= 0.0:
            return actions.new_zeros(()), {}
        if actions.shape[1] > 1:
            velocity = actions[:, 1:] - actions[:, :-1]
            smooth_penalty = torch.square(velocity).mean(dim=(1, 2), keepdim=False)
        else:
            smooth_penalty = torch.zeros(
                actions.shape[0], device=actions.device, dtype=actions.dtype
            )
        range_penalty = torch.square(actions).mean(dim=(1, 2))
        quality = (
            1.0
            - self.critic_quality_jerk_coef * smooth_penalty
            - self.critic_quality_range_coef * range_penalty
        )
        quality = quality.clamp(0.0, 1.0).reshape(-1, 1)
        mask = positive_mask
        loss = 0.5 * (
            self._masked_mean(torch.square(q1 - quality), mask)
            + self._masked_mean(torch.square(q2 - quality), mask)
        )
        return loss, {
            "quality_loss": float(loss.detach().float().item()),
            "quality_coef": float(self.critic_quality_coef),
            "quality_target_mean": float(
                self._masked_mean(quality, mask).detach().float().item()
            ),
        }

    def _action_quality_regression_loss(
        self,
        model,
        forward_type,
        curr_rl_state,
        actions,
        mc_return,
        positive_mask,
    ):
        if self.critic_action_quality_coef <= 0.0 or mc_return is None:
            return actions.new_zeros(()), {}

        mask = positive_mask if self.critic_action_quality_positive_only else None
        if mask is not None and float(mask.detach().sum().item()) <= 0.0:
            return actions.new_zeros(()), {}

        mse_ref = max(float(self.critic_action_quality_mse_ref), 1e-8)
        penalty_scale = float(self.critic_action_quality_penalty_scale)
        candidates = []
        if self.critic_action_quality_use_noisy:
            candidates.append(("noisy", self._make_noisy_action(actions)))
        if self.critic_action_quality_use_random:
            candidates.append(("random", self._make_random_action(actions)))
        if self.critic_action_quality_use_nearest:
            nearest_action, nearest_valid = self._make_nearest_return_mismatched_action(
                actions, curr_rl_state, positive_mask
            )
            if float(nearest_valid.detach().sum().item()) > 0.0:
                candidates.append(("nearest", (nearest_action, nearest_valid)))
        if not candidates:
            return actions.new_zeros(()), {}

        losses = []
        gaps = []
        target_means = []
        for _, candidate in candidates:
            candidate_mask = mask
            if isinstance(candidate, tuple):
                bad_action, valid_mask = candidate
                candidate_mask = (
                    valid_mask
                    if candidate_mask is None
                    else candidate_mask * valid_mask
                )
            else:
                bad_action = candidate

            q_bad_1, q_bad_2 = model(
                forward_type=forward_type,
                mode="critic",
                rl_state=curr_rl_state,
                action=bad_action,
                use_target=False,
            )
            action_mse = (
                torch.square(bad_action - actions)
                .reshape(actions.shape[0], -1)
                .mean(dim=-1, keepdim=True)
            )
            target = mc_return - penalty_scale * (action_mse / mse_ref)
            target = target.clamp_min(0.0).to(dtype=q_bad_1.dtype)
            loss = 0.5 * (
                self._masked_mean(torch.square(q_bad_1 - target), candidate_mask)
                + self._masked_mean(torch.square(q_bad_2 - target), candidate_mask)
            )
            losses.append(loss)
            gaps.append(
                self._masked_mean(
                    mc_return.to(dtype=q_bad_1.dtype) - target, candidate_mask
                )
            )
            target_means.append(self._masked_mean(target, candidate_mask))

        loss = torch.stack(losses).mean()
        gap = torch.stack(gaps).mean()
        target_mean = torch.stack(target_means).mean()
        return loss, {
            "action_quality_loss": float(loss.detach().float().item()),
            "action_quality_coef": float(self.critic_action_quality_coef),
            "action_quality_target_gap": float(gap.detach().float().item()),
            "action_quality_target_mean": float(target_mean.detach().float().item()),
        }

    def _local_action_margin_loss(
        self,
        model,
        forward_type,
        curr_rl_state,
        actions,
        q_min_data,
        positive_mask,
        nearest_match_state=None,
    ):
        if self.critic_local_action_margin_coef <= 0.0:
            return actions.new_zeros(()), {}

        mask = None
        if self.critic_local_action_margin_positive_only:
            mask = positive_mask
            if mask is not None and float(mask.detach().sum().item()) <= 0.0:
                return actions.new_zeros(()), {}

        candidates = []
        noisy_std = float(self.critic_local_action_margin_noisy_std)
        for _ in range(max(0, int(self.critic_local_action_margin_num_noisy))):
            if noisy_std > 0.0:
                candidates.append(
                    (actions + torch.randn_like(actions) * noisy_std, None)
                )
        if self.critic_local_action_margin_use_nearest:
            if self.critic_local_action_margin_nearest_key == "action":
                nearest_match_state = actions
            nearest_action, nearest_valid = self._make_nearest_return_mismatched_action(
                actions,
                nearest_match_state
                if nearest_match_state is not None
                else curr_rl_state,
                positive_mask,
            )
            if float(nearest_valid.detach().sum().item()) > 0.0:
                candidates.append((nearest_action, nearest_valid))
        if not candidates:
            return actions.new_zeros(()), {}

        mse_ref = max(float(self.critic_local_action_margin_mse_ref), 1e-8)
        base = float(self.critic_local_action_margin_base)
        scale = float(self.critic_local_action_margin_scale)
        losses = []
        gaps = []
        margins = []
        for bad_action, bad_mask in candidates:
            q_bad = self._critic_q_min(model, forward_type, curr_rl_state, bad_action)
            action_mse = (
                torch.square(bad_action - actions)
                .reshape(actions.shape[0], -1)
                .mean(dim=-1, keepdim=True)
            )
            margin = base + scale * (action_mse / mse_ref).clamp_min(0.0)
            this_mask = mask
            if bad_mask is not None:
                this_mask = bad_mask if this_mask is None else this_mask * bad_mask
            losses.append(
                self._masked_mean(torch.relu(margin + q_bad - q_min_data), this_mask)
            )
            gaps.append(self._masked_mean(q_min_data - q_bad, this_mask))
            margins.append(self._masked_mean(margin, this_mask))

        loss = torch.stack(losses).mean()
        gap = torch.stack(gaps).mean()
        margin_mean = torch.stack(margins).mean()
        return loss, {
            "local_action_margin_loss": float(loss.detach().float().item()),
            "local_action_margin_gap": float(gap.detach().float().item()),
            "local_action_margin_mean": float(margin_mean.detach().float().item()),
            "local_action_margin_coef": float(self.critic_local_action_margin_coef),
        }

    def _pairwise_stage_margin_loss(self, q_min_data, forward_inputs):
        if self.critic_pairwise_stage_margin_coef <= 0.0:
            return q_min_data.new_zeros(()), {}
        if not isinstance(forward_inputs, dict):
            return q_min_data.new_zeros(()), {}
        pair_id = forward_inputs.get("pair_id", None)
        pair_sign = forward_inputs.get("pair_sign", None)
        if not torch.is_tensor(pair_id) or not torch.is_tensor(pair_sign):
            return q_min_data.new_zeros(()), {}

        pair_id = pair_id.to(device=q_min_data.device).reshape(-1)
        pair_sign = pair_sign.to(
            device=q_min_data.device, dtype=q_min_data.dtype
        ).reshape(-1)
        q_flat = q_min_data.reshape(-1)
        losses = []
        gaps = []
        valid_pairs = 0
        for pid in torch.unique(pair_id):
            mask = pair_id == pid
            pos = mask & (pair_sign > 0)
            neg = mask & (pair_sign < 0)
            if not bool(pos.any().item()) or not bool(neg.any().item()):
                continue
            q_pos = q_flat[pos].mean()
            q_neg = q_flat[neg].mean()
            gap = q_pos - q_neg
            losses.append(
                torch.relu(q_flat.new_tensor(self.critic_pairwise_stage_margin) - gap)
            )
            gaps.append(gap)
            valid_pairs += 1
        if not losses:
            return q_min_data.new_zeros(()), {}
        loss = torch.stack(losses).mean()
        gap_mean = torch.stack(gaps).mean()
        return loss, {
            "pairwise_stage_margin_loss": float(loss.detach().float().item()),
            "pairwise_stage_gap": float(gap_mean.detach().float().item()),
            "pairwise_stage_valid_pairs": float(valid_pairs),
            "pairwise_stage_margin": float(self.critic_pairwise_stage_margin),
        }

    @staticmethod
    def _index_batch_tensor(value, index, batch_size):
        if value is None or not torch.is_tensor(value):
            return value
        if int(value.shape[0]) != int(batch_size):
            return value
        return value.index_select(0, index)

    def _pairwise_same_state_action_swap_loss(
        self,
        model,
        forward_type,
        curr_rl_state,
        actions,
        q_min_data,
        forward_inputs,
        critic_visual_tokens=None,
        critic_robot_state=None,
        critic_ref_action=None,
    ):
        if self.critic_pairwise_action_swap_coef <= 0.0:
            return q_min_data.new_zeros(()), {}
        if not isinstance(forward_inputs, dict):
            return q_min_data.new_zeros(()), {}
        pair_id = forward_inputs.get("pair_id", None)
        pair_sign = forward_inputs.get("pair_sign", None)
        if not torch.is_tensor(pair_id) or not torch.is_tensor(pair_sign):
            return q_min_data.new_zeros(()), {}

        batch_size = int(actions.shape[0])
        pair_id = pair_id.to(device=q_min_data.device).reshape(-1)
        pair_sign = pair_sign.to(
            device=q_min_data.device, dtype=q_min_data.dtype
        ).reshape(-1)
        q_flat = q_min_data.reshape(-1)
        losses = []
        gaps = []
        swapped_qs = []
        valid_pairs = 0
        margin = q_flat.new_tensor(float(self.critic_pairwise_action_swap_margin))
        for pid in torch.unique(pair_id):
            mask = pair_id == pid
            pos_idx = torch.where(mask & (pair_sign > 0))[0]
            neg_idx = torch.where(mask & (pair_sign < 0))[0]
            if pos_idx.numel() == 0 or neg_idx.numel() == 0:
                continue

            # Fix the successful state and swap in the same-stage failed action.
            # This directly tests whether Q depends on action, not only state.
            pos_idx = pos_idx[:1]
            neg_idx = neg_idx[:1]
            pos_rl_state = self._index_batch_tensor(curr_rl_state, pos_idx, batch_size)
            neg_action = actions.index_select(0, neg_idx)
            q_swap_1, q_swap_2 = model(
                forward_type=forward_type,
                mode="critic",
                rl_state=pos_rl_state,
                action=neg_action,
                use_target=False,
                critic_visual_tokens=self._index_batch_tensor(
                    critic_visual_tokens, pos_idx, batch_size
                ),
                critic_robot_state=self._index_batch_tensor(
                    critic_robot_state, pos_idx, batch_size
                ),
                critic_ref_action=self._index_batch_tensor(
                    critic_ref_action, pos_idx, batch_size
                ),
            )
            q_swap = torch.minimum(q_swap_1, q_swap_2).reshape(-1).mean()
            q_pos = q_flat[pos_idx].mean()
            gap = q_pos - q_swap
            losses.append(torch.relu(margin - gap))
            gaps.append(gap)
            swapped_qs.append(q_swap)
            valid_pairs += 1

        if not losses:
            return q_min_data.new_zeros(()), {}
        loss = torch.stack(losses).mean()
        gap_mean = torch.stack(gaps).mean()
        swap_q_mean = torch.stack(swapped_qs).mean()
        return loss, {
            "same_state_action_swap_loss": float(loss.detach().float().item()),
            "same_state_action_swap_gap": float(gap_mean.detach().float().item()),
            "same_state_action_swap_q_mean": float(swap_q_mean.detach().float().item()),
            "same_state_action_swap_valid_pairs": float(valid_pairs),
            "same_state_action_swap_margin": float(
                self.critic_pairwise_action_swap_margin
            ),
        }

    def compute_critic_loss(
        self,
        policy,
        model,
        batch,
        build_visual_feat_fn,
        reshape_action_fn,
        device,
        dtype,
        forward_type,
    ):
        """Compute TD3 critic loss.

        Args:
            policy: unwrapped policy (for target_actor_forward / target_critic_forward)
            model: FSDP-wrapped model (for current Q forward)
            batch: training batch dict
            build_visual_feat_fn: callable(visual_latent) -> visual_feat
            reshape_action_fn: callable(tensor, tensor_name) -> reshaped tensor
            device: torch device
            dtype: torch dtype
            forward_type: ForwardType.DEFAULT or ForwardType.TD3 depending on model

        Returns:
            (critic_loss, q1, q2, target_q_values, aux_dict)
        """
        curr_obs = batch["curr_obs"]
        next_obs = (
            batch.get("n_step_next_obs", batch["next_obs"])
            if self.use_n_step_target
            else batch["next_obs"]
        )
        actions = reshape_action_fn(
            batch["actions"].to(device, dtype=dtype),
            "batch.actions",
        )
        batch_size = int(actions.shape[0])
        rewards = batch["rewards"]
        terminations = batch["terminations"]

        immediate_rewards = self._as_batch_column(
            rewards,
            batch_size,
            device,
            dtype,
            reduce="sum",
        )
        n_step_return = (
            batch.get("n_step_return", None) if self.use_n_step_target else None
        )
        if n_step_return is not None:
            rewards_for_bootstrap = self._as_batch_column(
                n_step_return,
                batch_size,
                device,
                dtype,
                reduce="sum",
            )
            n_step_done = batch.get("n_step_done", None)
            if n_step_done is None:
                done_mask_bool = torch.zeros_like(
                    rewards_for_bootstrap, dtype=torch.bool
                )
            else:
                done_mask_bool = self._as_batch_column(
                    n_step_done,
                    batch_size,
                    device,
                    torch.float32,
                    reduce="any",
                ).bool()
            done_mask = done_mask_bool.to(dtype)
            chunk_discount = batch.get("n_step_discount", None)
            if chunk_discount is None:
                chunk_discount = (self._discount ** max(1, int(self.n_step))) * (
                    1.0 - done_mask
                )
            else:
                chunk_discount = self._as_batch_column(
                    chunk_discount,
                    batch_size,
                    device,
                    dtype,
                    reduce="first",
                )
        else:
            rewards_for_bootstrap = immediate_rewards
            done_mask_bool = terminations.to(device=device).bool()
            for done_key in ("dones", "truncations"):
                done_value = batch.get(done_key, None)
                if done_value is not None:
                    done_mask_bool = (
                        done_mask_bool | done_value.to(device=device).bool()
                    )
            done_mask = done_mask_bool.any(dim=-1, keepdim=True).to(dtype)
            # One replay transition corresponds to one high-level action chunk.
            # Rewards are stored per chunk, so gamma is defined on this same
            # high-level step rather than on each low-level substep.
            chunk_discount = self._discount * (1.0 - done_mask)

        curr_visual_feat = build_visual_feat_fn(self._get_visual_input(curr_obs))
        curr_ref_action = self._get_ref_action(
            curr_obs,
            actions,
            reshape_action_fn,
            device,
            dtype,
            "curr_obs.ref_action",
        )
        if self.critic_train_representation:
            _, curr_actor_aux = model(
                forward_type=forward_type,
                mode="actor",
                visual_feat=curr_visual_feat,
                robot_state=self._get_robot_state(curr_obs, device, dtype),
                ref_action=curr_ref_action,
                ref_action_dropout_p=0.0,
                use_target=False,
            )
            curr_rl_state = curr_actor_aux.get(
                "critic_rl_state", curr_actor_aux["rl_state"]
            )
            curr_critic_visual_tokens = curr_actor_aux.get("critic_visual_tokens", None)
            curr_critic_robot_state = curr_actor_aux.get("critic_robot_state", None)
            curr_critic_ref_action = curr_actor_aux.get("critic_ref_action", None)
        else:
            with torch.no_grad():
                _, curr_actor_aux = model(
                    forward_type=forward_type,
                    mode="actor",
                    visual_feat=self._detach_if_tensor(curr_visual_feat),
                    robot_state=self._get_robot_state(curr_obs, device, dtype),
                    ref_action=curr_ref_action,
                    ref_action_dropout_p=0.0,
                    use_target=False,
                )
                curr_rl_state = self._detach_rl_state(
                    curr_actor_aux.get("critic_rl_state", curr_actor_aux["rl_state"])
                )
                curr_critic_visual_tokens = curr_actor_aux.get(
                    "critic_visual_tokens", None
                )
                if curr_critic_visual_tokens is not None:
                    curr_critic_visual_tokens = curr_critic_visual_tokens.detach()
                curr_critic_robot_state = curr_actor_aux.get("critic_robot_state", None)
                if curr_critic_robot_state is not None:
                    curr_critic_robot_state = curr_critic_robot_state.detach()
                curr_critic_ref_action = curr_actor_aux.get("critic_ref_action", None)
                if curr_critic_ref_action is not None:
                    curr_critic_ref_action = curr_critic_ref_action.detach()

        with torch.no_grad():
            if self.critic_td_coef > 0.0:
                next_visual_feat = build_visual_feat_fn(
                    self._get_visual_input(next_obs)
                )
                next_ref_action = self._get_ref_action(
                    next_obs,
                    actions,
                    reshape_action_fn,
                    device,
                    dtype,
                    "next_obs.ref_action",
                )
                if self.critic_target_action_source in (
                    "data_next",
                    "next_data",
                    "behavior",
                ):
                    next_actions = self._get_ref_action(
                        next_obs,
                        actions,
                        reshape_action_fn,
                        device,
                        dtype,
                        "next_obs.ref_action",
                    )
                    _, next_actor_aux = policy.target_actor_forward(
                        visual_feat=self._detach_if_tensor(next_visual_feat),
                        robot_state=self._get_robot_state(next_obs, device, dtype),
                        ref_action=next_actions,
                        ref_action_dropout_p=float(self.actor_ref_action_dropout_p),
                    )
                else:
                    next_actions, next_actor_aux = policy.target_actor_forward(
                        visual_feat=self._detach_if_tensor(next_visual_feat),
                        robot_state=self._get_robot_state(next_obs, device, dtype),
                        ref_action=next_ref_action,
                        ref_action_dropout_p=float(self.actor_ref_action_dropout_p),
                    )
                if (
                    self.critic_target_action_source
                    not in ("data_next", "next_data", "behavior")
                    and self.target_policy_noise > 0.0
                ):
                    noise = torch.randn_like(next_actions) * self.target_policy_noise
                    noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                    next_actions = next_actions + noise
                if (
                    self.critic_target_action_source
                    not in ("data_next", "next_data", "behavior")
                    and self.target_action_clip is not None
                ):
                    next_actions = next_actions.clamp(
                        -self.target_action_clip, self.target_action_clip
                    )
                next_rl_state = next_actor_aux.get(
                    "critic_rl_state", next_actor_aux["rl_state"]
                )
                target_q1, target_q2 = policy.target_critic_forward(
                    rl_state=next_rl_state,
                    action=next_actions,
                    critic_rl_state=next_rl_state,
                    critic_visual_tokens=next_actor_aux.get(
                        "critic_visual_tokens", None
                    ),
                    critic_robot_state=next_actor_aux.get("critic_robot_state", None),
                    critic_ref_action=next_actor_aux.get("critic_ref_action", None),
                )
                target_q = torch.minimum(target_q1, target_q2)
                target_q_values = rewards_for_bootstrap + chunk_discount * target_q
            else:
                target_q_values = rewards_for_bootstrap

        q1, q2 = model(
            forward_type=forward_type,
            mode="critic",
            rl_state=curr_rl_state,
            action=actions,
            use_target=False,
            critic_visual_tokens=curr_critic_visual_tokens,
            critic_robot_state=curr_critic_robot_state,
            critic_ref_action=curr_critic_ref_action,
        )
        target_q_values = target_q_values.to(dtype=q1.dtype)
        td_loss = F.mse_loss(q1, target_q_values) + F.mse_loss(q2, target_q_values)
        critic_loss = self.critic_td_coef * td_loss
        q_min_data = torch.minimum(q1, q2)
        forward_inputs_for_metrics = batch.get("forward_inputs", None) or {}
        mc_return_for_metrics = forward_inputs_for_metrics.get("mc_return", None)
        positive_mask_for_action = None
        if mc_return_for_metrics is not None:
            mc_return_metric = self._as_batch_column(
                mc_return_for_metrics, batch_size, device, q1.dtype, reduce="sum"
            )
            positive_mask_for_action = (mc_return_metric > 1e-6).to(dtype=q1.dtype)
        else:
            positive_mask_for_action = (rewards_for_bootstrap > 1e-6).to(dtype=q1.dtype)

        aux = {
            "reward_mean": float(immediate_rewards.detach().float().mean().item()),
            "reward_max": float(immediate_rewards.detach().float().max().item()),
            "reward_positive_frac": float(
                (immediate_rewards.detach().float() > 0.0).float().mean().item()
            ),
            "n_step": float(self.n_step),
            "n_step_return_mean": float(
                rewards_for_bootstrap.detach().float().mean().item()
            ),
            "n_step_return_max": float(
                rewards_for_bootstrap.detach().float().max().item()
            ),
            "n_step_discount_mean": float(
                chunk_discount.detach().float().mean().item()
                if torch.is_tensor(chunk_discount)
                else float(chunk_discount)
            ),
            "done_frac": float(done_mask.detach().float().mean().item()),
            "td_critic_loss": float(td_loss.detach().float().item()),
            "td_critic_coef": float(self.critic_td_coef),
            "target_action_source_data": float(
                self.critic_target_action_source
                in ("data_next", "next_data", "behavior")
            ),
            "use_n_step_target": float(self.use_n_step_target),
            "training_stage_coupled": float(self.current_stage_name == "coupled_td3"),
        }

        self._add_action_diagnostics(
            aux,
            model,
            forward_type,
            curr_rl_state.detach()
            if torch.is_tensor(curr_rl_state) and curr_rl_state.requires_grad
            else curr_rl_state,
            actions.detach(),
            q_min_data.detach(),
            positive_mask_for_action.detach()
            if torch.is_tensor(positive_mask_for_action)
            else positive_mask_for_action,
        )

        contrastive_loss, contrastive_metrics = self._action_contrastive_loss(
            model,
            forward_type,
            curr_rl_state,
            actions,
            q_min_data,
            positive_mask_for_action,
        )
        if self.critic_action_contrastive_coef > 0.0:
            critic_loss = (
                critic_loss + self.critic_action_contrastive_coef * contrastive_loss
            )
            aux.update(contrastive_metrics)

        cql_loss, cql_metrics = self._cql_loss(
            model, forward_type, curr_rl_state, actions, q_min_data
        )
        if self.critic_cql_coef > 0.0:
            critic_loss = critic_loss + self.critic_cql_coef * cql_loss
            aux.update(cql_metrics)

        quality_loss, quality_metrics = self._quality_target_loss(
            q1, q2, actions, positive_mask_for_action
        )
        if self.critic_quality_coef > 0.0:
            critic_loss = critic_loss + self.critic_quality_coef * quality_loss
            aux.update(quality_metrics)

        action_quality_mc = None
        if mc_return_for_metrics is not None:
            action_quality_mc = self._as_batch_column(
                mc_return_for_metrics, batch_size, device, q1.dtype, reduce="sum"
            )
        action_quality_loss, action_quality_metrics = (
            self._action_quality_regression_loss(
                model,
                forward_type,
                curr_rl_state,
                actions,
                action_quality_mc,
                positive_mask_for_action,
            )
        )
        if self.critic_action_quality_coef > 0.0:
            critic_loss = (
                critic_loss + self.critic_action_quality_coef * action_quality_loss
            )
            aux.update(action_quality_metrics)

        local_margin_match_state = None
        if self.critic_local_action_margin_nearest_key == "robot_state":
            local_margin_match_state = curr_critic_robot_state
        elif self.critic_local_action_margin_nearest_key == "action":
            local_margin_match_state = actions
        local_margin_loss, local_margin_metrics = self._local_action_margin_loss(
            model,
            forward_type,
            curr_rl_state,
            actions,
            q_min_data,
            positive_mask_for_action,
            local_margin_match_state,
        )
        if self.critic_local_action_margin_coef > 0.0:
            critic_loss = (
                critic_loss + self.critic_local_action_margin_coef * local_margin_loss
            )
            aux.update(local_margin_metrics)

        pairwise_loss, pairwise_metrics = self._pairwise_stage_margin_loss(
            q_min_data,
            forward_inputs_for_metrics,
        )
        if self.critic_pairwise_stage_margin_coef > 0.0:
            critic_loss = (
                critic_loss + self.critic_pairwise_stage_margin_coef * pairwise_loss
            )
            aux.update(pairwise_metrics)

        same_state_action_swap_loss, same_state_action_swap_metrics = (
            self._pairwise_same_state_action_swap_loss(
                model,
                forward_type,
                curr_rl_state,
                actions,
                q_min_data,
                forward_inputs_for_metrics,
                curr_critic_visual_tokens,
                curr_critic_robot_state,
                curr_critic_ref_action,
            )
        )
        if self.critic_pairwise_action_swap_coef > 0.0:
            critic_loss = (
                critic_loss
                + self.critic_pairwise_action_swap_coef * same_state_action_swap_loss
            )
            aux.update(same_state_action_swap_metrics)

        if self.critic_action_grad_coef > 0.0:
            grad_actions = actions.detach().clone().requires_grad_(True)
            grad_q = self._critic_q_min(
                model, forward_type, curr_rl_state, grad_actions
            )
            action_grad = torch.autograd.grad(
                grad_q.sum(),
                grad_actions,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            grad_norm = action_grad.reshape(action_grad.shape[0], -1).norm(dim=-1)
            grad_norm = grad_norm.clamp(max=float(self.critic_action_grad_max_norm))
            grad_target = grad_norm.new_tensor(float(self.critic_action_grad_target))
            grad_loss = torch.square(torch.relu(grad_target - grad_norm)).mean()
            critic_loss = critic_loss + self.critic_action_grad_coef * grad_loss
            aux.update(
                {
                    "action_grad_loss": float(grad_loss.detach().float().item()),
                    "action_grad_norm_mean": float(
                        grad_norm.detach().float().mean().item()
                    ),
                    "action_grad_coef": float(self.critic_action_grad_coef),
                }
            )

        if self.critic_mc_return_coef > 0.0:
            forward_inputs = batch.get("forward_inputs", None) or {}
            mc_return = forward_inputs.get("mc_return", None)
            if mc_return is not None:
                mc_return = self._as_batch_column(
                    mc_return, batch_size, device, q1.dtype, reduce="sum"
                )
                mc_loss = F.mse_loss(q1, mc_return) + F.mse_loss(q2, mc_return)
                critic_loss = critic_loss + self.critic_mc_return_coef * mc_loss
                aux.update(
                    {
                        "mc_return_loss": float(mc_loss.detach().float().item()),
                        "mc_return_coef": float(self.critic_mc_return_coef),
                        "mc_return_mean": float(
                            mc_return.detach().float().mean().item()
                        ),
                        "mc_return_max": float(mc_return.detach().float().max().item()),
                        "mc_return_positive_frac": float(
                            (mc_return.detach().float() > 0.0).float().mean().item()
                        ),
                        "mc_return_zero_frac": float(
                            (mc_return.detach().float() <= 1e-6).float().mean().item()
                        ),
                    }
                )

        if mc_return_for_metrics is not None:
            mc_return_metric = self._as_batch_column(
                mc_return_for_metrics, batch_size, device, q1.dtype, reduce="sum"
            )
            positive_mask_metric = (mc_return_metric > 1e-6).to(dtype=q1.dtype)
            zero_mask_metric = (mc_return_metric <= 1e-6).to(dtype=q1.dtype)
            if (
                float(positive_mask_metric.detach().sum().item()) > 0.0
                and float(zero_mask_metric.detach().sum().item()) > 0.0
            ):
                q_min_metric = torch.minimum(q1, q2)
                positive_q_min_metric = (
                    q_min_metric * positive_mask_metric
                ).sum() / positive_mask_metric.sum().clamp_min(1.0)
                zero_q_min_metric = (
                    q_min_metric * zero_mask_metric
                ).sum() / zero_mask_metric.sum().clamp_min(1.0)
                aux.update(
                    {
                        "positive_q_min_mean": float(
                            positive_q_min_metric.detach().float().item()
                        ),
                        "zero_q_min_mean": float(
                            zero_q_min_metric.detach().float().item()
                        ),
                        "return_gap_min": float(
                            (positive_q_min_metric - zero_q_min_metric)
                            .detach()
                            .float()
                            .item()
                        ),
                    }
                )

        if self.critic_success_bce_coef > 0.0:
            forward_inputs = batch.get("forward_inputs", None) or {}
            mc_return = forward_inputs.get("mc_return", None)
            if mc_return is not None:
                mc_return = self._as_batch_column(
                    mc_return, batch_size, device, q1.dtype, reduce="sum"
                )
                labels = (mc_return > 1e-6).to(dtype=q1.dtype)
                threshold = q1.new_tensor(float(self.critic_success_bce_threshold))
                temperature = max(float(self.critic_success_bce_temperature), 1e-6)
                q1_logits = (q1 - threshold) / temperature
                q2_logits = (q2 - threshold) / temperature
                bce_loss = 0.5 * (
                    F.binary_cross_entropy_with_logits(q1_logits, labels)
                    + F.binary_cross_entropy_with_logits(q2_logits, labels)
                )
                critic_loss = critic_loss + self.critic_success_bce_coef * bce_loss
                with torch.no_grad():
                    prob = torch.sigmoid(0.5 * (q1_logits + q2_logits))
                    pred = (prob > 0.5).to(dtype=q1.dtype)
                    acc = (pred == labels).to(dtype=q1.dtype).mean()
                aux.update(
                    {
                        "success_bce_loss": float(bce_loss.detach().float().item()),
                        "success_bce_coef": float(self.critic_success_bce_coef),
                        "success_bce_threshold": float(
                            self.critic_success_bce_threshold
                        ),
                        "success_bce_temperature": float(
                            self.critic_success_bce_temperature
                        ),
                        "success_bce_acc": float(acc.detach().float().item()),
                    }
                )

        if self.critic_positive_return_coef > 0.0:
            forward_inputs = batch.get("forward_inputs", None) or {}
            mc_return = forward_inputs.get("mc_return", None)
            if mc_return is not None:
                mc_return = self._as_batch_column(
                    mc_return, batch_size, device, q1.dtype, reduce="sum"
                )
                positive_mask = (mc_return > 1e-6).to(dtype=q1.dtype)
                if float(positive_mask.detach().sum().item()) > 0.0:
                    denom = (2.0 * positive_mask.sum()).clamp_min(1.0)
                    positive_loss = (
                        (torch.square(q1 - mc_return) * positive_mask).sum()
                        + (torch.square(q2 - mc_return) * positive_mask).sum()
                    ) / denom
                    critic_loss = (
                        critic_loss + self.critic_positive_return_coef * positive_loss
                    )
                    positive_count = float(positive_mask.detach().sum().item())
                    aux.update(
                        {
                            "positive_return_loss": float(
                                positive_loss.detach().float().item()
                            ),
                            "positive_return_coef": float(
                                self.critic_positive_return_coef
                            ),
                            "positive_return_frac": float(
                                positive_mask.detach().float().mean().item()
                            ),
                            "positive_return_mean": float(
                                (mc_return * positive_mask)
                                .detach()
                                .float()
                                .sum()
                                .item()
                                / max(1.0, positive_count)
                            ),
                        }
                    )

        if self.critic_zero_return_coef > 0.0:
            forward_inputs = batch.get("forward_inputs", None) or {}
            mc_return = forward_inputs.get("mc_return", None)
            if mc_return is not None:
                mc_return = self._as_batch_column(
                    mc_return, batch_size, device, q1.dtype, reduce="sum"
                )
                zero_mask = (mc_return <= 1e-6).to(dtype=q1.dtype)
                if float(zero_mask.detach().sum().item()) > 0.0:
                    denom = (2.0 * zero_mask.sum()).clamp_min(1.0)
                    zero_loss = (
                        (torch.square(q1) * zero_mask).sum()
                        + (torch.square(q2) * zero_mask).sum()
                    ) / denom
                    critic_loss = critic_loss + self.critic_zero_return_coef * zero_loss
                    aux.update(
                        {
                            "zero_return_loss": float(
                                zero_loss.detach().float().item()
                            ),
                            "zero_return_coef": float(self.critic_zero_return_coef),
                            "zero_return_frac": float(
                                zero_mask.detach().float().mean().item()
                            ),
                            "zero_return_count": float(
                                zero_mask.detach().float().sum().item()
                            ),
                        }
                    )

        if self.critic_return_margin_coef > 0.0:
            forward_inputs = batch.get("forward_inputs", None) or {}
            mc_return = forward_inputs.get("mc_return", None)
            if mc_return is not None:
                mc_return = self._as_batch_column(
                    mc_return, batch_size, device, q1.dtype, reduce="sum"
                )
                positive_mask = (mc_return > 1e-6).to(dtype=q1.dtype)
                zero_mask = (mc_return <= 1e-6).to(dtype=q1.dtype)
                if (
                    float(positive_mask.detach().sum().item()) > 0.0
                    and float(zero_mask.detach().sum().item()) > 0.0
                ):
                    positive_q1_mean = (
                        q1 * positive_mask
                    ).sum() / positive_mask.sum().clamp_min(1.0)
                    zero_q1_mean = (q1 * zero_mask).sum() / zero_mask.sum().clamp_min(
                        1.0
                    )
                    positive_q2_mean = (
                        q2 * positive_mask
                    ).sum() / positive_mask.sum().clamp_min(1.0)
                    zero_q2_mean = (q2 * zero_mask).sum() / zero_mask.sum().clamp_min(
                        1.0
                    )
                    q_min = torch.minimum(q1, q2)
                    positive_q_min_mean = (
                        q_min * positive_mask
                    ).sum() / positive_mask.sum().clamp_min(1.0)
                    zero_q_min_mean = (
                        q_min * zero_mask
                    ).sum() / zero_mask.sum().clamp_min(1.0)
                    return_gap_q1 = positive_q1_mean - zero_q1_mean
                    return_gap_q2 = positive_q2_mean - zero_q2_mean
                    return_gap_min = positive_q_min_mean - zero_q_min_mean
                    return_gap = 0.5 * (return_gap_q1 + return_gap_q2)
                    margin = q1.new_tensor(float(self.critic_return_margin))
                    margin_loss = 0.5 * (
                        torch.square(torch.relu(margin - return_gap_q1))
                        + torch.square(torch.relu(margin - return_gap_q2))
                    )
                    critic_loss = (
                        critic_loss + self.critic_return_margin_coef * margin_loss
                    )
                    aux.update(
                        {
                            "return_margin_loss": float(
                                margin_loss.detach().float().item()
                            ),
                            "return_margin_coef": float(self.critic_return_margin_coef),
                            "return_margin": float(self.critic_return_margin),
                            "return_gap": float(return_gap.detach().float().item()),
                            "return_gap_q1": float(
                                return_gap_q1.detach().float().item()
                            ),
                            "return_gap_q2": float(
                                return_gap_q2.detach().float().item()
                            ),
                            "positive_q_mean": float(
                                (0.5 * (positive_q1_mean + positive_q2_mean))
                                .detach()
                                .float()
                                .item()
                            ),
                            "zero_q_mean": float(
                                (0.5 * (zero_q1_mean + zero_q2_mean))
                                .detach()
                                .float()
                                .item()
                            ),
                            "positive_q1_mean": float(
                                positive_q1_mean.detach().float().item()
                            ),
                            "zero_q1_mean": float(zero_q1_mean.detach().float().item()),
                            "positive_q2_mean": float(
                                positive_q2_mean.detach().float().item()
                            ),
                            "zero_q2_mean": float(zero_q2_mean.detach().float().item()),
                            "positive_q_min_mean": float(
                                positive_q_min_mean.detach().float().item()
                            ),
                            "zero_q_min_mean": float(
                                zero_q_min_mean.detach().float().item()
                            ),
                            "return_gap_min": float(
                                return_gap_min.detach().float().item()
                            ),
                        }
                    )

        return critic_loss, q1, q2, target_q_values, aux

    # ------------------------------------------------------------------
    # Actor
    # ------------------------------------------------------------------

    def compose_actor_loss(self, q_pi: torch.Tensor | None, bc_loss: torch.Tensor):
        """Compose actor loss from Q-value term and BC loss.

        trust_region mode is intentionally excluded — it requires parameter
        snapshot/restore logic that belongs in the worker.

        Returns:
            (actor_loss, metrics_dict)
        """
        q_term = bc_loss.new_zeros(())
        q_weight = float(self.actor_q_coef)
        effective_bc_coef = float(self.bc_coef)
        hard_penalty = bc_loss.new_zeros(())
        guard_active = 0.0
        q_pi_for_loss = None

        if q_pi is not None:
            q_pi_for_loss = q_pi
            q_term = (-q_pi_for_loss).mean()

        if self.actor_bc_guard_mode == "weighted":
            effective_bc_coef = float(self.actor_bc_weighted_coef)
        elif self.actor_bc_guard_mode == "hard_penalty":
            exceed = torch.clamp(bc_loss - self.actor_bc_guard_threshold, min=0.0)
            if float(exceed.detach().item()) > 0.0:
                guard_active = 1.0
            hard_penalty = self.actor_bc_penalty_coef * (
                exceed**self.actor_bc_penalty_power
            )

        actor_loss = q_weight * q_term + effective_bc_coef * bc_loss + hard_penalty
        metrics = {
            "bc_coef_effective": effective_bc_coef,
            "q_pi_used_for_loss": (
                float(q_pi_for_loss.mean().detach().item())
                if q_pi_for_loss is not None
                else 0.0
            ),
            "bc_guard_mode": float(
                0
                if self.actor_bc_guard_mode == "none"
                else 1
                if self.actor_bc_guard_mode == "weighted"
                else 2
                if self.actor_bc_guard_mode == "hard_penalty"
                else 3
                if self.actor_bc_guard_mode == "trust_region"
                else -1
            ),
            "bc_guard_threshold": float(self.actor_bc_guard_threshold),
            "bc_guard_penalty": float(hard_penalty.detach().item()),
            "bc_guard_active": guard_active,
            "q_weight": q_weight,
        }
        return actor_loss, metrics

    # ------------------------------------------------------------------
    # Scheduling helpers
    # ------------------------------------------------------------------

    def should_update_actor(self, update_step: int) -> bool:
        return update_step % self.critic_actor_ratio == 0

    def apply_training_stage(self, update_step: int) -> str:
        if self.coupling_start_step >= 0 and update_step >= self.coupling_start_step:
            self.current_stage_name = "coupled_td3"
            self.critic_target_action_source = self.coupled_critic_target_action_source
            self.actor_q_coef = self.coupled_actor_q_coef
            self.bc_coef = self.coupled_bc_coef
            self.target_policy_noise = self.coupled_target_policy_noise
            self.target_noise_clip = self.coupled_target_noise_clip
            self.target_action_clip = self.coupled_target_action_clip
        else:
            self.current_stage_name = "bootstrap"
            self.critic_target_action_source = self.base_critic_target_action_source
            self.actor_q_coef = self.base_actor_q_coef
            self.bc_coef = self.base_bc_coef
            self.target_policy_noise = self.base_target_policy_noise
            self.target_noise_clip = self.base_target_noise_clip
            self.target_action_clip = self.base_target_action_clip
        return self.current_stage_name

    def get_target_update_tau(
        self,
        update_step: int,
        stage_actor_bc_only: bool,
        stage_freeze_actor: bool,
        actor_updated: bool,
        critic_updated: bool,
    ) -> float | None:
        """Return tau for target update, or None if no update should happen."""
        if update_step % self.target_update_freq != 0:
            return None

        if stage_actor_bc_only:
            if not actor_updated:
                return None
            return 1.0
        elif stage_freeze_actor:
            if not critic_updated:
                return None
        else:
            if not (actor_updated or critic_updated):
                return None

        return self.target_tau

    def set_discount(self, discount: float):
        self._discount = discount

    def set_action_horizon(self, action_horizon: int):
        self._action_horizon = action_horizon
