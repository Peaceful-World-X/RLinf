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
        next_obs = batch.get("n_step_next_obs", batch["next_obs"])
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
        n_step_return = batch.get("n_step_return", None)
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
                curr_rl_state = curr_actor_aux.get(
                    "critic_rl_state", curr_actor_aux["rl_state"]
                ).detach()
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
                    )
                else:
                    next_actions, next_actor_aux = policy.target_actor_forward(
                        visual_feat=self._detach_if_tensor(next_visual_feat),
                        robot_state=self._get_robot_state(next_obs, device, dtype),
                        ref_action=next_ref_action,
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
            "training_stage_coupled": float(self.current_stage_name == "coupled_td3"),
        }

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

        forward_inputs_for_metrics = batch.get("forward_inputs", None) or {}
        mc_return_for_metrics = forward_inputs_for_metrics.get("mc_return", None)
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
