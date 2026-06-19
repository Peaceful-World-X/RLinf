# Copyright 2026 The RLinf Authors.
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

"""Synchronous TD3 policy worker for RLinf.

Mirrors EmbodiedSACFSDPPolicy but uses TD3Algorithm (deterministic policy,
dual critics, policy delay, no entropy temperature).
"""

import os

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from rlinf.algorithms.td3 import TD3Algorithm
from rlinf.data.embodied_buffer_dataset import (
    PreloadReplayBufferDataset,
    ReplayBufferDataset,
    replay_buffer_collate_fn,
)
from rlinf.data.embodied_io_struct import Trajectory
from rlinf.data.replay_buffer import TrajectoryReplayBuffer
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.scheduler import Channel, Worker
from rlinf.utils import drq
from rlinf.utils.distributed import all_reduce_dict
from rlinf.utils.metric_utils import append_to_dict, compute_split_num
from rlinf.utils.nested_dict_process import put_tensor_device, split_dict_to_chunk
from rlinf.utils.offline_td3_visualization import (
    SimpleUrdfKinematics,
    actions_to_absolute_joint_targets,
    evaluate_validation_trajectory,
    extract_three_view_images,
    load_action_norm_stats,
    load_key_segment_start_map,
    load_validation_trajectories,
    plot_action_chunk_segments_3d,
    plot_action_chunk_triplet_segments_3d,
    plot_action_mse_heatmaps,
    plot_boundary_trajectory_3d,
    plot_critic_timeline_with_images,
    plot_pt_gt_reconstruction_segments_3d,
    plot_pt_state_trajectory_3d,
    plot_q_values,
    plot_trajectory_3d,
    plot_trajectory_3d_with_state,
    save_json,
    tcp_trajectories_from_joint_targets,
)
from rlinf.utils.utils import clear_memory
from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor


class EmbodiedTD3FSDPPolicy(EmbodiedFSDPActor):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.replay_buffer = None
        self.target_model = None
        self.demo_buffer = None
        self.actor_optimizer = None
        self.critic_optimizer = None
        self.update_step = 0
        self.enable_drq = bool(getattr(self.cfg.actor, "enable_drq", False))

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def init_worker(self):
        self.setup_model_and_optimizer(initialize_target=True)
        self.setup_td3_components()
        self.soft_update_target_model(tau=1.0)
        self._setup_rollout_weight_dst_ranks()
        if self.cfg.actor.get("compile_model", False):
            self.model = torch.compile(self.model, mode="default")
            self.target_model = torch.compile(self.target_model, mode="default")

    def setup_model_and_optimizer(self, initialize_target=False):
        module = self.model_provider_func()
        if initialize_target:
            target_module = self.model_provider_func()

        self.model = self._strategy.wrap_model(
            model=module, device_mesh=self._device_mesh
        )
        if self.torch_dtype is None:
            self.torch_dtype = next(self.model.parameters()).dtype

        if initialize_target:
            self.target_model = self._strategy.wrap_model(
                model=target_module, device_mesh=self._device_mesh
            )
            self.target_model.requires_grad_(False)
            self.target_model_initialized = True

        # actor_optimizer: all params except critic-filtered modules.
        # critic_optimizer: critic heads, and optionally the RL-token encoder
        # so the critic can learn a task-reward representation while keeping
        # the VLA backbone and decoder frozen.
        critic_param_filters = ["critic_head_1", "critic_head_2"]
        if bool(self.cfg.algorithm.get("critic_train_rl_token_encoder", False)):
            if bool(
                self.cfg.actor.model.get("critic_separate_rl_token_encoder", False)
            ):
                critic_param_filters.append("critic_rl_token_encoder")
            else:
                critic_param_filters.append("rl_token_autoencoder.encoder")
        param_filters = {"critic": critic_param_filters}
        filtered_optim_config = {"critic": self.cfg.actor.critic_optim}
        optimizers = self.build_optimizers(
            model=self.model,
            main_optim_config=self.cfg.actor.optim,
            param_filters=param_filters,
            filtered_optim_config=filtered_optim_config,
        )
        self.actor_optimizer = optimizers[0]
        self.critic_optimizer = optimizers[1]
        # Keep SAC-compatible aliases so base class helpers work
        self.optimizer = self.actor_optimizer
        self.qf_optimizer = self.critic_optimizer

        self.build_lr_schedulers()
        self.grad_scaler = self.build_grad_scaler(
            self.cfg.actor.fsdp_config.grad_scaler
        )
        self.log_on_first_rank(
            "TD3 trainable parameters: "
            + ", ".join(
                name
                for name, param in self.model.named_parameters()
                if param.requires_grad
            )
        )

    def build_lr_schedulers(self):
        self.lr_scheduler = self.build_lr_scheduler(
            self.actor_optimizer, self.cfg.actor.optim
        )
        self.qf_lr_scheduler = self.build_lr_scheduler(
            self.critic_optimizer, self.cfg.actor.critic_optim
        )

    def setup_td3_components(self):
        seed = self.cfg.actor.get("seed", 1234)
        auto_save_path = self.cfg.algorithm.replay_buffer.get("auto_save_path", None)
        if auto_save_path is None:
            auto_save_path = os.path.join(
                self.cfg.runner.logger.log_path, f"replay_buffer/rank_{self._rank}"
            )
        else:
            auto_save_path = os.path.join(auto_save_path, f"rank_{self._rank}")

        self.replay_buffer = TrajectoryReplayBuffer(
            seed=seed,
            enable_cache=self.cfg.algorithm.replay_buffer.enable_cache,
            cache_size=self.cfg.algorithm.replay_buffer.cache_size,
            sample_window_size=self.cfg.algorithm.replay_buffer.sample_window_size,
            auto_save=self.cfg.algorithm.replay_buffer.get("auto_save", False),
            auto_save_path=auto_save_path,
            trajectory_format=self.cfg.algorithm.replay_buffer.get(
                "trajectory_format", "pt"
            ),
        )
        self.replay_buffer.mc_return_gamma = float(self.cfg.algorithm.gamma)
        if hasattr(self.replay_buffer, "set_n_step"):
            self.replay_buffer.set_n_step(
                int(self.cfg.algorithm.get("n_step", 1)),
                float(self.cfg.algorithm.gamma),
            )

        min_demo_buffer_size = 0
        if self.cfg.algorithm.get("demo_buffer", None) is not None:
            demo_auto_save = self.cfg.algorithm.demo_buffer.get("auto_save_path", None)
            if demo_auto_save is None:
                demo_auto_save = os.path.join(
                    self.cfg.runner.logger.log_path, f"demo_buffer/rank_{self._rank}"
                )
            else:
                demo_auto_save = os.path.join(demo_auto_save, f"rank_{self._rank}")
            self.demo_buffer = TrajectoryReplayBuffer(
                seed=seed,
                enable_cache=self.cfg.algorithm.demo_buffer.enable_cache,
                cache_size=self.cfg.algorithm.demo_buffer.cache_size,
                sample_window_size=self.cfg.algorithm.demo_buffer.sample_window_size,
                auto_save=self.cfg.algorithm.demo_buffer.get("auto_save", False),
                auto_save_path=demo_auto_save,
                trajectory_format="pt",
            )
            self.demo_buffer.mc_return_gamma = float(self.cfg.algorithm.gamma)
            if hasattr(self.demo_buffer, "set_n_step"):
                self.demo_buffer.set_n_step(
                    int(self.cfg.algorithm.get("n_step", 1)),
                    float(self.cfg.algorithm.gamma),
                )
            min_demo_buffer_size = self.cfg.algorithm.demo_buffer.min_buffer_size

        buffer_dataset_cls = (
            PreloadReplayBufferDataset
            if self.cfg.algorithm.replay_buffer.get("enable_preload", False)
            else ReplayBufferDataset
        )
        self.buffer_dataset = buffer_dataset_cls(
            replay_buffer=self.replay_buffer,
            demo_buffer=self.demo_buffer,
            batch_size=self.cfg.actor.global_batch_size // self._world_size,
            min_replay_buffer_size=self.cfg.algorithm.replay_buffer.min_buffer_size,
            min_demo_buffer_size=min_demo_buffer_size,
            prefetch_size=self.cfg.algorithm.replay_buffer.get("prefetch_size", 10),
        )
        self.buffer_dataloader = DataLoader(
            self.buffer_dataset,
            batch_size=1,
            num_workers=0,
            drop_last=True,
            collate_fn=replay_buffer_collate_fn,
        )
        self.buffer_dataloader_iter = iter(self.buffer_dataloader)

        self.td3_algorithm = TD3Algorithm(
            self.cfg.algorithm, self.cfg.actor.get("policy_head", None)
        )
        self.td3_algorithm.set_discount(self.cfg.algorithm.gamma)
        action_horizon = self.cfg.actor.model.get(
            "action_horizon", self.cfg.actor.model.num_action_chunks
        )
        self.td3_algorithm.set_action_horizon(action_horizon)
        self.target_update_type = self.cfg.algorithm.get("target_update_type", "all")

    # ------------------------------------------------------------------
    # Target network soft update (float32 shadow for bf16 precision)
    # ------------------------------------------------------------------

    def soft_update_target_model(self, tau=None):
        if tau is None:
            tau = self.cfg.algorithm.tau
        assert self.target_model_initialized

        with torch.no_grad():
            if not hasattr(self, "_target_shadow_f32"):
                for (n1, online), (n2, target) in zip(
                    self.model.named_parameters(),
                    self.target_model.named_parameters(),
                ):
                    assert n1 == n2
                    target.data.mul_(1.0 - tau).add_(online.data * tau)
            else:
                for (n1, online), (n2, target) in zip(
                    self.model.named_parameters(),
                    self.target_model.named_parameters(),
                ):
                    assert n1 == n2
                    shadow = self._target_shadow_f32[n1]
                    shadow.mul_(1.0 - tau).add_(online.data.float(), alpha=tau)
                    target.data.copy_(shadow.to(target.data.dtype))
            self._soft_update_internal_policy_targets(tau)

    def _canonical_param_name(self, name):
        for prefix in ("module.", "_fsdp_wrapped_module."):
            if name.startswith(prefix):
                name = name[len(prefix) :]
        marker = "._fsdp_wrapped_module."
        if marker in name:
            name = name.split(marker, 1)[1]
        return name

    def _soft_update_internal_policy_targets(self, tau):
        online_params = {
            self._canonical_param_name(name): param
            for name, param in self.model.named_parameters()
        }
        target_params = {
            self._canonical_param_name(name): param
            for name, param in self.target_model.named_parameters()
        }
        prefix_pairs = (
            ("rl_token_autoencoder.", "target_rl_token_autoencoder."),
            ("critic_rl_token_encoder.", "target_critic_rl_token_encoder."),
            ("actor_head.", "target_actor_head."),
            ("critic_head_1.", "target_critic_head_1."),
            ("critic_head_2.", "target_critic_head_2."),
        )
        for source_prefix, target_prefix in prefix_pairs:
            for source_name, online in online_params.items():
                if not source_name.startswith(source_prefix):
                    continue
                suffix = source_name[len(source_prefix) :]
                target = target_params.get(target_prefix + suffix, None)
                if target is None:
                    continue
                target.data.mul_(1.0 - tau).add_(online.data, alpha=tau)

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------

    def build_visual_feat_fn(self, visual_latent):
        # Identity: prefix extraction happens inside policy.td3_forward
        return visual_latent

    def get_visual_input(self, obs):
        return obs["visual_latent"] if "visual_latent" in obs else obs

    def get_robot_state(self, obs):
        robot_state = obs.get("robot_state", obs.get("states", None))
        if robot_state is None:
            return None
        return robot_state.to(self.device, dtype=self.torch_dtype)

    def get_ref_action(self, obs, fallback_action, name):
        ref_action = obs.get("ref_action", None)
        if ref_action is None:
            return fallback_action
        return self.reshape_action_fn(
            ref_action.to(self.device, dtype=self.torch_dtype),
            name,
        )

    def reshape_action_fn(self, action, name):
        action_horizon = self.cfg.actor.model.get(
            "action_horizon", self.cfg.actor.model.num_action_chunks
        )
        action_dim = self.cfg.actor.model.action_dim
        action = action.reshape(action.shape[0], action_horizon, action_dim)
        clip = self.cfg.algorithm.get("behavior_action_clip", None)
        if clip is not None:
            clip = float(clip)
            action = action.clamp(-clip, clip)
        return action

    @Worker.timer("forward_critic")
    def forward_critic(self, batch):
        critic_loss, q1, q2, target_q, aux = self.td3_algorithm.compute_critic_loss(
            policy=self.target_model,
            model=self.model,
            batch=batch,
            build_visual_feat_fn=self.build_visual_feat_fn,
            reshape_action_fn=self.reshape_action_fn,
            device=self.device,
            dtype=self.torch_dtype,
            forward_type=ForwardType.TD3,
        )
        metrics = {
            "critic_loss": critic_loss.item(),
            "q1_mean": q1.mean().item(),
            "q2_mean": q2.mean().item(),
            "target_q_mean": target_q.mean().item(),
            **aux,
        }
        # monitor actor action MSE against ground truth every step (no grad)
        with torch.no_grad():
            curr_obs = batch["curr_obs"]
            visual_feat = self.build_visual_feat_fn(self.get_visual_input(curr_obs))
            sampled_actions = self.reshape_action_fn(
                batch["actions"].to(self.device, dtype=self.torch_dtype),
                "batch.actions",
            )
            ref_action = self.get_ref_action(
                curr_obs, sampled_actions, "curr_obs.ref_action"
            )
            actions, actor_aux = self.model(
                forward_type=ForwardType.TD3,
                mode="actor",
                visual_feat=visual_feat,
                robot_state=self.get_robot_state(curr_obs),
                ref_action=ref_action,
                ref_action_dropout_p=0.0,
                use_target=False,
                compute_recon_loss=False,
            )
            actor_q1, actor_q2 = self.model(
                forward_type=ForwardType.TD3,
                mode="critic",
                rl_state=actor_aux.get("critic_rl_state", actor_aux["rl_state"]),
                action=actions,
                use_target=False,
            )
            q_data = torch.minimum(q1.detach(), q2.detach())
            q_actor = torch.minimum(actor_q1, actor_q2)
            metrics["action_q_actor_mean"] = q_actor.mean().item()
            metrics["action_gap_data_actor"] = (q_data - q_actor).mean().item()
            metrics["action_rank_data_gt_actor"] = (
                (q_data > q_actor).float().mean().item()
            )
            metrics["actor_action_mse"] = torch.nn.functional.mse_loss(
                actions, sampled_actions
            ).item()
            metrics["actor_ref_action_mse"] = torch.nn.functional.mse_loss(
                actions, ref_action
            ).item()
        return critic_loss, metrics

    @Worker.timer("forward_actor")
    def forward_actor(self, batch):
        curr_obs = batch["curr_obs"]
        visual_feat = self.build_visual_feat_fn(self.get_visual_input(curr_obs))
        sampled_actions = self.reshape_action_fn(
            batch["actions"].to(self.device, dtype=self.torch_dtype),
            "batch.actions",
        )
        ref_action = self.get_ref_action(
            curr_obs,
            sampled_actions,
            "curr_obs.ref_action",
        )
        recon_coef = getattr(self.cfg.actor.model, "recon_loss_coef", 0.1)

        actions, actor_aux = self.model(
            forward_type=ForwardType.TD3,
            mode="actor",
            visual_feat=visual_feat,
            robot_state=self.get_robot_state(curr_obs),
            ref_action=ref_action,
            ref_action_dropout_p=float(self.td3_algorithm.actor_ref_action_dropout_p),
            use_target=False,
            compute_recon_loss=recon_coef > 0.0,
        )

        q_pi = None
        q_pi_mean = 0.0
        if float(self.td3_algorithm.actor_q_coef) > 0.0:
            q1, q2 = self.model(
                forward_type=ForwardType.TD3_Q,
                rl_state=actor_aux.get(
                    "critic_rl_state", actor_aux["rl_state"]
                ).detach(),
                action=actions,
            )
            q_pi = torch.minimum(q1, q2)
            q_pi_mean = q_pi.mean().item()

        bc_loss = torch.nn.functional.mse_loss(actions, sampled_actions)

        actor_loss, actor_metrics = self.td3_algorithm.compose_actor_loss(q_pi, bc_loss)

        if recon_coef > 0.0 and "recon_loss" in actor_aux:
            recon_loss = actor_aux["recon_loss"]
            actor_loss = actor_loss + recon_coef * recon_loss
            actor_metrics["recon_loss"] = recon_loss.item()

        actor_metrics["bc_loss"] = bc_loss.item()
        actor_metrics["actor_ref_action_mse"] = torch.nn.functional.mse_loss(
            actions, ref_action
        ).item()
        actor_metrics["q_pi"] = q_pi_mean
        return actor_loss, actor_metrics

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    @Worker.timer("update_one_epoch")
    def update_one_epoch(self):
        global_batch_size_per_rank = (
            self.cfg.actor.global_batch_size // self._world_size
        )
        stage_name = self.td3_algorithm.apply_training_stage(self.update_step)
        tail_window_size = self._update_tail_curriculum_window()
        with self.worker_timer("sample"):
            balanced_cfg = (
                self.cfg.algorithm.get("balanced_return_sampling", None) or {}
            )
            fixed_debug_cfg = self.cfg.algorithm.get("fixed_debug_batch", None) or {}
            return_bin_cfg = self.cfg.algorithm.get("return_bin_sampling", None) or {}
            paired_stage_cfg = (
                self.cfg.algorithm.get("paired_stage_sampling", None) or {}
            )
            if bool(fixed_debug_cfg.get("enabled", False)) and hasattr(
                self.replay_buffer, "sample_fixed_balanced_mc_return"
            ):
                global_batch = self.replay_buffer.sample_fixed_balanced_mc_return(
                    global_batch_size_per_rank,
                    tail_window_size=tail_window_size,
                )
            elif bool(paired_stage_cfg.get("enabled", False)) and hasattr(
                self.replay_buffer, "sample_paired_stage_mc_return"
            ):
                global_batch = self.replay_buffer.sample_paired_stage_mc_return(
                    global_batch_size_per_rank,
                    tail_window_size=tail_window_size,
                )
            elif bool(return_bin_cfg.get("enabled", False)) and hasattr(
                self.replay_buffer, "sample_return_bins"
            ):
                global_batch = self.replay_buffer.sample_return_bins(
                    global_batch_size_per_rank,
                    bin_edges=list(
                        return_bin_cfg.get("bin_edges", [0.0, 0.25, 0.5, 0.75])
                    ),
                    bin_weights=list(return_bin_cfg.get("bin_weights", [])),
                    tail_window_size=tail_window_size,
                )
            elif bool(balanced_cfg.get("enabled", False)) and hasattr(
                self.replay_buffer, "sample_balanced_mc_return"
            ):
                global_batch = self.replay_buffer.sample_balanced_mc_return(
                    global_batch_size_per_rank,
                    tail_window_size=tail_window_size,
                )
            else:
                global_batch = next(self.buffer_dataloader_iter)

        micro_batches = split_dict_to_chunk(
            global_batch,
            global_batch_size_per_rank // self.cfg.actor.micro_batch_size,
        )
        for i, batch in enumerate(micro_batches):
            batch = put_tensor_device(batch, device=self.device)
            if self.enable_drq:
                drq.apply_drq(batch["curr_obs"], pad=4)
                drq.apply_drq(batch["next_obs"], pad=4)
            micro_batches[i] = batch

        actor_bc_only = bool(self.cfg.algorithm.get("stage_actor_bc_only", False))
        critic_updated = False
        if actor_bc_only:
            metrics_data = {
                "td3/critic_loss": 0.0,
                "critic/skipped": 1.0,
                "critic/tail_window_size": float(tail_window_size),
                "training/stage_coupled": float(stage_name == "coupled_td3"),
            }
        else:
            # Critic update
            self.critic_optimizer.zero_grad()
            all_critic_metrics = {}
            critic_losses = []
            for batch in micro_batches:
                loss, metrics = self.forward_critic(batch)
                (loss / self.gradient_accumulation).backward()
                critic_losses.append(loss.item())
                append_to_dict(all_critic_metrics, metrics)
            critic_grad_norm = self.model.clip_grad_norm_(
                max_norm=self.cfg.actor.critic_optim.clip_grad
            )
            self.critic_optimizer.step()
            self.qf_lr_scheduler.step()
            critic_updated = True

            metrics_data = {
                "td3/critic_loss": np.mean(critic_losses),
                "critic/lr": self.critic_optimizer.param_groups[0]["lr"],
                "critic/grad_norm": critic_grad_norm,
                "critic/tail_window_size": float(tail_window_size),
                "training/stage_coupled": float(stage_name == "coupled_td3"),
                **{f"critic/{k}": np.mean(v) for k, v in all_critic_metrics.items()},
            }

        # Actor update (policy delay)
        actor_updated = False
        freeze_actor = bool(self.cfg.algorithm.get("stage_freeze_actor", False))
        if (
            self.td3_algorithm.should_update_actor(self.update_step)
            and not freeze_actor
        ):
            self.actor_optimizer.zero_grad()
            all_actor_metrics = {}
            actor_losses = []
            for batch in micro_batches:
                loss, metrics = self.forward_actor(batch)
                (loss / self.gradient_accumulation).backward()
                actor_losses.append(loss.item())
                append_to_dict(all_actor_metrics, metrics)
            actor_grad_norm = self.model.clip_grad_norm_(
                max_norm=self.cfg.actor.optim.clip_grad
            )
            self.actor_optimizer.step()
            self.lr_scheduler.step()
            actor_updated = True
            metrics_data.update(
                {
                    "td3/actor_loss": np.mean(actor_losses),
                    "actor/lr": self.actor_optimizer.param_groups[0]["lr"],
                    "actor/grad_norm": actor_grad_norm,
                    **{f"actor/{k}": np.mean(v) for k, v in all_actor_metrics.items()},
                }
            )
        elif freeze_actor:
            metrics_data["actor/frozen"] = 1.0

        # Target soft update
        tau = self.td3_algorithm.get_target_update_tau(
            update_step=self.update_step,
            stage_actor_bc_only=self.cfg.algorithm.get("stage_actor_bc_only", False),
            stage_freeze_actor=self.cfg.algorithm.get("stage_freeze_actor", False),
            actor_updated=actor_updated,
            critic_updated=critic_updated,
        )
        if tau is not None and self.target_model_initialized:
            self.soft_update_target_model(tau=tau)

        return metrics_data

    def _update_tail_curriculum_window(self):
        cfg = self.cfg.algorithm.get("tail_curriculum", None)
        if cfg is None or not bool(cfg.get("enabled", False)):
            if hasattr(self.replay_buffer, "set_sample_tail_window_size"):
                self.replay_buffer.set_sample_tail_window_size(0)
            return 0

        start = max(1, int(cfg.get("start_window", 8)))
        end = int(cfg.get("end_window", 0))
        hold_steps = max(0, int(cfg.get("hold_steps", 0)))
        warmup_steps = max(1, int(cfg.get("warmup_steps", 200)))
        if end <= 0:
            end = max(
                int(info.get("num_samples", start))
                for info in self.replay_buffer._trajectory_index.values()
            )
        curriculum_step = max(0, int(self.update_step) - hold_steps)
        progress = min(1.0, float(curriculum_step) / float(warmup_steps))
        window = int(round(start + progress * (end - start)))
        window = max(1, window)
        self.replay_buffer.set_sample_tail_window_size(window)
        return window

    # ------------------------------------------------------------------
    # Trajectory reception
    # ------------------------------------------------------------------

    def add_trajectories(self, trajectories: list):
        self.replay_buffer.add_trajectories(trajectories)

    async def recv_rollout_trajectories(self, input_channel: Channel):
        clear_memory(sync=False)
        send_num = self._component_placement.get_world_size("env") * self.stage_num
        recv_num = self._component_placement.get_world_size("actor")
        split_num = compute_split_num(send_num, recv_num)

        recv_list = []
        for _ in range(split_num):
            trajectory: Trajectory = await input_channel.get(async_op=True).async_wait()
            recv_list.append(trajectory)

        self.replay_buffer.add_trajectories(recv_list)

        if self.demo_buffer is not None:
            intervene_list = []
            for traj in recv_list:
                trajs = traj.extract_intervene_traj()
                if trajs is not None:
                    intervene_list.extend(trajs)
            if intervene_list:
                self.demo_buffer.add_trajectories(intervene_list)

    # ------------------------------------------------------------------
    # Main training entry point
    # ------------------------------------------------------------------

    def process_train_metrics(self, metrics):
        replay_buffer_stats = {
            f"replay_buffer/{k}": v for k, v in self.replay_buffer.get_stats().items()
        }
        append_to_dict(metrics, replay_buffer_stats)

        mean_metric_dict = {}
        for key, value in metrics.items():
            if isinstance(value, list) and value:
                cpu_values = [
                    v.detach().cpu().item() if isinstance(v, torch.Tensor) else v
                    for v in value
                ]
                mean_metric_dict[key] = np.mean(cpu_values)
            else:
                mean_metric_dict[key] = (
                    value.detach().cpu().item()
                    if isinstance(value, torch.Tensor)
                    else value
                )
        return all_reduce_dict(mean_metric_dict, op=torch.distributed.ReduceOp.AVG)

    @Worker.timer("run_training")
    def run_training(self):
        min_buffer_size = self.cfg.algorithm.replay_buffer.get("min_buffer_size", 100)
        if not self.replay_buffer.is_ready(min_buffer_size):
            self.log_on_first_rank(
                f"Replay buffer size {len(self.replay_buffer)} < {min_buffer_size}, skipping"
            )
            return {}

        train_actor_steps = max(
            min_buffer_size, self.cfg.algorithm.get("train_actor_steps", 0)
        )
        _ = self.replay_buffer.is_ready(train_actor_steps)  # logged internally

        assert (
            self.cfg.actor.global_batch_size
            % (self.cfg.actor.micro_batch_size * self._world_size)
            == 0
        )
        self.gradient_accumulation = (
            self.cfg.actor.global_batch_size
            // self.cfg.actor.micro_batch_size
            // self._world_size
        )

        self.model.train()
        metrics = {}
        for _ in range(self.cfg.algorithm.get("update_epoch", 1)):
            append_to_dict(metrics, self.update_one_epoch())
            self.update_step += 1

        mean_metrics = self.process_train_metrics(metrics)
        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()
        return mean_metrics

    def compute_advantages_and_returns(self):
        return {}

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, save_base_path, step):
        if self.cfg.runner.get("actor_critic_only_checkpoint", True):
            self._save_actor_critic_checkpoint(save_base_path, step)
        else:
            self._strategy.save_checkpoint(
                model=self.model,
                optimizers=[self.actor_optimizer, self.critic_optimizer],
                lr_schedulers=[self.lr_scheduler, self.qf_lr_scheduler],
                save_path=save_base_path,
                checkpoint_format=(
                    "local_shard"
                    if self.cfg.actor.fsdp_config.use_orig_params
                    else "dcp"
                ),
            )

            target_save_path = os.path.join(
                save_base_path, "td3_components/target_model"
            )
            os.makedirs(target_save_path, exist_ok=True)
            target_state_dict = self._strategy.get_model_state_dict(
                self.target_model, cpu_offload=False, full_state_dict=True
            )
            torch.save(
                target_state_dict,
                os.path.join(target_save_path, f"checkpoint_rank_{self._rank}.pt"),
            )

            buffer_save_path = os.path.join(
                save_base_path, f"td3_components/replay_buffer/rank_{self._rank}"
            )
            self.replay_buffer.save_checkpoint(buffer_save_path)

        if self.cfg.runner.get("offline_validation_visualization", None):
            self._save_validation_visualizations(save_base_path, step)

    def _checkpoint_key(self, name, prefixes):
        clean_name = self._canonical_param_name(name)
        for prefix in prefixes:
            if clean_name.startswith(prefix):
                return clean_name
            marker = f".{prefix}"
            if marker in clean_name:
                return clean_name.split(marker, 1)[1]
        return None

    def _filtered_named_tensors(self, model, prefixes):
        state = {}
        for name, tensor in model.named_parameters():
            checkpoint_key = self._checkpoint_key(name, prefixes)
            if checkpoint_key is not None:
                state[checkpoint_key] = tensor.detach().cpu().clone()
        for name, tensor in model.named_buffers():
            checkpoint_key = self._checkpoint_key(name, prefixes)
            if checkpoint_key is not None:
                state[checkpoint_key] = tensor.detach().cpu().clone()
        return state

    def _load_filtered_named_tensors(self, model, state):
        if not state:
            return
        state = self._normalize_actor_critic_state_keys(state)
        params = {
            self._canonical_param_name(name): param
            for name, param in model.named_parameters()
        }
        buffers = {
            self._canonical_param_name(name): tensor
            for name, tensor in model.named_buffers()
        }
        with torch.no_grad():
            for name, value in state.items():
                target = params.get(name, buffers.get(name, None))
                if target is not None:
                    target.copy_(value.to(device=target.device, dtype=target.dtype))

    def _normalize_actor_critic_state_keys(self, state):
        remapped = {}
        prefix_pairs = (
            ("target_actor_head.", "actor_head."),
            ("target_critic_head_1.", "critic_head_1."),
            ("target_critic_head_2.", "critic_head_2."),
            ("target_critic_rl_token_encoder.", "critic_rl_token_encoder."),
        )
        for name, value in state.items():
            out_name = name
            for old_prefix, new_prefix in prefix_pairs:
                if name.startswith(old_prefix):
                    out_name = new_prefix + name[len(old_prefix) :]
                    break
            remapped[out_name] = value
        return remapped

    def _save_actor_critic_checkpoint(self, save_base_path, step):
        os.makedirs(save_base_path, exist_ok=True)
        online_prefixes = ("actor_head.", "critic_head_1.", "critic_head_2.")
        if bool(self.cfg.algorithm.get("critic_train_rl_token_encoder", False)):
            if bool(
                self.cfg.actor.model.get("critic_separate_rl_token_encoder", False)
            ):
                encoder_prefix = "critic_rl_token_encoder."
            else:
                encoder_prefix = "rl_token_autoencoder.encoder."
            online_prefixes = (
                encoder_prefix,
                "actor_head.",
                "critic_head_1.",
                "critic_head_2.",
            )
        target_prefixes = online_prefixes
        payload = {
            "format": "rlinf_td3_actor_critic_only",
            "global_step": int(step),
            "update_step": int(self.update_step),
            "model": self._filtered_named_tensors(self.model, online_prefixes),
            "target_model": self._filtered_named_tensors(
                self.target_model, target_prefixes
            ),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "actor_lr_scheduler": self.lr_scheduler.state_dict()
            if self.lr_scheduler is not None
            else None,
            "critic_lr_scheduler": self.qf_lr_scheduler.state_dict()
            if self.qf_lr_scheduler is not None
            else None,
            "notes": (
                "Contains only actor_head and critic_head_* weights plus optimizer "
                "state. target_model stores EMA copies under the same head names. "
                "VLA backbone and RL token autoencoder are intentionally excluded."
            ),
        }
        torch.save(payload, os.path.join(save_base_path, "actor_critic.pt"))
        save_json(
            os.path.join(save_base_path, "checkpoint_summary.json"),
            {
                "global_step": int(step),
                "update_step": int(self.update_step),
                "model_keys": sorted(payload["model"].keys()),
                "target_model_keys": sorted(payload["target_model"].keys()),
                "actor_optimizer_param_groups": len(
                    payload["actor_optimizer"].get("param_groups", [])
                ),
                "critic_optimizer_param_groups": len(
                    payload["critic_optimizer"].get("param_groups", [])
                ),
            },
        )

    def _save_validation_visualizations(self, save_base_path, step):
        viz_cfg = self.cfg.runner.offline_validation_visualization
        if not bool(viz_cfg.get("enabled", True)):
            return

        data_paths = list(
            viz_cfg.get("data_paths", self.cfg.algorithm.get("offline_data_paths", []))
        )
        num_trajectories = int(viz_cfg.get("num_trajectories", 3))
        num_success_trajectories = int(viz_cfg.get("num_success_trajectories", 1))
        num_failure_trajectories = int(viz_cfg.get("num_failure_trajectories", 1))
        chunk_count = int(viz_cfg.get("num_chunks_per_trajectory", 6))
        urdf_path = str(viz_cfg.get("urdf_path", ""))
        if not urdf_path or not os.path.isfile(urdf_path):
            self.log_on_first_rank(
                f"Skipping validation visualization: invalid urdf_path={urdf_path!r}"
            )
            return

        out_root = os.path.join(save_base_path, "validation_visualizations")
        os.makedirs(out_root, exist_ok=True)
        if viz_cfg.get("split_success_failure", False):
            trajectories = [
                ("success", item)
                for item in load_validation_trajectories(
                    data_paths, num_success_trajectories, success=True
                )
            ] + [
                ("failure", item)
                for item in load_validation_trajectories(
                    data_paths, num_failure_trajectories, success=False
                )
            ]
        else:
            trajectories = [
                ("all", item)
                for item in load_validation_trajectories(data_paths, num_trajectories)
            ]
        kinematics = SimpleUrdfKinematics(
            urdf_path=urdf_path,
            base_link=str(viz_cfg.get("base_link", "base_link")),
        )

        min_q = np.array(
            viz_cfg.get(
                "min_qpos",
                [-2.618, 0.0, -2.967, -1.745, -1.22, -2.0944, 0.0],
            ),
            dtype=np.float64,
        )
        max_q = np.array(
            viz_cfg.get(
                "max_qpos",
                [2.618, 3.14, 0.0, 1.745, 1.22, 2.0944, 1.0],
            ),
            dtype=np.float64,
        )
        if min_q.shape[0] < 14:
            low = np.tile(min_q[:7], 2)
            high = np.tile(max_q[:7], 2)
        else:
            low = min_q[:14]
            high = max_q[:14]

        base_offsets = np.asarray(
            viz_cfg.get("dual_arm_base_offsets", [[0.0, 0.22, 0.0], [0.0, -0.22, 0.0]]),
            dtype=np.float64,
        )
        joint_action_mode = str(viz_cfg.get("joint_action_mode", "absolute"))
        delta_action_scale = float(viz_cfg.get("delta_action_scale", 0.01))
        tip_link = str(viz_cfg.get("tip_link", "gripper"))
        norm_stats_path = viz_cfg.get("action_norm_stats_path", None)
        action_mean = action_std = None
        if joint_action_mode == "normalized_delta":
            if norm_stats_path is None:
                raise ValueError(
                    "offline_validation_visualization.action_norm_stats_path is required "
                    "for normalized_delta visualization"
                )
            action_mean, action_std = load_action_norm_stats(
                str(norm_stats_path),
                float(viz_cfg.get("action_norm_std_floor", 1e-6)),
            )
        action_horizon = int(
            self.cfg.actor.model.get(
                "action_horizon", self.cfg.actor.model.num_action_chunks
            )
        )
        key_segment_start_map = load_key_segment_start_map(
            viz_cfg.get("key_segment_summary_path", None)
        )
        image_thumb_size = int(viz_cfg.get("image_thumb_size", 64))

        summary = {
            "global_step": int(step),
            "update_step": int(self.update_step),
            "urdf_path": urdf_path,
            "action_space": joint_action_mode,
            "delta_action_scale": delta_action_scale,
            "note": (
                "Actor/GT raw actions are in the training action space. For 3D FK, "
                "GT and actor chunks are converted back to absolute joint targets with "
                "the configured action transform. PT chunks are drawn directly from "
                "forward_inputs.env_action_absolute when that field is available. For "
                "normalized_delta, reconstruction follows OpenPI DeltaActions/AbsoluteActions: "
                "the 12 arm-joint dims are denormalized and added to curr_obs.states, "
                "while the two gripper dims remain absolute. The URDF is a single-arm "
                "Piper model reused for left/right arms with simple display offsets."
            ),
            "trajectories": [],
        }

        was_training = self.model.training
        self.model.eval()
        minimal_viz = bool(viz_cfg.get("minimal", False))
        timeline_only = bool(viz_cfg.get("timeline_only", False))
        for split_name, (traj_path, trajectory) in trajectories:
            traj_name = traj_path.stem
            out_dir = os.path.join(out_root, split_name, traj_name)
            os.makedirs(out_dir, exist_ok=True)
            result = evaluate_validation_trajectory(self, trajectory, chunk_count)
            chunk_indices = result["chunk_indices"]
            key_segment_start = key_segment_start_map.get(traj_path.name, None)
            train_mask = np.ones_like(chunk_indices, dtype=bool)
            if key_segment_start is not None:
                train_mask = chunk_indices >= int(key_segment_start)
            images = extract_three_view_images(
                trajectory,
                chunk_indices,
                thumb_size=image_thumb_size,
            )
            absolute_actions_all = None
            forward_inputs = trajectory.get("forward_inputs", {})
            if isinstance(forward_inputs, dict) and torch.is_tensor(
                forward_inputs.get("env_action_absolute", None)
            ):
                absolute_actions_all = np.asarray(
                    forward_inputs["env_action_absolute"][:, 0].float().cpu(),
                    dtype=np.float64,
                ).reshape(-1, action_horizon, 14)
            delta_anchor_mode = str(
                viz_cfg.get("delta_visualization_anchor", "robot_state")
            )
            pt_joint_chunks = []
            gt_reconstructed_joint_chunks = []
            actor_joint_chunks = []
            for chunk_id, start_state in enumerate(result["start_states"]):
                gt_reconstructed_chunk = actions_to_absolute_joint_targets(
                    result["gt_actions"][chunk_id],
                    start_state,
                    low,
                    high,
                    joint_action_mode,
                    delta_action_scale,
                    action_mean,
                    action_std,
                )
                gt_reconstructed_joint_chunks.append(gt_reconstructed_chunk)
                if absolute_actions_all is not None:
                    pt_chunk = absolute_actions_all[int(chunk_indices[chunk_id])].copy()
                else:
                    pt_chunk = gt_reconstructed_chunk.copy()
                pt_joint_chunks.append(pt_chunk)
                actor_anchor = start_state
                if (
                    joint_action_mode == "normalized_delta"
                    and delta_anchor_mode == "pt_action_first"
                ):
                    actor_anchor = pt_chunk[0]
                actor_joint_chunks.append(
                    actions_to_absolute_joint_targets(
                        result["actor_actions"][chunk_id],
                        actor_anchor,
                        low,
                        high,
                        joint_action_mode,
                        delta_action_scale,
                        action_mean,
                        action_std,
                    )
                )
            gt_reconstructed_joints = np.concatenate(
                gt_reconstructed_joint_chunks, axis=0
            )
            actor_joints = np.concatenate(actor_joint_chunks, axis=0)
            q_actor_dense = np.repeat(
                result["critic_q_actor"],
                action_horizon,
            )
            chunk_gaps = (
                np.diff(chunk_indices) if len(chunk_indices) > 1 else np.asarray([])
            )
            segment_lengths = None
            if len(chunk_gaps) > 0 and not np.all(chunk_gaps == 1):
                segment_lengths = [action_horizon] * len(chunk_indices)
            gt_reconstructed_tcp = tcp_trajectories_from_joint_targets(
                gt_reconstructed_joints, kinematics, base_offsets, tip_link=tip_link
            )
            actor_tcp = tcp_trajectories_from_joint_targets(
                actor_joints, kinematics, base_offsets, tip_link=tip_link
            )
            state_joints = np.concatenate(
                [
                    np.asarray(
                        trajectory["curr_obs"]["states"][:, 0].float().cpu(),
                        dtype=np.float64,
                    ),
                    np.asarray(
                        trajectory["next_obs"]["states"][-1:, 0].float().cpu(),
                        dtype=np.float64,
                    ),
                ],
                axis=0,
            )
            state_tcp = tcp_trajectories_from_joint_targets(
                np.clip(state_joints, low, high),
                kinematics,
                base_offsets,
                tip_link=tip_link,
            )
            pt_tcp_segments = [
                tcp_trajectories_from_joint_targets(
                    chunk, kinematics, base_offsets, tip_link=tip_link
                )
                for chunk in pt_joint_chunks
            ]
            gt_reconstructed_tcp_segments = [
                tcp_trajectories_from_joint_targets(
                    chunk, kinematics, base_offsets, tip_link=tip_link
                )
                for chunk in gt_reconstructed_joint_chunks
            ]
            actor_tcp_segments = [
                tcp_trajectories_from_joint_targets(
                    chunk, kinematics, base_offsets, tip_link=tip_link
                )
                for chunk in actor_joint_chunks
            ]
            gt_chunk_start_error = np.asarray(
                [
                    np.max(np.abs(chunk[0] - state))
                    for chunk, state in zip(
                        gt_reconstructed_joint_chunks, result["start_states"]
                    )
                ],
                dtype=np.float64,
            )
            pt_vs_gt_reconstructed_abs = np.asarray(pt_joint_chunks) - np.asarray(
                gt_reconstructed_joint_chunks
            )
            pt_vs_gt_reconstructed_max_abs = np.max(
                np.abs(pt_vs_gt_reconstructed_abs).reshape(len(pt_joint_chunks), -1),
                axis=1,
            )
            pt_vs_gt_reconstructed_mean_abs = np.mean(
                np.abs(pt_vs_gt_reconstructed_abs).reshape(len(pt_joint_chunks), -1),
                axis=1,
            )
            actor_vs_gt_reconstructed_abs = np.asarray(actor_joint_chunks) - np.asarray(
                gt_reconstructed_joint_chunks
            )
            actor_vs_gt_reconstructed_mse = np.mean(
                np.square(actor_vs_gt_reconstructed_abs).reshape(
                    len(actor_joint_chunks), -1
                ),
                axis=1,
            )
            gt_boundary_joints = np.stack(
                [chunk[0] for chunk in gt_reconstructed_joint_chunks], axis=0
            )
            actor_boundary_joints = np.stack(
                [chunk[0] for chunk in actor_joint_chunks], axis=0
            )
            gt_boundary_tcp = tcp_trajectories_from_joint_targets(
                gt_boundary_joints, kinematics, base_offsets, tip_link=tip_link
            )
            actor_boundary_tcp = tcp_trajectories_from_joint_targets(
                actor_boundary_joints, kinematics, base_offsets, tip_link=tip_link
            )
            full_pt_boundary_joints = []
            raw_actions_all = np.asarray(
                trajectory["actions"][:, 0].float().cpu(), dtype=np.float64
            )
            raw_actions_all = raw_actions_all.reshape(
                raw_actions_all.shape[0], action_horizon, -1
            )
            curr_states_all = np.asarray(
                trajectory["curr_obs"]["states"][:, 0].float().cpu(), dtype=np.float64
            )
            if absolute_actions_all is not None:
                full_pt_joint_chunks = [chunk.copy() for chunk in absolute_actions_all]
                full_pt_boundary_joints = [
                    chunk[0].copy() for chunk in full_pt_joint_chunks
                ]
            else:
                for raw_chunk, start_state in zip(raw_actions_all, curr_states_all):
                    abs_chunk = actions_to_absolute_joint_targets(
                        raw_chunk,
                        start_state,
                        low,
                        high,
                        joint_action_mode,
                        delta_action_scale,
                        action_mean,
                        action_std,
                    )
                    full_pt_boundary_joints.append(abs_chunk[0])
                full_pt_joint_chunks = [
                    actions_to_absolute_joint_targets(
                        raw_chunk,
                        start_state,
                        low,
                        high,
                        joint_action_mode,
                        delta_action_scale,
                        action_mean,
                        action_std,
                    )
                    for raw_chunk, start_state in zip(raw_actions_all, curr_states_all)
                ]
            full_gt_reconstructed_joint_chunks = [
                actions_to_absolute_joint_targets(
                    raw_chunk,
                    start_state,
                    low,
                    high,
                    joint_action_mode,
                    delta_action_scale,
                    action_mean,
                    action_std,
                )
                for raw_chunk, start_state in zip(raw_actions_all, curr_states_all)
            ]
            full_pt_tcp_segments = [
                tcp_trajectories_from_joint_targets(
                    chunk, kinematics, base_offsets, tip_link=tip_link
                )
                for chunk in full_pt_joint_chunks
            ]
            full_gt_reconstructed_tcp_segments = [
                tcp_trajectories_from_joint_targets(
                    chunk, kinematics, base_offsets, tip_link=tip_link
                )
                for chunk in full_gt_reconstructed_joint_chunks
            ]
            full_pt_boundary_tcp = tcp_trajectories_from_joint_targets(
                np.stack(full_pt_boundary_joints, axis=0),
                kinematics,
                base_offsets,
                tip_link=tip_link,
            )

            if not timeline_only:
                if not minimal_viz:
                    plot_trajectory_3d(
                        os.path.join(out_dir, "trajectory_3d.png"),
                        gt_reconstructed_tcp,
                        actor_tcp,
                        q_actor_dense,
                        segment_lengths=segment_lengths,
                    )
                    plot_pt_state_trajectory_3d(
                        os.path.join(out_dir, "pt_state_trajectory_3d.png"),
                        state_tcp,
                        full_pt_boundary_tcp,
                    )
                    plot_action_chunk_segments_3d(
                        os.path.join(out_dir, "pt_action_chunks_3d.png"),
                        state_tcp,
                        full_pt_tcp_segments,
                        np.arange(len(full_pt_tcp_segments), dtype=np.int64),
                    )
                    plot_boundary_trajectory_3d(
                        os.path.join(out_dir, "trajectory_3d_state_chunks.png"),
                        state_tcp,
                        gt_boundary_tcp,
                        actor_boundary_tcp,
                        result["chunk_indices"],
                        result["critic_q_data"],
                        result["critic_q_actor"],
                    )
                    plot_action_chunk_segments_3d(
                        os.path.join(out_dir, "selected_action_chunks_3d.png"),
                        state_tcp,
                        pt_tcp_segments,
                        result["chunk_indices"],
                        actor_tcp_segments=actor_tcp_segments,
                        q_data=result["critic_q_data"],
                        q_actor=result["critic_q_actor"],
                    )
                plot_action_chunk_triplet_segments_3d(
                    os.path.join(out_dir, "selected_action_chunks_triplet_3d.png"),
                    state_tcp,
                    pt_tcp_segments,
                    gt_reconstructed_tcp_segments,
                    actor_tcp_segments,
                )
                plot_action_chunk_triplet_segments_3d(
                    os.path.join(
                        out_dir, "selected_pt_gt_actor_action_chunks_global_3d.png"
                    ),
                    state_tcp,
                    pt_tcp_segments,
                    gt_reconstructed_tcp_segments,
                    actor_tcp_segments,
                )
                plot_pt_gt_reconstruction_segments_3d(
                    os.path.join(out_dir, "pt_vs_gt_reconstruction_3d.png"),
                    state_tcp,
                    pt_tcp_segments,
                    gt_reconstructed_tcp_segments,
                    float(np.max(pt_vs_gt_reconstructed_max_abs)),
                    float(np.max(pt_vs_gt_reconstructed_mean_abs)),
                )
            full_pt_vs_gt_abs = np.asarray(full_pt_joint_chunks) - np.asarray(
                full_gt_reconstructed_joint_chunks
            )
            full_pt_vs_gt_max_abs = float(np.max(np.abs(full_pt_vs_gt_abs)))
            full_pt_vs_gt_mean_abs = float(np.mean(np.abs(full_pt_vs_gt_abs)))
            if not timeline_only:
                plot_pt_gt_reconstruction_segments_3d(
                    os.path.join(out_dir, "full_pt_vs_gt_reconstruction_3d.png"),
                    state_tcp,
                    full_pt_tcp_segments,
                    full_gt_reconstructed_tcp_segments,
                    full_pt_vs_gt_max_abs,
                    full_pt_vs_gt_mean_abs,
                )
                plot_action_chunk_segments_3d(
                    os.path.join(out_dir, "full_pt_action_chunks_global_3d.png"),
                    state_tcp,
                    full_pt_tcp_segments,
                    np.arange(len(full_pt_tcp_segments), dtype=np.int64),
                )
                if not minimal_viz:
                    plot_trajectory_3d_with_state(
                        os.path.join(out_dir, "trajectory_3d_forecast_chunks.png"),
                        state_tcp,
                        gt_reconstructed_tcp_segments,
                        actor_tcp_segments,
                        result["chunk_indices"],
                        result["critic_q_data"],
                        result["critic_q_actor"],
                    )
            plot_q_values(
                os.path.join(out_dir, "q_values.png"),
                result["critic_q_data"],
                result["critic_q_actor"],
                result["action_mse"],
                chunk_indices=result["chunk_indices"],
                mc_return=result["mc_return"],
            )
            plot_critic_timeline_with_images(
                os.path.join(out_dir, "critic_timeline_with_images.png"),
                result["chunk_indices"],
                result["critic_q_data"],
                result["critic_q_actor"],
                result["mc_return"],
                rewards=result["sample_rewards"],
                critic_mc_mse=result.get("critic_mc_mse"),
                images=images,
                train_mask=train_mask,
                title=(
                    f"{split_name}/{traj_name} critic timeline"
                    + (
                        f" (trained chunks >= {int(key_segment_start)})"
                        if key_segment_start is not None
                        else ""
                    )
                ),
            )
            if not timeline_only:
                plot_action_mse_heatmaps(
                    os.path.join(out_dir, "action_mse_heatmaps.png"),
                    result["action_mse_matrix"],
                    result["chunk_indices"],
                    result["action_mse"],
                )
            metrics = {
                "split": split_name,
                "trajectory_file": str(traj_path),
                "chunk_indices": result["chunk_indices"],
                "critic_q_data": result["critic_q_data"],
                "critic_q_actor": result["critic_q_actor"],
                "critic_q_data_q1": result.get("critic_q_data_q1"),
                "critic_q_data_q2": result.get("critic_q_data_q2"),
                "critic_q_actor_q1": result.get("critic_q_actor_q1"),
                "critic_q_actor_q2": result.get("critic_q_actor_q2"),
                "mc_return": result["mc_return"],
                "all_mc_return": result["all_mc_return"],
                "mc_discount_per_chunk": result["mc_discount_per_chunk"],
                "action_mse": result["action_mse"],
                "action_mse_matrix": result["action_mse_matrix"],
                "critic_mc_mse": result.get("critic_mc_mse"),
                "critic_train_mask": train_mask,
                "key_segment_start_chunk": (
                    None if key_segment_start is None else int(key_segment_start)
                ),
                "start_states": result["start_states"],
                "gt_actions": result["gt_actions"],
                "actor_actions": result["actor_actions"],
                "pt_selected_absolute_action_chunks": np.asarray(pt_joint_chunks),
                "gt_reconstructed_absolute_action_chunks": np.asarray(
                    gt_reconstructed_joint_chunks
                ),
                "actor_absolute_action_chunks": np.asarray(actor_joint_chunks),
                "full_pt_absolute_action_chunks": np.asarray(full_pt_joint_chunks),
                "gt_chunk0_minus_state_max_abs": gt_chunk_start_error,
                "pt_vs_gt_reconstructed_max_abs": pt_vs_gt_reconstructed_max_abs,
                "pt_vs_gt_reconstructed_mean_abs": pt_vs_gt_reconstructed_mean_abs,
                "full_pt_vs_gt_reconstructed_max_abs": full_pt_vs_gt_max_abs,
                "full_pt_vs_gt_reconstructed_mean_abs": full_pt_vs_gt_mean_abs,
                "actor_vs_gt_reconstructed_mse_exec_space": actor_vs_gt_reconstructed_mse,
                "delta_visualization_anchor": delta_anchor_mode,
                "pt_action_source": (
                    "forward_inputs.env_action_absolute"
                    if absolute_actions_all is not None
                    else "actions + configured transform"
                ),
                "raw_gt_action_min": float(result["gt_actions"].min()),
                "raw_gt_action_max": float(result["gt_actions"].max()),
                "raw_actor_action_min": float(result["actor_actions"].min()),
                "raw_actor_action_max": float(result["actor_actions"].max()),
                "sample_rewards": result["sample_rewards"],
                "sample_terminations": result["sample_terminations"],
            }
            save_json(os.path.join(out_dir, "metrics.json"), metrics)
            summary["trajectories"].append(metrics)

        if was_training:
            self.model.train()
        save_json(os.path.join(out_root, "summary.json"), summary)

    def _reset_td3_critic_state(self):
        policy = self.model.module if hasattr(self.model, "module") else self.model
        target_policy = (
            self.target_model.module
            if hasattr(self.target_model, "module")
            else self.target_model
        )
        for maybe_policy in (policy, target_policy):
            for attr in ("critic_head_1", "critic_head_2"):
                head = getattr(maybe_policy, attr, None)
                if head is not None:
                    head.apply(self._reset_linear_module)
            critic_encoder = getattr(maybe_policy, "critic_rl_token_encoder", None)
            actor_encoder = getattr(
                getattr(maybe_policy, "rl_token_autoencoder", None),
                "encoder",
                None,
            )
            if critic_encoder is not None and actor_encoder is not None:
                critic_encoder.load_state_dict(actor_encoder.state_dict())
            if hasattr(maybe_policy, "target_critic_head_1") and hasattr(
                maybe_policy, "critic_head_1"
            ):
                maybe_policy.target_critic_head_1.load_state_dict(
                    maybe_policy.critic_head_1.state_dict()
                )
            if hasattr(maybe_policy, "target_critic_head_2") and hasattr(
                maybe_policy, "critic_head_2"
            ):
                maybe_policy.target_critic_head_2.load_state_dict(
                    maybe_policy.critic_head_2.state_dict()
                )
            if (
                hasattr(maybe_policy, "target_critic_rl_token_encoder")
                and maybe_policy.target_critic_rl_token_encoder is not None
                and critic_encoder is not None
            ):
                maybe_policy.target_critic_rl_token_encoder.load_state_dict(
                    critic_encoder.state_dict()
                )
        if hasattr(self, "_target_shadow_f32"):
            self._target_shadow_f32 = {
                name: param.detach().float().clone()
                for name, param in self.target_model.named_parameters()
            }
        if self.critic_optimizer is not None:
            self.critic_optimizer.state.clear()
        if self.qf_lr_scheduler is not None:
            self.qf_lr_scheduler = self.build_lr_scheduler(
                self.critic_optimizer, self.cfg.actor.critic_optim
            )

    def _apply_configured_optimizer_lrs(self):
        if self.actor_optimizer is not None:
            actor_lr = float(self.cfg.actor.optim.lr)
            for group in self.actor_optimizer.param_groups:
                group["lr"] = actor_lr
        if self.critic_optimizer is not None:
            critic_lr = float(self.cfg.actor.critic_optim.lr)
            for group in self.critic_optimizer.param_groups:
                group["lr"] = critic_lr

    @staticmethod
    def _reset_linear_module(module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.kaiming_uniform_(module.weight, a=5**0.5)
            if module.bias is not None:
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1 / (fan_in**0.5) if fan_in > 0 else 0
                torch.nn.init.uniform_(module.bias, -bound, bound)

    def load_checkpoint(self, load_base_path):
        reset_critic = bool(self.cfg.algorithm.get("reset_critic_on_resume", False))
        actor_critic_path = os.path.join(load_base_path, "actor_critic.pt")
        if os.path.isfile(actor_critic_path):
            map_location = self.device
            if isinstance(map_location, int):
                map_location = (
                    torch.device(f"cuda:{map_location}")
                    if torch.cuda.is_available()
                    else torch.device("cpu")
                )
            payload = torch.load(actor_critic_path, map_location=map_location)
            self._load_filtered_named_tensors(self.model, payload.get("model", {}))
            if "target_model" in payload:
                self._load_filtered_named_tensors(
                    self.target_model, payload.get("target_model", {})
                )
            if payload.get("actor_optimizer") is not None:
                self.actor_optimizer.load_state_dict(payload["actor_optimizer"])
            if payload.get("critic_optimizer") is not None and not reset_critic:
                self.critic_optimizer.load_state_dict(payload["critic_optimizer"])
            if payload.get("actor_lr_scheduler") is not None:
                self.lr_scheduler.load_state_dict(payload["actor_lr_scheduler"])
            if payload.get("critic_lr_scheduler") is not None and not reset_critic:
                self.qf_lr_scheduler.load_state_dict(payload["critic_lr_scheduler"])
            if reset_critic:
                self._reset_td3_critic_state()
                self.log_on_first_rank(
                    "Reset TD3 critic heads after loading checkpoint."
                )
            self._apply_configured_optimizer_lrs()
            if bool(self.cfg.algorithm.get("reset_update_step_on_resume", False)):
                self.update_step = 0
            else:
                self.update_step = int(payload.get("update_step", 0))
            return

        self._strategy.load_checkpoint(
            model=self.model,
            optimizers=[self.actor_optimizer, self.critic_optimizer],
            lr_schedulers=[self.lr_scheduler, self.qf_lr_scheduler],
            load_path=load_base_path,
            checkpoint_format=(
                "local_shard" if self.cfg.actor.fsdp_config.use_orig_params else "dcp"
            ),
        )

        target_load_path = os.path.join(load_base_path, "td3_components/target_model")
        target_state_dict = torch.load(
            os.path.join(target_load_path, f"checkpoint_rank_{self._rank}.pt"),
            map_location=self.device,
        )
        self.target_model.load_state_dict(target_state_dict)
        if reset_critic:
            self._reset_td3_critic_state()
            self.log_on_first_rank("Reset TD3 critic heads after loading checkpoint.")

        buffer_load_path = os.path.join(
            load_base_path, f"td3_components/replay_buffer/rank_{self._rank}"
        )
        self.replay_buffer.load_checkpoint(buffer_load_path)
