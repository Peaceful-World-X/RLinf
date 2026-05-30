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

        # actor_optimizer: all params except critic heads
        # critic_optimizer: critic_head_1 + critic_head_2
        param_filters = {"critic": ["critic_head_1", "critic_head_2"]}
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
        self.grad_scaler = self.build_grad_scaler(self.cfg.actor.fsdp_config.grad_scaler)

    def build_lr_schedulers(self):
        self.lr_scheduler = self.build_lr_scheduler(self.actor_optimizer, self.cfg.actor.optim)
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
            trajectory_format=self.cfg.algorithm.replay_buffer.get("trajectory_format", "pt"),
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
            self.cfg.algorithm, self.cfg.actor.policy_head
        )
        self.td3_algorithm.set_discount(self.cfg.algorithm.gamma)
        self.td3_algorithm.set_action_horizon(self.cfg.actor.model.action_horizon)
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
        action_horizon = self.cfg.actor.model.action_horizon
        action_dim = self.cfg.actor.model.action_dim
        return action.reshape(action.shape[0], action_horizon, action_dim)

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
            ref_action_dropout_p=0.0,
            use_target=False,
            compute_recon_loss=recon_coef > 0.0,
        )

        q1, q2 = self.model(
            forward_type=ForwardType.TD3_Q,
            rl_state=actor_aux["rl_state"].detach(),
            action=actions,
        )
        q_pi = torch.minimum(q1, q2)

        bc_loss = torch.nn.functional.mse_loss(actions, ref_action)

        actor_loss, actor_metrics = self.td3_algorithm.compose_actor_loss(q_pi, bc_loss)

        if recon_coef > 0.0 and "recon_loss" in actor_aux:
            recon_loss = actor_aux["recon_loss"]
            actor_loss = actor_loss + recon_coef * recon_loss
            actor_metrics["recon_loss"] = recon_loss.item()

        actor_metrics["bc_loss"] = bc_loss.item()
        actor_metrics["q_pi"] = q_pi.mean().item()
        return actor_loss, actor_metrics

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    @Worker.timer("update_one_epoch")
    def update_one_epoch(self):
        global_batch_size_per_rank = self.cfg.actor.global_batch_size // self._world_size
        with self.worker_timer("sample"):
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

        metrics_data = {
            "td3/critic_loss": np.mean(critic_losses),
            "critic/lr": self.critic_optimizer.param_groups[0]["lr"],
            "critic/grad_norm": critic_grad_norm,
            **{f"critic/{k}": np.mean(v) for k, v in all_critic_metrics.items()},
        }

        # Actor update (policy delay)
        actor_updated = False
        if self.td3_algorithm.should_update_actor(self.update_step):
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
            metrics_data.update({
                "td3/actor_loss": np.mean(actor_losses),
                "actor/lr": self.actor_optimizer.param_groups[0]["lr"],
                "actor/grad_norm": actor_grad_norm,
                **{f"actor/{k}": np.mean(v) for k, v in all_actor_metrics.items()},
            })

        # Target soft update
        tau = self.td3_algorithm.get_target_update_tau(
            update_step=self.update_step,
            stage_actor_bc_only=self.cfg.algorithm.get("stage_actor_bc_only", False),
            stage_freeze_actor=self.cfg.algorithm.get("stage_freeze_actor", False),
            actor_updated=actor_updated,
            critic_updated=True,
        )
        if tau is not None and self.target_model_initialized:
            self.soft_update_target_model(tau=tau)

        return metrics_data

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
            f"replay_buffer/{k}": v
            for k, v in self.replay_buffer.get_stats().items()
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
        self._strategy.save_checkpoint(
            model=self.model,
            optimizers=[self.actor_optimizer, self.critic_optimizer],
            lr_schedulers=[self.lr_scheduler, self.qf_lr_scheduler],
            save_path=save_base_path,
            checkpoint_format=(
                "local_shard" if self.cfg.actor.fsdp_config.use_orig_params else "dcp"
            ),
        )

        target_save_path = os.path.join(save_base_path, "td3_components/target_model")
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

    def load_checkpoint(self, load_base_path):
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

        buffer_load_path = os.path.join(
            load_base_path, f"td3_components/replay_buffer/rank_{self._rank}"
        )
        self.replay_buffer.load_checkpoint(buffer_load_path)
