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

import os

from torch.utils.data import DataLoader

from rlinf.algorithms.td3 import TD3Algorithm
from rlinf.data.embodied_buffer_dataset import (
    ReplayBufferDataset,
    replay_buffer_collate_fn,
)
from rlinf.data.replay_buffer import TrajectoryReplayBuffer
from rlinf.workers.actor.fsdp_td3_policy_worker import EmbodiedTD3FSDPPolicy


class OfflineTD3FSDPPolicy(EmbodiedTD3FSDPPolicy):
    """TD3 policy worker that trains on pre-collected offline .pt trajectory data."""

    def _inject_task_descriptions(self, obs):
        """Return obs dict with task_descriptions injected (batch-size-aware)."""
        if isinstance(obs, dict) and "task_descriptions" not in obs:
            task_desc = self.cfg.algorithm.get("task_descriptions", "")
            batch_size = next(v.shape[0] for v in obs.values() if hasattr(v, "shape"))
            obs = dict(obs)
            obs["task_descriptions"] = [task_desc] * batch_size
        return obs

    def build_visual_feat_fn(self, visual_latent):
        return self._inject_task_descriptions(visual_latent)

    def get_visual_input(self, obs):
        return obs

    def setup_td3_components(self):
        data_paths = list(self.cfg.algorithm.get("offline_data_paths", []))
        if not data_paths:
            raise ValueError(
                "algorithm.offline_data_paths must be a non-empty list of directories"
            )

        seed = self.cfg.actor.get("seed", 1234)
        self.replay_buffer = TrajectoryReplayBuffer(
            seed=seed,
            enable_cache=self.cfg.algorithm.replay_buffer.enable_cache,
            cache_size=self.cfg.algorithm.replay_buffer.cache_size,
            sample_window_size=self.cfg.algorithm.replay_buffer.sample_window_size,
            auto_save=False,
        )

        for path in data_paths:
            if not os.path.isdir(path):
                raise FileNotFoundError(f"offline_data_paths entry not found: {path}")
            self.replay_buffer.load_checkpoint(
                path,
                is_distributed=(self._world_size > 1),
                local_rank=self._rank,
                world_size=self._world_size,
            )
            self.log_on_first_rank(
                f"Loaded {self.replay_buffer.size} trajectories from {path}"
            )

        self.demo_buffer = None
        self.buffer_dataset = ReplayBufferDataset(
            replay_buffer=self.replay_buffer,
            demo_buffer=None,
            batch_size=self.cfg.actor.global_batch_size // self._world_size,
            min_replay_buffer_size=1,
            min_demo_buffer_size=0,
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
        self.td3_algorithm.set_action_horizon(self.cfg.actor.model.num_action_chunks)
        self.target_update_type = self.cfg.algorithm.get("target_update_type", "all")

    def run_training(self):
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
        metrics = self.update_one_epoch()
        self.update_step += 1
        metrics["__global_step"] = self.update_step
        return metrics
