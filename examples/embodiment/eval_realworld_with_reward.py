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

"""Real-world eval with model inference, human-in-the-loop intervention,
and post-rollout keyboard reward labeling.

Supports:
  - Pure pi0.5 rollout  (actor.model.model_type = openpi)
  - pi0.5 + RLToken + TD3 rollout  (actor.model.model_type = openpi_rl_token)

Usage:
    bash examples/embodiment/run_realworld_eval_with_reward.sh \
        realworld_piper_peginsertion_pi05_eval_with_reward
"""

import os
import time

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, open_dict
from tqdm import tqdm

from rlinf.data.embodied_io_struct import ChunkStepResult, EmbodiedRolloutResult
from rlinf.data.replay_buffer import TrajectoryReplayBuffer
from rlinf.envs.realworld.common.keyboard.keyboard_listener import KeyboardListener
from rlinf.envs.realworld.realworld_env import RealWorldEnv
from rlinf.models import get_model
from rlinf.scheduler import Cluster, ComponentPlacement, Worker


class EvalWithRewardCollector(Worker):
    def __init__(self, cfg: DictConfig):
        super().__init__()

        self.cfg = cfg
        self.num_rollouts = cfg.runner.num_data_episodes
        self.num_action_chunks = cfg.actor.model.num_action_chunks
        self.action_dim = cfg.actor.model.action_dim

        # Build model (same path as MultiStepRolloutWorker.init_worker).
        # Auto-select model_type: openpi_rl_token when rl_token_path is set.
        rollout_model_cfg = cfg.actor.model
        with open_dict(rollout_model_cfg):
            rollout_model_cfg.model_path = cfg.rollout.model.model_path
            rollout_model_cfg.precision = cfg.rollout.model.precision
            if rollout_model_cfg.get("rl_token_path", None):
                rollout_model_cfg.model_type = "openpi_rl_token"
            else:
                rollout_model_cfg.model_type = "openpi"
        self.model = get_model(rollout_model_cfg)
        self.model.eval()

        self.env = RealWorldEnv(
            cfg.env.eval,
            num_envs=1,
            seed_offset=0,
            total_num_processes=1,
            worker_info=self.worker_info,
        )

        buffer_path = os.path.join(cfg.runner.logger.log_path, "demos")
        self.log_info(f"Initializing ReplayBuffer at: {buffer_path}")
        self.buffer = TrajectoryReplayBuffer(
            seed=cfg.seed if hasattr(cfg, "seed") else 1234,
            enable_cache=False,
            auto_save=True,
            auto_save_path=buffer_path,
            trajectory_format="pt",
        )

        self._keyboard = KeyboardListener()

    def _process_obs(self, obs: dict) -> dict:
        if not self.cfg.runner.get("record_task_description", False):
            obs.pop("task_descriptions", None)
        ret = {}
        for key, val in obs.items():
            if isinstance(val, np.ndarray):
                val = torch.from_numpy(val)
            val = val.cpu()
            ret["main_images" if key == "images" else key] = val.clone()
        return ret

    def _wait_enter(self, rollout_idx: int):
        self.log_info(
            f"[Rollout {rollout_idx + 1}/{self.num_rollouts}] Press ENTER to start..."
        )
        with open("/dev/tty") as tty:
            tty.readline()

    def _get_keyboard_reward(self) -> tuple[float, bool]:
        """Block until a/b/c is pressed. Returns (reward, done)."""
        self.log_info(
            "Give reward: [a] fail(-1, done)  [b] neutral(0)  [c] success(1, done)"
        )
        while True:
            key = self._keyboard.get_key()
            if key == "a":
                return -1.0, True
            if key == "b":
                return 0.0, False
            if key == "c":
                return 1.0, True
            time.sleep(0.02)

    def run(self):
        saved = 0
        progress_bar = tqdm(range(self.num_rollouts), desc="Eval rollouts with reward:")

        for rollout_idx in range(self.num_rollouts):
            self._wait_enter(rollout_idx)

            obs, _ = self.env.reset()
            current_obs = self._process_obs(obs)
            rollout = EmbodiedRolloutResult(
                max_episode_length=self.cfg.env.eval.max_episode_steps,
            )

            max_chunk_steps = (
                self.cfg.env.eval.max_steps_per_rollout_epoch // self.num_action_chunks
            )

            done = False
            for _ in range(max_chunk_steps):
                with torch.no_grad():
                    actions, result = self.model.predict_action_batch(
                        env_obs=current_obs, mode="eval", compute_values=False
                    )

                if isinstance(actions, np.ndarray):
                    actions = torch.from_numpy(actions)

                # actions: [1, num_action_chunks * action_dim]
                chunk_actions = (
                    actions.reshape(1, self.num_action_chunks, self.action_dim)
                    .cpu()
                    .numpy()
                )

                # Execute full chunk via env.chunk_step
                (
                    obs_list,
                    chunk_rewards,
                    chunk_terminations,
                    chunk_truncations,
                    infos_list,
                ) = self.env.chunk_step(chunk_actions)

                next_obs = obs_list[-1]
                next_obs_processed = self._process_obs(next_obs)

                # Flatten chunk signals to single step for trajectory
                reward_tensor = chunk_rewards.sum(dim=1, keepdim=True)  # [1, 1]
                terminated_tensor = chunk_terminations[:, -1:].bool()  # [1, 1]
                truncated_tensor = chunk_truncations[:, -1:].bool()  # [1, 1]
                done_tensor = terminated_tensor | truncated_tensor

                # Recover intervene_action if any
                action_tensor = torch.as_tensor(
                    actions.reshape(1, -1).cpu().numpy(), dtype=torch.float32
                )
                last_info = infos_list[-1]
                if "intervene_action" in last_info:
                    ia = last_info["intervene_action"]
                    if isinstance(ia, torch.Tensor):
                        action_tensor = ia.reshape(1, -1).float().cpu()

                step = ChunkStepResult(
                    actions=action_tensor,
                    rewards=reward_tensor.float(),
                    dones=done_tensor,
                    terminations=terminated_tensor,
                    truncations=truncated_tensor,
                    forward_inputs={"action": action_tensor},
                )
                rollout.append_step_result(step)
                rollout.append_transitions(
                    curr_obs=current_obs, next_obs=next_obs_processed
                )

                current_obs = next_obs_processed
                done = bool(done_tensor.any().item())
                if done:
                    break

            # Post-rollout: human assigns reward via keyboard
            keyboard_reward, _ = self._get_keyboard_reward()
            self.log_info(f"Keyboard reward: {keyboard_reward}")

            trajectory = rollout.to_trajectory()
            # Overwrite final step reward with keyboard label
            if trajectory.rewards is not None and len(trajectory.rewards) > 0:
                trajectory.rewards[-1] = torch.full_like(
                    trajectory.rewards[-1], keyboard_reward
                )
            # Mark all steps as human-labeled (mirrors collect_real_data.py)
            if trajectory.intervene_flags is not None:
                trajectory.intervene_flags = torch.ones_like(trajectory.intervene_flags)

            self.buffer.add_trajectories([trajectory])
            saved += 1
            progress_bar.update(1)
            self.log_info(
                f"Saved rollout {rollout_idx + 1} "
                f"(reward={keyboard_reward}, total saved: {saved})"
            )

        self.buffer.close()
        self.log_info(
            f"Finished. {saved} rollouts saved to: "
            f"{os.path.join(self.cfg.runner.logger.log_path, 'demos')}"
        )
        self.env.close()


@hydra.main(
    version_base="1.1",
    config_path="config",
    config_name="realworld_piper_peginsertion_pi05_eval_with_reward",
)
def main(cfg: DictConfig):
    cluster = Cluster(cluster_cfg=cfg.cluster)
    component_placement = ComponentPlacement(cfg, cluster)
    env_placement = component_placement.get_strategy("env")
    collector = EvalWithRewardCollector.create_group(cfg).launch(
        cluster, name=cfg.env.group_name, placement_strategy=env_placement
    )
    collector.run().wait()


if __name__ == "__main__":
    main()
