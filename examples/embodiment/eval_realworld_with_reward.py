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

import json
import os
import time

os.environ.setdefault("RLINF_SKIP_ROS_CLEANUP", "1")

import hydra
import numpy as np
import torch
import torch.multiprocessing as mp
from omegaconf import DictConfig, OmegaConf, open_dict
from tqdm import tqdm

from rlinf.data.embodied_io_struct import ChunkStepResult, EmbodiedRolloutResult
from rlinf.data.replay_buffer import TrajectoryReplayBuffer
from rlinf.scheduler import Worker

mp.set_start_method("spawn", force=True)


class EvalWithRewardCollector(Worker):
    def __init__(self, cfg: DictConfig):
        super().__init__()

        from rlinf.envs.realworld.common.keyboard.keyboard_listener import (
            KeyboardListener,
        )
        from rlinf.envs.realworld.realworld_env import RealWorldEnv
        from rlinf.models import get_model

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

        self._keyboard = KeyboardListener()

        buffer_path = os.path.join(cfg.runner.logger.log_path, "demos")
        self.log_info(f"Initializing ReplayBuffer at: {buffer_path}")
        self.buffer = TrajectoryReplayBuffer(
            seed=cfg.seed if hasattr(cfg, "seed") else 1234,
            enable_cache=False,
            auto_save=True,
            auto_save_path=buffer_path,
            trajectory_format="pt",
        )

    def _process_obs(self, obs: dict) -> dict:
        if not self.cfg.runner.get("record_task_description", False):
            obs.pop("task_descriptions", None)
        ret = {}
        for key, val in obs.items():
            out_key = "main_images" if key == "images" else key
            if isinstance(val, list):
                ret[out_key] = list(val)
                continue
            if isinstance(val, np.ndarray):
                val = torch.from_numpy(val)
            val = val.cpu()
            ret[out_key] = val.clone()
        return ret

    def _wait_enter(self, rollout_idx: int):
        print(
            f"[Rollout {rollout_idx + 1}/{self.num_rollouts}] Press ENTER to start...",
            flush=True,
        )
        while not self._keyboard.consume_press("Key.enter"):
            time.sleep(0.05)

    def _execute_action_chunk(self, chunk_actions):
        eval_cfg = self.cfg.env.eval
        break_after_intervention = (
            eval_cfg.get("break_chunk_after_intervention", False)
            if hasattr(eval_cfg, "get")
            else getattr(eval_cfg, "break_chunk_after_intervention", False)
        )
        if not break_after_intervention:
            return self.env.chunk_step(chunk_actions)

        obs_list = []
        infos_list = []
        chunk_rewards = []
        raw_chunk_terminations = []
        raw_chunk_truncations = []
        raw_chunk_intervene_actions = []
        raw_chunk_intervene_flags = []

        was_intervening = False
        for step_idx in range(chunk_actions.shape[1]):
            actions = chunk_actions[:, step_idx]
            obs, reward, terminations, truncations, infos = self.env.step(
                actions, auto_reset=False
            )

            obs_list.append(obs)
            infos_list.append(infos)
            chunk_rewards.append(torch.as_tensor(reward))
            raw_chunk_terminations.append(torch.as_tensor(terminations))
            raw_chunk_truncations.append(torch.as_tensor(truncations))

            intervene_flag = infos.get("intervene_flag")
            current_intervening = False
            if intervene_flag is not None:
                intervene_flag = torch.as_tensor(intervene_flag, dtype=torch.bool)
                current_intervening = bool(intervene_flag.any().item())
                raw_chunk_intervene_flags.append(intervene_flag)
                if "intervene_action" in infos:
                    raw_chunk_intervene_actions.append(
                        torch.as_tensor(infos["intervene_action"])
                    )

            done = bool((raw_chunk_terminations[-1] | raw_chunk_truncations[-1]).any())
            intervention_released = was_intervening and not current_intervening
            if done or intervention_released:
                break
            was_intervening = was_intervening or current_intervening

        chunk_rewards = torch.stack(chunk_rewards, dim=1)
        chunk_terminations = torch.stack(raw_chunk_terminations, dim=1)
        chunk_truncations = torch.stack(raw_chunk_truncations, dim=1)

        infos_last = infos_list[-1] if infos_list else {}
        if raw_chunk_intervene_flags:
            infos_last["intervene_flag"] = torch.stack(raw_chunk_intervene_flags, dim=1)
            if raw_chunk_intervene_actions:
                infos_last["intervene_action"] = torch.stack(
                    raw_chunk_intervene_actions, dim=1
                ).reshape(chunk_actions.shape[0], -1)
            infos_list[-1] = infos_last

        return (
            obs_list,
            chunk_rewards,
            chunk_terminations,
            chunk_truncations,
            infos_list,
        )

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

                (
                    obs_list,
                    chunk_rewards,
                    chunk_terminations,
                    chunk_truncations,
                    infos_list,
                ) = self._execute_action_chunk(chunk_actions)

                next_obs = obs_list[-1]
                next_obs_processed = self._process_obs(next_obs)

                # Flatten chunk signals to single step for trajectory
                reward_tensor = chunk_rewards.sum(dim=1, keepdim=True)  # [1, 1]
                terminated_tensor = chunk_terminations[:, -1:].bool()  # [1, 1]
                truncated_tensor = chunk_truncations[:, -1:].bool()  # [1, 1]
                done_tensor = terminated_tensor | truncated_tensor

                action_tensor = torch.as_tensor(
                    actions.reshape(1, -1).cpu().numpy(), dtype=torch.float32
                )

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

            trajectory = rollout.to_trajectory()
            self.buffer.add_trajectories([trajectory])
            saved += 1
            progress_bar.update(1)
            self.log_info(f"Saved rollout {rollout_idx + 1} (total saved: {saved})")

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
def main(cfg: DictConfig) -> None:
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    from rlinf.scheduler import Cluster
    from rlinf.utils.placement import HybridComponentPlacement

    cluster = Cluster(cluster_cfg=cfg.cluster)
    component_placement = HybridComponentPlacement(cfg, cluster)
    env_placement = component_placement.get_strategy("env")
    collector = EvalWithRewardCollector.create_group(cfg).launch(
        cluster, name=cfg.env.group_name, placement_strategy=env_placement
    )
    collector.run().wait()


if __name__ == "__main__":
    main()
