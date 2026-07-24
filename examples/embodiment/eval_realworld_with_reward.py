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


def _to_cpu_tensor(value):
    if isinstance(value, np.ndarray):
        value = torch.from_numpy(value)
    if torch.is_tensor(value):
        return value.cpu().clone()
    return None


def _pad_substep_tensor(value: torch.Tensor, target_steps: int) -> torch.Tensor:
    if value.shape[1] >= target_steps:
        return value[:, :target_steps].contiguous()
    pad_steps = target_steps - value.shape[1]
    pad = value[:, -1:].expand(value.shape[0], pad_steps, *value.shape[2:])
    return torch.cat([value, pad], dim=1).contiguous()


def build_substep_forward_inputs(
    *,
    obs_list,
    chunk_actions,
    target_steps: int,
    chunk_valid_mask=None,
):
    """Build online-compatible per-executed-action observation fields."""
    if not torch.is_tensor(chunk_actions):
        chunk_actions = torch.as_tensor(chunk_actions)
    chunk_actions = chunk_actions.cpu().to(torch.float32)
    batch_size = int(chunk_actions.shape[0])
    executed_steps = min(len(obs_list), int(target_steps))

    if chunk_valid_mask is None:
        chunk_valid_mask = torch.zeros(batch_size, target_steps, dtype=torch.bool)
        if executed_steps > 0:
            chunk_valid_mask[:, :executed_steps] = True
    else:
        chunk_valid_mask = torch.as_tensor(chunk_valid_mask, dtype=torch.bool).cpu()
        if chunk_valid_mask.dim() == 1:
            chunk_valid_mask = chunk_valid_mask[None, :]
        if chunk_valid_mask.shape[1] < target_steps:
            pad = torch.zeros(
                chunk_valid_mask.shape[0],
                target_steps - chunk_valid_mask.shape[1],
                dtype=torch.bool,
            )
            chunk_valid_mask = torch.cat([chunk_valid_mask, pad], dim=1)
        chunk_valid_mask = chunk_valid_mask[:, :target_steps].contiguous()

    forward_inputs = {
        "chunk_valid_mask": chunk_valid_mask,
        "executed_env_action_absolute": chunk_actions.reshape(batch_size, -1),
    }

    for obs_key, out_key in (
        ("main_images", "substep_main_images"),
        ("extra_view_images", "substep_extra_view_images"),
        ("states", "substep_states"),
    ):
        values = []
        for obs in obs_list[:target_steps]:
            if not isinstance(obs, dict) or obs_key not in obs:
                continue
            value = _to_cpu_tensor(obs[obs_key])
            if value is not None:
                values.append(value)
        if values:
            stacked = torch.stack(values, dim=1)
            forward_inputs[out_key] = _pad_substep_tensor(stacked, target_steps)

    return forward_inputs


def apply_intervention_to_rollout_step(
    rollout: EmbodiedRolloutResult,
    infos_last: dict,
) -> None:
    """Apply human intervention action/flags to the last appended rollout step."""
    if not isinstance(infos_last, dict):
        return
    if "intervene_action" not in infos_last or "intervene_flag" not in infos_last:
        return

    intervene_action = (
        torch.as_tensor(infos_last["intervene_action"]).cpu().to(torch.float32)
    )
    intervene_flag = torch.as_tensor(infos_last["intervene_flag"]).cpu().to(torch.bool)
    if intervene_flag.dim() == 1:
        intervene_flag = intervene_flag[:, None]
    if intervene_action.numel() == 0 or intervene_flag.numel() == 0:
        return
    if not rollout.actions:
        return

    last_action = rollout.actions[-1]
    if not torch.is_tensor(last_action) or last_action.dim() != 2:
        return
    batch_size = int(last_action.shape[0])
    executed_chunks = int(intervene_flag.shape[1])
    if intervene_action.shape[0] != batch_size:
        return
    action_dim = int(intervene_action.shape[-1] // max(executed_chunks, 1))
    if action_dim <= 0:
        return
    full_chunks = int(last_action.shape[-1] // action_dim)
    if full_chunks <= 0:
        return
    if executed_chunks < full_chunks:
        pad_chunks = full_chunks - executed_chunks
        intervene_flag = torch.cat(
            [
                intervene_flag,
                torch.zeros(batch_size, pad_chunks, dtype=torch.bool),
            ],
            dim=1,
        )
        intervene_action_chunks = intervene_action.reshape(
            batch_size, executed_chunks, action_dim
        )
        intervene_action = torch.cat(
            [
                intervene_action_chunks,
                torch.zeros(batch_size, pad_chunks, action_dim, dtype=torch.float32),
            ],
            dim=1,
        ).reshape(batch_size, -1)
    else:
        intervene_flag = intervene_flag[:, :full_chunks].contiguous()
        intervene_action = (
            intervene_action.reshape(batch_size, executed_chunks, action_dim)[
                :, :full_chunks
            ]
            .reshape(batch_size, -1)
            .contiguous()
        )

    rollout.update_last_actions(intervene_action, intervene_flag)

    if not rollout.forward_inputs:
        return
    last_fi = rollout.forward_inputs[-1]
    exec_env_action = last_fi.get("executed_env_action_absolute", None)
    if not torch.is_tensor(exec_env_action):
        return
    batch_size, num_action_chunks = intervene_flag.shape[:2]
    flags = intervene_flag.reshape(batch_size, num_action_chunks, 1)
    human_env_action = intervene_action.reshape(batch_size, num_action_chunks, -1)
    policy_env_action = exec_env_action.reshape(batch_size, num_action_chunks, -1)
    executed_env_action = human_env_action * flags + policy_env_action * (~flags)
    last_fi["executed_env_action_absolute"] = (
        executed_env_action.reshape(batch_size, -1).cpu().contiguous()
    )


def maybe_delete_last_saved_rollout(
    *,
    keyboard,
    buffer,
    saved: int,
    progress_bar,
    log_fn,
    delete_key: str = "d",
    preconsumed_key: str | None = None,
) -> int:
    key = preconsumed_key
    if key is None:
        key = keyboard.consume_any_press((delete_key,), include_held=False)
    if key is None:
        return saved

    result = buffer.delete_last_trajectory()
    if not result.get("deleted", False):
        log_fn(f"[eval-with-reward] key='{key}' delete ignored: {result.get('reason')}")
        return saved

    saved = max(0, saved - 1)
    progress_bar.n = saved
    progress_bar.set_postfix({"saved": saved})
    progress_bar.refresh()
    log_fn(
        "[eval-with-reward] key='{}' deleted trajectory_id={} samples={} "
        "file_deleted={} remaining_trajectories={} remaining_samples={} path={}".format(
            key,
            result.get("trajectory_id"),
            result.get("num_samples"),
            result.get("file_deleted"),
            result.get("num_trajectories"),
            result.get("total_samples"),
            result.get("path"),
        )
    )
    return saved


def wait_for_rollout_start_or_delete(
    *,
    keyboard,
    buffer,
    saved: int,
    progress_bar,
    log_fn,
    total_rollouts: int,
    delete_key: str = "d",
    sleep_fn=time.sleep,
    prompt_fn=print,
) -> int:
    prompt_fn(
        f"[Rollout {saved + 1}/{total_rollouts}] "
        "Press ENTER to start, or press d to delete last saved rollout...",
        flush=True,
    )
    while True:
        if keyboard.consume_press("Key.enter"):
            return saved
        key = keyboard.consume_any_press((delete_key,), include_held=False)
        if key == delete_key:
            saved = maybe_delete_last_saved_rollout(
                keyboard=keyboard,
                buffer=buffer,
                saved=saved,
                progress_bar=progress_bar,
                log_fn=log_fn,
                delete_key=delete_key,
                preconsumed_key=key,
            )
            prompt_fn(
                f"[Rollout {saved + 1}/{total_rollouts}] "
                "Press ENTER to start, or press d to delete last saved rollout...",
                flush=True,
            )
        sleep_fn(0.05)


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
            f"[Rollout {rollout_idx + 1}/{self.num_rollouts}] "
            "Press ENTER to start, or press d to delete last saved rollout...",
            flush=True,
        )
        while not self._keyboard.consume_press("Key.enter"):
            time.sleep(0.05)

    def _smooth_chunk_actions(self, chunk_actions):
        """Apply each sub-env's process_action_chunk (Butterworth smoothing +
        sliding-window boundary blend) to the whole chunk, mirroring
        RealWorldEnv.chunk_step's pre-processing (realworld_env.py:335-345),
        before the actions are executed step-by-step.
        """
        processed = (
            chunk_actions.clone()
            if isinstance(chunk_actions, torch.Tensor)
            else chunk_actions.copy()
        )
        for env_idx, sub_env in enumerate(self.env.env.envs):
            unwrapped_env = sub_env.unwrapped
            if hasattr(unwrapped_env, "process_action_chunk"):
                processed[env_idx] = unwrapped_env.process_action_chunk(
                    chunk_actions[env_idx]
                )
        return processed

    def _execute_action_chunk(self, chunk_actions):
        eval_cfg = self.cfg.env.eval
        break_after_intervention = (
            eval_cfg.get("break_chunk_after_intervention", False)
            if hasattr(eval_cfg, "get")
            else getattr(eval_cfg, "break_chunk_after_intervention", False)
        )
        if not break_after_intervention:
            return self.env.chunk_step(chunk_actions)

        chunk_actions = self._smooth_chunk_actions(chunk_actions)

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
        rollout_attempt = 0
        progress_bar = tqdm(
            total=self.num_rollouts,
            desc="Eval rollouts with reward:",
        )
        progress_bar.set_postfix({"saved": saved})

        while saved < self.num_rollouts:
            saved = maybe_delete_last_saved_rollout(
                keyboard=self._keyboard,
                buffer=self.buffer,
                saved=saved,
                progress_bar=progress_bar,
                log_fn=self.log_info,
            )
            saved = wait_for_rollout_start_or_delete(
                keyboard=self._keyboard,
                buffer=self.buffer,
                saved=saved,
                progress_bar=progress_bar,
                log_fn=self.log_info,
                total_rollouts=self.num_rollouts,
            )

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
                infos_last = infos_list[-1] if infos_list else {}
                substep_obs_list = (
                    infos_last.get("substep_obs", None)
                    if isinstance(infos_last, dict)
                    else None
                )
                if not isinstance(substep_obs_list, (list, tuple)):
                    substep_obs_list = obs_list
                processed_substep_obs = [
                    self._process_obs(dict(substep_obs))
                    for substep_obs in substep_obs_list
                    if isinstance(substep_obs, dict)
                ]
                forward_inputs = {
                    "action": action_tensor,
                    **build_substep_forward_inputs(
                        obs_list=processed_substep_obs,
                        chunk_actions=torch.as_tensor(chunk_actions),
                        target_steps=self.num_action_chunks,
                        chunk_valid_mask=(
                            infos_last.get("chunk_valid_mask", None)
                            if isinstance(infos_last, dict)
                            else None
                        ),
                    ),
                }

                step = ChunkStepResult(
                    actions=action_tensor,
                    rewards=reward_tensor.float(),
                    dones=done_tensor,
                    terminations=terminated_tensor,
                    truncations=truncated_tensor,
                    forward_inputs=forward_inputs,
                )
                rollout.append_step_result(step)
                apply_intervention_to_rollout_step(rollout, infos_last)
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
            rollout_attempt += 1
            progress_bar.n = saved
            progress_bar.set_postfix({"saved": saved})
            progress_bar.refresh()
            self.log_info(
                f"Saved rollout attempt {rollout_attempt} (total saved: {saved})"
            )
            saved = maybe_delete_last_saved_rollout(
                keyboard=self._keyboard,
                buffer=self.buffer,
                saved=saved,
                progress_bar=progress_bar,
                log_fn=self.log_info,
            )

        progress_bar.close()
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
