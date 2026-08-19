# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0

"""PiperX-only asynchronous TTRTC execution for online OpenRLToken."""

from __future__ import annotations

import concurrent.futures

import numpy as np
import torch

from rlinf.data.embodied_io_struct import EnvOutput
from rlinf.models.embodiment.openpi.ttrtc import DelayEstimator
from rlinf.workers.env.async_env_worker import AsyncEnvWorker


class PiperXTTRTCEnvWorker(AsyncEnvWorker):
    """Overlap PiperX control with the next inference request."""

    def __init__(self, cfg):
        super().__init__(cfg)
        rtc = cfg.rollout.ttrtc
        self._rtc = rtc
        self._execute_horizon = int(rtc.execute_horizon)
        self.n_train_chunk_steps = (
            int(cfg.env.train.max_steps_per_rollout_epoch) // self._execute_horizon
        )
        self._delay = DelayEstimator(
            rtc.initial_delay,
            rtc.trained_delay_max,
            rtc.delay_window_size,
            rtc.delay_quantile,
            rtc.delay_margin_steps,
        )
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._request_id = 0
        self._reset_ttrtc_state()

    def _reset_ttrtc_state(self):
        self._actor_plan = None
        self._base_plan = None
        self._plan_index = 0
        self._request_context = None
        self._request_obs = None
        self._live_obs = None
        self._step_records = []
        self._executed_actions = []
        self._store_current = True
        self._cycle_count = 0
        self._episode_ended = False
        self._break_cycle = False
        self._consecutive_stale = 0
        self._last_intervening = None

    def bootstrap_step(self):
        self._reset_ttrtc_state()
        return super().bootstrap_step()

    def send_env_batch(self, rollout_channel, env_batch, mode="train"):
        if mode != "train":
            return super().send_env_batch(rollout_channel, env_batch, mode)
        previous_actor = previous_base = None
        if self._actor_plan is not None and self._plan_index < len(self._actor_plan):
            previous_actor = self._actor_plan[self._plan_index :].copy()
            previous_base = self._base_plan[self._plan_index :].copy()
        predicted = 0
        if previous_actor is not None:
            predicted = max(self._delay.predict(), int(self._rtc.min_prefix_steps))
            predicted = min(predicted, int(self._rtc.trained_delay_max))
        context = {
            "request_id": self._request_id,
            "predicted_delay": predicted,
            "execute_horizon": self._execute_horizon,
            "filter_enabled": bool(self._rtc.filter_enabled),
            "prev_actor_action": previous_actor,
            "prev_base_action": previous_base,
        }
        self._request_id += 1
        self._request_context = context
        self._request_obs = env_batch["obs"]
        self._live_obs = env_batch["obs"]
        payload = dict(env_batch)
        payload["ttrtc_context"] = context
        return super().send_env_batch(rollout_channel, payload, mode)

    def recv_rollout_results(self, input_channel, mode="train"):
        if mode != "train":
            return super().recv_rollout_results(input_channel, mode)
        future = self._executor.submit(
            super().recv_rollout_results, input_channel, mode
        )
        bootstrap_only = self._cycle_count >= self.n_train_chunk_steps
        self._step_records = []
        self._executed_actions = []
        while (
            not future.done()
            and not bootstrap_only
            and not self._episode_ended
            and not self._break_cycle
        ):
            if self._actor_plan is None:
                break
            if self._plan_index >= len(self._actor_plan):
                raise RuntimeError("TTRTC action plan depleted during inference")
            self._execute_one(self._actor_plan[self._plan_index])
            self._plan_index += 1
        result = future.result()
        self._current_rollout_result = result
        if bootstrap_only:
            self._cycle_count = 0
        return result

    def _execute_one(self, action):
        tensor = torch.as_tensor(action, dtype=torch.float32)[None, None]
        output, info, online = super().env_interact_step(tensor, 0, self._live_obs)
        self._step_records.append((output, info, online))
        self._executed_actions.append(np.asarray(action, dtype=np.float32).copy())
        self._live_obs = output.obs
        if output.dones is not None and bool(output.dones.any()):
            self._episode_ended = True
        flags = output.intervene_flags
        if flags is not None and flags.numel():
            intervening = bool(flags.reshape(-1)[-1].item())
            if (
                self._last_intervening is not None
                and intervening != self._last_intervening
            ):
                self._break_cycle = True
            self._last_intervening = intervening
        valid = online.get("chunk_valid_mask")
        if torch.is_tensor(valid) and not bool(valid.any()):
            self._break_cycle = True

    def env_interact_step(self, chunk_actions, stage_id, curr_obs=None):
        if stage_id != 0:
            raise ValueError("PiperX TTRTC requires one pipeline stage")
        context = self._request_context or {}
        actor = (
            torch.as_tensor(chunk_actions)
            .reshape(-1, int(self._rtc.action_horizon), 14)[0]
            .cpu()
            .to(torch.float32)
            .numpy()
        )
        base_tensor = self._current_rollout_result.forward_inputs.get(
            "ttrtc_base_env_action"
        )
        if base_tensor is None:
            raise RuntimeError("TTRTC rollout is missing filtered basepolicy actions")
        base = base_tensor.reshape(-1, len(actor), 14)[0].cpu().float().numpy()
        elapsed = len(self._step_records)
        prefix = int(context.get("predicted_delay", 0))
        previous_actor = context.get("prev_actor_action")
        previous_base = context.get("prev_base_action")
        mismatch = 0.0
        if previous_actor is not None and prefix:
            count = min(prefix, len(previous_actor), len(actor))
            mismatch = max(
                float(
                    np.max(np.abs(actor[:count] - np.asarray(previous_actor)[:count]))
                ),
                float(np.max(np.abs(base[:count] - np.asarray(previous_base)[:count]))),
            )
        if mismatch > float(self._rtc.prefix_tolerance):
            raise RuntimeError(f"TTRTC prefix mismatch: {mismatch}")

        stale = previous_actor is not None and elapsed > prefix
        self._store_current = (
            not stale and not self._episode_ended and not self._break_cycle
        )
        if self._break_cycle:
            self._actor_plan = None
            self._base_plan = None
            self._plan_index = 0
        elif stale:
            self._consecutive_stale += 1
            if self._consecutive_stale > int(self._rtc.max_consecutive_stale):
                raise RuntimeError("Too many consecutive stale TTRTC responses")
        else:
            self._consecutive_stale = 0
            if previous_actor is not None:
                self._delay.observe(elapsed)
            self._actor_plan = actor
            self._base_plan = base
            self._plan_index = elapsed

        while (
            len(self._step_records) < self._execute_horizon
            and not self._episode_ended
            and not self._break_cycle
        ):
            if self._actor_plan is None or self._plan_index >= len(self._actor_plan):
                raise RuntimeError("TTRTC action plan depleted before execute horizon")
            self._execute_one(self._actor_plan[self._plan_index])
            self._plan_index += 1

        output, info, online = self._aggregate_steps(curr_obs)
        self._cycle_count += 1
        self._write_executed_replay_fields(curr_obs, online)
        return output, info, online

    def _aggregate_steps(self, curr_obs):
        if not self._step_records:
            raise RuntimeError("TTRTC cycle executed no actions")
        horizon = int(self.cfg.actor.model.num_action_chunks)
        outputs = [item[0] for item in self._step_records]

        def pad(values, fill=0):
            tensor = torch.cat(values, dim=1)
            if tensor.shape[1] < horizon:
                shape = (tensor.shape[0], horizon - tensor.shape[1], *tensor.shape[2:])
                tensor = torch.cat(
                    [tensor, torch.full(shape, fill, dtype=tensor.dtype)], dim=1
                )
            return tensor[:, :horizon]

        rewards = pad([out.rewards for out in outputs], 0)
        terms = pad([out.terminations for out in outputs], False)
        truncs = pad([out.truncations for out in outputs], False)
        dones = torch.logical_or(terms, truncs)
        intervene_actions = None
        intervene_flags = None
        if any(out.intervene_actions is not None for out in outputs):
            action_values = []
            flag_values = []
            for out in outputs:
                action_values.append(
                    out.intervene_actions
                    if out.intervene_actions is not None
                    else torch.zeros((1, 1, 14), dtype=torch.float32)
                )
                flag_values.append(
                    out.intervene_flags
                    if out.intervene_flags is not None
                    else torch.zeros((1, 1), dtype=torch.bool)
                )
            intervene_actions = pad(action_values, 0)
            intervene_flags = pad(flag_values, False)
        valid_values = [
            item[2].get("chunk_valid_mask")[:, :1] for item in self._step_records
        ]
        valid = pad(valid_values, False).to(torch.bool)
        actions = torch.zeros((1, horizon, 14), dtype=torch.float32)
        count = min(len(self._executed_actions), horizon)
        actions[0, :count] = torch.from_numpy(
            np.asarray(self._executed_actions[:count])
        )
        actions *= valid.unsqueeze(-1)
        online = {
            "chunk_valid_mask": valid,
            "executed_env_action_absolute": actions.reshape(1, -1),
            "executed_action_stats": self._executed_action_stats(actions),
        }
        online.update(self._stack_substep_obs([out.obs for out in outputs]))
        final_obs = next(
            (out.final_obs for out in reversed(outputs) if out.final_obs is not None),
            None,
        )
        output = EnvOutput(
            obs=outputs[-1].obs,
            final_obs=final_obs,
            rewards=rewards,
            dones=dones,
            terminations=terms,
            truncations=truncs,
            intervene_actions=intervene_actions,
            intervene_flags=intervene_flags,
        )
        info = {}
        for _, step_info, _ in self._step_records:
            info.update(step_info)
        return output, info, online

    def _write_executed_replay_fields(self, curr_obs, online):
        result = self._current_rollout_result
        executed_env = online["executed_env_action_absolute"]
        action = self._training_action_from_env_action(executed_env, curr_obs)
        mask = online["chunk_valid_mask"].unsqueeze(-1)
        action = (action.reshape(mask.shape[0], mask.shape[1], -1) * mask).reshape(
            action.shape[0], -1
        )
        result.forward_inputs["action"] = action
        result.forward_inputs["executed_action"] = action
        result.forward_inputs["model_action"] = action
        result.forward_inputs["actor_action"] = action
        result.forward_inputs["env_action_absolute"] = executed_env
        result.forward_inputs["executed_env_action_absolute"] = executed_env
        result.forward_inputs["chunk_valid_mask"] = online["chunk_valid_mask"]

    def _append_step_result_if_trainable(self, *args, **kwargs):
        if not self._store_current:
            return False
        return super()._append_step_result_if_trainable(*args, **kwargs)

    async def stop(self):
        self._executor.shutdown(wait=False, cancel_futures=True)
        await super().stop()
