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

from examples.embodiment.eval_realworld_with_reward import (
    apply_intervention_to_rollout_step,
    build_substep_forward_inputs,
    maybe_delete_last_saved_rollout,
    wait_for_rollout_start_or_delete,
)
from rlinf.data.embodied_io_struct import ChunkStepResult, EmbodiedRolloutResult


def test_build_substep_forward_inputs_matches_online_replay_shapes():
    obs_list = [
        {
            "main_images": torch.full((1, 4, 4, 3), step, dtype=torch.uint8),
            "extra_view_images": torch.full((1, 2, 4, 4, 3), step, dtype=torch.uint8),
            "states": torch.full((1, 14), float(step), dtype=torch.float32),
        }
        for step in range(3)
    ]
    chunk_actions = torch.arange(1 * 5 * 14, dtype=torch.float32).reshape(1, 5, 14)
    chunk_valid_mask = torch.tensor([[True, True, True, False, False]])

    forward_inputs = build_substep_forward_inputs(
        obs_list=obs_list,
        chunk_actions=chunk_actions,
        target_steps=5,
        chunk_valid_mask=chunk_valid_mask,
    )

    assert forward_inputs["substep_main_images"].shape == (1, 5, 4, 4, 3)
    assert forward_inputs["substep_extra_view_images"].shape == (1, 5, 2, 4, 4, 3)
    assert forward_inputs["substep_states"].shape == (1, 5, 14)
    assert forward_inputs["executed_env_action_absolute"].shape == (1, 70)
    assert forward_inputs["chunk_valid_mask"].shape == (1, 5)
    assert forward_inputs["chunk_valid_mask"].tolist() == [
        [True, True, True, False, False]
    ]
    assert torch.equal(
        forward_inputs["substep_main_images"][:, 3], obs_list[-1]["main_images"]
    )
    assert torch.equal(forward_inputs["substep_states"][:, 4], obs_list[-1]["states"])


def test_apply_intervention_to_rollout_step_marks_saved_flags_and_actions():
    rollout = EmbodiedRolloutResult(max_episode_length=10)
    policy_action = torch.arange(10, dtype=torch.float32).reshape(1, 10)
    human_action = torch.full((1, 10), -1.0, dtype=torch.float32)
    intervene_flag = torch.tensor([[False, True, False, False, False]])

    rollout.append_step_result(
        ChunkStepResult(
            actions=policy_action.clone(),
            rewards=torch.zeros(1, 1),
            dones=torch.zeros(1, 1, dtype=torch.bool),
            terminations=torch.zeros(1, 1, dtype=torch.bool),
            truncations=torch.zeros(1, 1, dtype=torch.bool),
            forward_inputs={
                "action": policy_action.clone(),
                "executed_env_action_absolute": policy_action.clone(),
            },
        )
    )

    apply_intervention_to_rollout_step(
        rollout,
        {
            "intervene_action": human_action,
            "intervene_flag": intervene_flag,
        },
    )

    saved = rollout.to_trajectory()
    assert saved.intervene_flags.shape == (1, 1, 10)
    assert saved.intervene_flags[0, 0].reshape(5, 2).tolist() == [
        [False, False],
        [True, True],
        [False, False],
        [False, False],
        [False, False],
    ]
    assert saved.actions[0, 0].reshape(5, 2)[1].tolist() == [-1.0, -1.0]
    assert rollout.forward_inputs[-1]["policy_action_before_intervention"].equal(
        policy_action
    )
    assert rollout.forward_inputs[-1]["action"].reshape(5, 2)[1].tolist() == [
        -1.0,
        -1.0,
    ]
    assert rollout.forward_inputs[-1]["executed_env_action_absolute"].reshape(5, 2)[
        1
    ].tolist() == [-1.0, -1.0]


def test_apply_intervention_pads_short_chunk_flags_to_full_action_horizon():
    rollout = EmbodiedRolloutResult(max_episode_length=10)
    policy_action = torch.arange(30, dtype=torch.float32).reshape(1, 30)
    human_action = torch.tensor([[100.0, 101.0, 200.0, 201.0]], dtype=torch.float32)
    intervene_flag = torch.tensor([[False, True]])

    rollout.append_step_result(
        ChunkStepResult(
            actions=policy_action.clone(),
            rewards=torch.zeros(1, 1),
            dones=torch.zeros(1, 1, dtype=torch.bool),
            terminations=torch.zeros(1, 1, dtype=torch.bool),
            truncations=torch.zeros(1, 1, dtype=torch.bool),
            forward_inputs={
                "action": policy_action.clone(),
                "executed_env_action_absolute": policy_action.clone(),
            },
        )
    )

    apply_intervention_to_rollout_step(
        rollout,
        {
            "intervene_action": human_action,
            "intervene_flag": intervene_flag,
        },
    )

    saved = rollout.to_trajectory()
    assert saved.intervene_flags.shape == (1, 1, 30)
    assert saved.intervene_flags[0, 0].reshape(15, 2).any(
        dim=-1
    ).nonzero().tolist() == [[1]]
    assert saved.actions[0, 0].reshape(15, 2)[1].tolist() == [200.0, 201.0]
    assert saved.actions[0, 0].reshape(15, 2)[2].tolist() == [4.0, 5.0]


class _FakeKeyboard:
    def __init__(self, key=None, keys=None):
        self.key = key
        self.keys = list(keys or [])

    def consume_any_press(self, keys, *, include_held=True):
        if self.keys:
            key = self.keys.pop(0)
            if key in keys:
                return key
            return None
        if self.key in keys:
            key = self.key
            self.key = None
            return key
        return None


class _FakeBuffer:
    def __init__(self, result):
        self.result = result

    def delete_last_trajectory(self):
        return self.result


class _FakeProgressBar:
    def __init__(self, n=0):
        self.n = n
        self.refresh_count = 0
        self.postfix = None

    def set_postfix(self, values):
        self.postfix = values

    def refresh(self):
        self.refresh_count += 1


class _FakeLogger:
    def __init__(self):
        self.messages = []

    def __call__(self, message):
        self.messages.append(message)


def test_delete_last_saved_rollout_updates_count_and_progress_bar():
    progress_bar = _FakeProgressBar(n=3)
    logger = _FakeLogger()

    saved = maybe_delete_last_saved_rollout(
        keyboard=_FakeKeyboard("d"),
        buffer=_FakeBuffer(
            {
                "deleted": True,
                "trajectory_id": 2,
                "num_samples": 10,
                "file_deleted": True,
                "num_trajectories": 2,
                "total_samples": 20,
                "path": "/tmp/trajectory_2.pt",
            }
        ),
        saved=3,
        progress_bar=progress_bar,
        log_fn=logger,
    )

    assert saved == 2
    assert progress_bar.n == 2
    assert progress_bar.postfix == {"saved": 2}
    assert progress_bar.refresh_count == 1
    assert "deleted trajectory_id=2" in logger.messages[0]


def test_delete_last_saved_rollout_ignores_missing_key():
    progress_bar = _FakeProgressBar(n=3)

    saved = maybe_delete_last_saved_rollout(
        keyboard=_FakeKeyboard(None),
        buffer=_FakeBuffer({"deleted": True}),
        saved=3,
        progress_bar=progress_bar,
        log_fn=lambda message: None,
    )

    assert saved == 3
    assert progress_bar.n == 3
    assert progress_bar.refresh_count == 0


def test_wait_for_rollout_start_handles_delete_before_enter():
    progress_bar = _FakeProgressBar(n=3)
    logger = _FakeLogger()

    saved = wait_for_rollout_start_or_delete(
        keyboard=_FakeKeyboard(keys=["d", "Key.enter"]),
        buffer=_FakeBuffer(
            {
                "deleted": True,
                "trajectory_id": 2,
                "num_samples": 10,
                "file_deleted": True,
                "num_trajectories": 2,
                "total_samples": 20,
                "path": "/tmp/trajectory_2.pt",
            }
        ),
        saved=3,
        progress_bar=progress_bar,
        log_fn=logger,
        total_rollouts=5,
        sleep_fn=lambda _: None,
        prompt_fn=lambda *args, **kwargs: None,
    )

    assert saved == 2
    assert progress_bar.n == 2
    assert "deleted trajectory_id=2" in logger.messages[0]
