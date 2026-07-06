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
from omegaconf import OmegaConf

from examples.embodiment.intervention_classifier_gate import (
    InterventionClassifierGate,
    InterventionGateDecision,
)
from rlinf.data.embodied_io_struct import (
    ChunkStepResult,
    EmbodiedRolloutResult,
    Trajectory,
)
from rlinf.data.replay_buffer import TrajectoryReplayBuffer
from rlinf.workers.actor.fsdp_td3_policy_worker import EmbodiedTD3FSDPPolicy
from rlinf.workers.env.env_worker import EnvWorker


class _FakeGate(InterventionClassifierGate):
    def __init__(self, decisions):
        self.decisions = list(decisions)
        self.enabled = True
        self.model = None
        self.device = torch.device("cpu")
        self.last_probability = None

    def decide_actor_intervention(self, zrl, chunk_idx, warm_up_chunks, threshold):
        return self.decisions.pop(0)


class _RolloutResult:
    def __init__(
        self,
        actor_env_actions,
        ref_env_action,
        actor_action=None,
        ref_action=None,
    ):
        if actor_action is None:
            actor_action = actor_env_actions[..., 7:13].reshape(
                actor_env_actions.shape[0], -1
            )
        if ref_action is None:
            ref_action = ref_env_action[..., 7:13].reshape(ref_env_action.shape[0], -1)
        self.actions = actor_env_actions.clone()
        self.forward_inputs = {
            "ref_env_action_absolute": ref_env_action.clone().reshape(
                ref_env_action.shape[0], -1
            ),
            "ref_action": ref_action.clone().reshape(ref_env_action.shape[0], -1),
            "action": actor_action.clone().reshape(actor_env_actions.shape[0], -1),
            "model_action": actor_action.clone().reshape(
                actor_env_actions.shape[0], -1
            ),
            "actor_action": actor_action.clone().reshape(
                actor_env_actions.shape[0], -1
            ),
            "rollout_control_source": torch.ones(
                actor_env_actions.shape[0], 1, dtype=torch.long
            ),
            "rl_token": torch.zeros(actor_env_actions.shape[0], 4),
        }


def _worker(warm_up_chunks=16):
    worker = object.__new__(EnvWorker)
    worker.cfg = OmegaConf.create(
        {
            "actor": {
                "model": {
                    "num_action_chunks": 3,
                    "action_dim": 6,
                    "env_action_dim": 14,
                    "controlled_action_indices": [7, 8, 9, 10, 11, 12],
                    "action_space": "absolute",
                }
            },
            "rollout": {
                "vla_warmup_chunk_steps": warm_up_chunks,
            },
        }
    )
    worker.vla_warmup_chunk_steps = warm_up_chunks
    worker.rollout_control_mode = "warmup"
    worker._online_action_norm_cache = None
    worker._intervention_gate = InterventionClassifierGate(model=None)
    worker.intervention_classifier_threshold = 0.8
    worker._last_intervention_gate_decision = None
    return worker


def test_warmup_forces_full_pi05_action():
    worker = _worker(warm_up_chunks=2)
    pi05 = torch.arange(1 * 3 * 14, dtype=torch.float32).reshape(1, 3, 14)
    actor = torch.full((1, 3, 14), -100.0)
    ref_action = torch.full((1, 18), 42.0)
    rollout = _RolloutResult(actor, pi05, ref_action=ref_action)

    forced = worker._force_vla_rollout_result_for_warmup(
        rollout,
        chunk_step_idx=1,
        obs_source=None,
    )

    assert forced is True
    assert torch.equal(rollout.actions, pi05)
    assert torch.equal(
        rollout.forward_inputs["executed_env_action_absolute"], pi05.reshape(1, -1)
    )
    assert torch.equal(rollout.forward_inputs["action"], ref_action)
    assert torch.equal(rollout.forward_inputs["executed_action"], ref_action)
    assert rollout.forward_inputs["rollout_control_source"].item() == 0


def test_after_warmup_replaces_only_right_arm_joints_and_keeps_grippers_from_pi05():
    worker = _worker(warm_up_chunks=2)
    pi05 = torch.arange(1 * 3 * 14, dtype=torch.float32).reshape(1, 3, 14)
    actor = torch.full((1, 3, 14), -100.0)
    actor[..., 7:13] = 1000.0
    actor[..., 13] = 7777.0
    actor_action = torch.full((1, 18), 123.0)
    rollout = _RolloutResult(actor, pi05, actor_action=actor_action)

    forced = worker._force_vla_rollout_result_for_warmup(
        rollout,
        chunk_step_idx=2,
        obs_source=None,
    )

    assert forced is False
    expected = pi05.clone()
    expected[..., 7:13] = 1000.0
    assert torch.equal(rollout.actions, expected)
    assert torch.equal(rollout.actions[..., :7], pi05[..., :7])
    assert torch.equal(rollout.actions[..., 13], pi05[..., 13])
    assert torch.equal(
        rollout.forward_inputs["executed_env_action_absolute"], expected.reshape(1, -1)
    )
    assert torch.equal(rollout.forward_inputs["action"], actor_action)
    assert torch.equal(rollout.forward_inputs["executed_action"], actor_action)
    assert rollout.forward_inputs["rollout_control_source"].item() == 1


def test_classifier_disabled_matches_warmup_chunk_boundary_regression():
    worker = _worker(warm_up_chunks=16)
    pi05 = torch.zeros(1, 3, 14)
    actor = torch.zeros(1, 3, 14)
    rollout = _RolloutResult(actor, pi05)

    assert worker._should_force_vla_rollout(rollout, chunk_step_idx=15) is True
    assert worker._should_force_vla_rollout(rollout, chunk_step_idx=16) is False


def test_classifier_enabled_overrides_warmup_chunk_boundary():
    worker = _worker(warm_up_chunks=16)
    worker._intervention_gate = _FakeGate(
        [
            InterventionGateDecision(
                actor_intervene=True,
                mode="classifier",
                classifier_probability=0.9,
                classifier_triggered=True,
            ),
            InterventionGateDecision(
                actor_intervene=False,
                mode="classifier",
                classifier_probability=0.1,
                classifier_triggered=False,
            ),
        ]
    )
    pi05 = torch.zeros(1, 3, 14)
    actor = torch.zeros(1, 3, 14)
    rollout = _RolloutResult(actor, pi05)

    # chunk_step_idx=5 is well before warm_up_chunks=16, but the classifier
    # says the actor should intervene, so no VLA should be forced.
    forced_early = worker._should_force_vla_rollout(rollout, chunk_step_idx=5)
    # chunk_step_idx=20 is past warm_up_chunks=16, but the classifier says no,
    # so VLA should still be forced (no latch, decided independently).
    forced_late = worker._should_force_vla_rollout(rollout, chunk_step_idx=20)

    assert forced_early is False
    assert forced_late is True
    assert worker._last_intervention_gate_decision.classifier_probability == 0.1


def test_warmup_forced_chunk_is_appended_to_rollout_result_for_replay_save():
    worker = _worker(warm_up_chunks=2)
    rollout_store = EmbodiedRolloutResult(max_episode_length=100)
    action = torch.ones(1, 18)
    forward_inputs = {"action": action, "executed_action": action}

    worker._append_step_result_if_trainable(
        rollout_store,
        forced_vla=True,
        result=ChunkStepResult(
            actions=action,
            forward_inputs=forward_inputs,
            rewards=torch.zeros(1, 3),
            dones=torch.zeros(1, 3, dtype=torch.bool),
            truncations=torch.zeros(1, 3, dtype=torch.bool),
            terminations=torch.zeros(1, 3, dtype=torch.bool),
        ),
        save_flags=None,
    )

    assert len(rollout_store.actions) == 1
    assert len(rollout_store.forward_inputs) == 1
    assert len(rollout_store.rewards) == 1


def test_actor_controlled_chunk_is_appended_to_rollout_result():
    worker = _worker(warm_up_chunks=2)
    rollout_store = EmbodiedRolloutResult(max_episode_length=100)
    action = torch.ones(1, 18)
    forward_inputs = {"action": action, "executed_action": action}

    worker._append_step_result_if_trainable(
        rollout_store,
        forced_vla=False,
        result=ChunkStepResult(
            actions=action,
            forward_inputs=forward_inputs,
            rewards=torch.zeros(1, 3),
            dones=torch.zeros(1, 3, dtype=torch.bool),
            truncations=torch.zeros(1, 3, dtype=torch.bool),
            terminations=torch.zeros(1, 3, dtype=torch.bool),
        ),
        save_flags=None,
    )

    assert len(rollout_store.actions) == 1
    assert len(rollout_store.forward_inputs) == 1
    assert len(rollout_store.rewards) == 1


def test_replay_buffer_samples_only_actor_controlled_chunks_when_filter_is_set():
    buffer = TrajectoryReplayBuffer(
        seed=1234,
        enable_cache=True,
        cache_size=1,
        sample_window_size=1,
        auto_save=False,
        cache_forward_input_keys=["rollout_control_source"],
    )
    buffer.set_sample_forward_input_filter("rollout_control_source", min_value=1)
    trajectory = Trajectory(max_episode_length=4)
    trajectory.model_weights_id = "weights"
    trajectory.actions = torch.arange(4, dtype=torch.float32).reshape(4, 1, 1)
    trajectory.rewards = torch.zeros(4, 1)
    trajectory.dones = torch.zeros(4, 1, dtype=torch.bool)
    trajectory.terminations = torch.zeros(4, 1, dtype=torch.bool)
    trajectory.truncations = torch.zeros(4, 1, dtype=torch.bool)
    trajectory.forward_inputs = {
        "rollout_control_source": torch.tensor([0, 0, 1, 1]).reshape(4, 1, 1)
    }

    buffer.add_trajectories([trajectory])
    batch = buffer.sample(32)

    assert torch.equal(
        batch["forward_inputs"]["rollout_control_source"],
        torch.ones_like(batch["forward_inputs"]["rollout_control_source"]),
    )


class _ReplayBufferWithFilterRecorder:
    def __init__(self):
        self.calls = []

    def set_sample_forward_input_filter(self, key, *, max_value=None, min_value=None):
        self.calls.append({"key": key, "max_value": max_value, "min_value": min_value})


def test_td3_worker_excludes_warmup_chunks_from_training_when_configured():
    worker = object.__new__(EmbodiedTD3FSDPPolicy)
    worker.cfg = OmegaConf.create(
        {
            "algorithm": {
                "replay_buffer": {
                    "exclude_warmup_chunks_from_training": True,
                }
            }
        }
    )
    worker.replay_buffer = _ReplayBufferWithFilterRecorder()

    worker._configure_replay_buffer_sample_filter()

    assert worker.replay_buffer.calls == [
        {"key": "rollout_control_source", "max_value": None, "min_value": 1}
    ]


def test_td3_worker_uses_all_chunks_when_warmup_exclusion_is_disabled():
    worker = object.__new__(EmbodiedTD3FSDPPolicy)
    worker.cfg = OmegaConf.create(
        {
            "algorithm": {
                "replay_buffer": {
                    "exclude_warmup_chunks_from_training": False,
                }
            }
        }
    )
    worker.replay_buffer = _ReplayBufferWithFilterRecorder()

    worker._configure_replay_buffer_sample_filter()

    assert worker.replay_buffer.calls == []
