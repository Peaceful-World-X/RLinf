# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import torch

from rlinf.data.replay_video import decode_h264_tensor
from rlinf.models.embodiment.openpi.openpi_action_model import (
    OpenPi0ForRLActionPrediction,
    _batch_ttrtc_prev_action,
    _token_conditioned_rmsnorm_forward,
)
from rlinf.models.embodiment.openpi.ttrtc import (
    DelayEstimator,
    pin_actor_prefix,
    prepare_basepolicy,
)
from rlinf.workers.env.piperx_ttrtc_env_worker import PiperXTTRTCEnvWorker


def test_ttrtc_filters_basepolicy_before_actor_prefix_is_pinned():
    base = torch.zeros(1, 50, 14)
    base[0, 10:15, 0] = torch.tensor([0.0, 0.01, -0.01, 0.01, -0.01])
    base[0, :, 6] = torch.linspace(0.0, 1.0, 50)
    previous = np.full((1, 40, 14), 0.25, dtype=np.float32)
    context = {
        "predicted_delay": 10,
        "prev_base_action": previous,
        "filter_enabled": True,
    }

    filtered = prepare_basepolicy(base, context)

    torch.testing.assert_close(filtered[:, :10], torch.full((1, 10, 14), 0.25))
    torch.testing.assert_close(filtered[0, 10:, 6], base[0, 10:, 6])
    assert not torch.equal(filtered[0, 10:15, 0], base[0, 10:15, 0])


def test_ttrtc_actor_prefix_pin_is_not_a_filter():
    actor = torch.randn(1, 50, 14)
    previous = torch.randn(1, 40, 14)
    result = pin_actor_prefix(
        actor,
        {"predicted_delay": 10, "prev_actor_action": previous.numpy()},
    )
    torch.testing.assert_close(result[:, :10], previous[:, :10])
    torch.testing.assert_close(result[:, 10:], actor[:, 10:])


def test_ttrtc_delay_estimator_uses_client_defaults():
    estimator = DelayEstimator(initial=8, maximum=10, window=20, quantile=0.9)
    assert estimator.predict() == 8
    for value in (7, 8, 9, 10):
        estimator.observe(value)
    assert estimator.predict() == 10


def test_ttrtc_skips_legacy_15_frame_warmup_mix():
    worker = object.__new__(PiperXTTRTCEnvWorker)
    rollout_result = object()

    forced = worker._force_vla_rollout_result_for_warmup(
        rollout_result, chunk_step_idx=0
    )

    assert forced is False


def test_ttrtc_restores_intervention_action_time_dimension():
    worker = object.__new__(PiperXTTRTCEnvWorker)
    worker.cfg = type(
        "Cfg",
        (),
        {"actor": type("Actor", (), {"model": type("Model", (), {"env_action_dim": 14})()})()},
    )()

    action = worker._reshape_intervene_action_step(torch.zeros(1, 14))

    assert action.shape == (1, 1, 14)


def test_ttrtc_adarms_accepts_per_action_token_conditioning():
    norm = type("Norm", (), {})()
    norm.cond_dim = 4
    norm.dense = torch.nn.Linear(4, 12)
    norm.weight = None
    norm._norm = lambda inputs: inputs.float()
    inputs = torch.ones(1, 5, 4)
    condition = torch.ones(1, 5, 4)

    output, gate = _token_conditioned_rmsnorm_forward(norm, inputs, condition)

    assert output.shape == inputs.shape
    assert gate.shape == inputs.shape


def test_ttrtc_previous_action_keeps_batch_and_horizon_dimensions():
    previous = np.zeros((40, 14), dtype=np.float32)

    batched = _batch_ttrtc_prev_action(previous)

    assert batched.shape == (1, 40, 14)


def test_ttrtc_manual_actor_switch_selects_base_plan():
    worker = object.__new__(PiperXTTRTCEnvWorker)
    worker._actor_plan = np.ones((2, 14), dtype=np.float32)
    worker._base_plan = np.zeros((2, 14), dtype=np.float32)
    worker._manual_actor_enabled = False
    worker._force_actor_enabled = False

    assert worker._execution_plan() is worker._base_plan

    worker._force_actor_enabled = True
    assert worker._execution_plan() is worker._actor_plan


def test_ttrtc_actor_switch_break_discards_old_plans_and_resumes_next_cycle():
    worker = object.__new__(PiperXTTRTCEnvWorker)
    worker._rtc = SimpleNamespace(action_horizon=2, prefix_tolerance=1e-4)
    worker._execute_horizon = 1
    worker._request_context = {}
    worker._current_rollout_result = SimpleNamespace(
        forward_inputs={"ttrtc_base_env_action": torch.zeros(1, 2, 14)}
    )
    worker._step_records = []
    worker._executed_actions = []
    worker._episode_ended = False
    worker._break_cycle = False
    worker._consecutive_stale = 0
    worker._actor_plan = None
    worker._base_plan = None
    worker._plan_index = 0
    worker._manual_actor_enabled = True
    worker._force_actor_enabled = False
    worker._cycle_count = 0

    def execute_one(_action):
        worker._step_records.append(object())
        worker._break_cycle = True
        worker._manual_actor_enabled = False

    worker._execute_one = execute_one
    worker._aggregate_steps = lambda _obs: ("output", "info", "online")
    worker._write_executed_replay_fields = lambda _obs, _online: None

    result = worker.env_interact_step(torch.ones(1, 2, 14), 0)

    assert result == ("output", "info", "online")
    assert worker._store_current is False
    assert worker._actor_plan is None
    assert worker._base_plan is None
    assert worker._plan_index == 0
    assert worker._break_cycle is False


def test_ttrtc_base_plan_is_marked_as_vla_in_replay():
    worker = object.__new__(PiperXTTRTCEnvWorker)
    worker._manual_actor_enabled = False
    worker._force_actor_enabled = False
    worker._current_rollout_result = SimpleNamespace(
        forward_inputs={"rollout_control_source": torch.ones(1, 1, dtype=torch.long)}
    )
    worker._training_action_from_env_action = lambda _action, _obs: torch.ones(
        1, 24
    )
    online = {
        "executed_env_action_absolute": torch.zeros(1, 28),
        "chunk_valid_mask": torch.ones(1, 2, dtype=torch.bool),
    }

    worker._write_executed_replay_fields(None, online)

    assert worker._current_rollout_result.forward_inputs[
        "rollout_control_source"
    ].item() == 0


def test_piperx_streams_replay_images_during_execution():
    worker = object.__new__(PiperXTTRTCEnvWorker)
    worker.replay_image_encoding = "h264"
    worker.save_online_substep_obs = True
    worker.replay_video_fps = 30
    worker.cfg = SimpleNamespace(
        actor=SimpleNamespace(model={"num_action_chunks": 15})
    )
    worker._replay_video_executor = ThreadPoolExecutor(max_workers=1)
    frame = torch.zeros((1, 1, 16, 16, 3), dtype=torch.uint8)
    try:
        callback, finish, _ = worker._stream_replay_step()
        stored, _ = callback({"main_images": frame}, {}, 0)
        callback({"main_images": frame}, {}, 1)
        payload = finish()["substep_main_images"]
        decoded = decode_h264_tensor(payload)
    finally:
        worker._replay_video_executor.shutdown(wait=True)

    assert "main_images" not in stored
    assert decoded.shape == (1, 15, 1, 16, 16, 3)


def test_piperx_step_records_do_not_retain_raw_replay_images():
    obs = {
        "states": torch.zeros(1, 14),
        "main_images": torch.zeros(1, 16, 16, 3, dtype=torch.uint8),
        "wrist_images": torch.zeros(1, 2, 16, 16, 3, dtype=torch.uint8),
    }

    stored = PiperXTTRTCEnvWorker._without_replay_images(obs)

    assert set(stored) == {"states"}
    assert set(obs) == {"states", "main_images", "wrist_images"}


def test_ttrtc_denoising_keeps_prefix_fixed_and_uses_zero_prefix_time():
    model = type("Model", (), {})()
    model.config = SimpleNamespace(
        num_steps=2,
        action_horizon=4,
        action_dim=3,
        action_chunk=4,
        action_env_dim=3,
        noise_method="flow_ode",
        joint_logprob=False,
        ignore_last=False,
        is_nft=False,
        add_value_head=False,
        value_after_vlm=False,
    )
    model.action_in_proj = torch.nn.Linear(3, 3, bias=False)
    model.use_vlm_value = False
    model._preprocess_observation = lambda observation, train: (
        None,
        None,
        None,
        None,
        observation.state,
    )
    model._build_prefix_cache = lambda *args: (None, torch.ones(1, 1), None)
    model.sample_noise = lambda shape, device: torch.zeros(shape, device=device)
    model.get_logprob_norm = lambda *args: torch.zeros_like(args[0])
    seen_times = []

    def get_velocity(state, actions, timestep, *args):
        seen_times.append(timestep.clone())
        return torch.zeros_like(actions), torch.zeros_like(actions)

    model.get_velocity = get_velocity
    model._get_noise_level = lambda **kwargs: torch.tensor(0.0)
    model.sample_mean_var_val = (
        lambda *args, **kwargs: OpenPi0ForRLActionPrediction.sample_mean_var_val(
            model, *args, **kwargs
        )
    )
    observation = SimpleNamespace(state=torch.zeros(1, 3))
    previous = torch.full((1, 4, 3), 0.25)

    result = OpenPi0ForRLActionPrediction.sample_actions(
        model,
        observation,
        noise=torch.ones(1, 4, 3),
        mode="eval",
        compute_values=False,
        ttrtc_prev_action=previous,
        ttrtc_prefix_steps=2,
    )

    torch.testing.assert_close(result["actions"][:, :2], previous[:, :2])
    assert len(seen_times) == 2
    for timestep in seen_times:
        torch.testing.assert_close(timestep[:, :2], torch.zeros(1, 2))
        assert torch.all(timestep[:, 2:] > 0)
