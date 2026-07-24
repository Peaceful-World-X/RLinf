# Copyright 2026 The GIGA Authors.
#
from unittest.mock import MagicMock

import gymnasium as gym
from gymnasium import spaces


class _FakeKeyboardListener:
    def __init__(self):
        self.cleared = []

    def clear_keys(self, keys=None):
        self.cleared.append(keys)

    def consume_any_press(self, keys, *, include_held=True):
        return None


class _DummyEnv(gym.Env):
    observation_space = spaces.Dict({"state": spaces.Dict({})})
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,))

    def __init__(self):
        self.reset_calls = []

    def reset(self, *, seed=None, options=None):
        self.reset_calls.append(options)
        return {"state": {}}, {}

    def step(self, action):
        return {"state": {}}, 0.0, False, False, {}


def test_keyboard_reward_reset_waits_for_confirmation_when_armed(monkeypatch):
    from rlinf.envs.realworld.common.wrappers import reward_done_wrapper

    monkeypatch.setattr(reward_done_wrapper, "KeyboardListener", _FakeKeyboardListener)
    wrapper = reward_done_wrapper.KeyboardRewardDoneWrapper(_DummyEnv())
    wrapper._reset_armed = True
    wrapper._wait_for_reset_confirmation = MagicMock()

    wrapper.reset(options={"skip_wait_for_enter_after_reset": True})

    wrapper._wait_for_reset_confirmation.assert_called_once_with()
    assert wrapper._reset_armed is False
