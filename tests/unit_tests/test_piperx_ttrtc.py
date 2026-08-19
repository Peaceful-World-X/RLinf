# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0

import numpy as np
import torch

from rlinf.models.embodiment.openpi.ttrtc import (
    DelayEstimator,
    pin_actor_prefix,
    prepare_basepolicy,
)


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
