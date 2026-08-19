# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0

"""Small TTRTC runtime shared by PiperX rollout and OpenRLToken."""

from __future__ import annotations

import math
from collections import deque

import numpy as np
import torch

JOINT_INDICES = (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12)


class DelayEstimator:
    def __init__(self, initial=8, maximum=10, window=20, quantile=0.9, margin=0):
        self.initial = int(initial)
        self.maximum = int(maximum)
        self.quantile = float(quantile)
        self.margin = int(margin)
        self.samples = deque(maxlen=int(window))

    def observe(self, ticks: int) -> None:
        self.samples.append(max(0, int(ticks)))

    def predict(self) -> int:
        if not self.samples:
            return self.initial
        value = math.ceil(float(np.quantile(np.asarray(self.samples), self.quantile)))
        return min(self.maximum, max(0, value + self.margin))


def _project_zero_sum_box(proposal: np.ndarray, limit: float) -> np.ndarray:
    correction = np.zeros_like(proposal, dtype=np.float64)
    free = np.ones(len(proposal), dtype=bool)
    for _ in range(len(proposal) + 1):
        if not np.any(free):
            break
        offset = (proposal[free].sum() + correction[~free].sum()) / np.count_nonzero(
            free
        )
        trial = proposal[free] - offset
        clipped = (trial < -limit) | (trial > limit)
        if not np.any(clipped):
            correction[free] = trial
            break
        indices = np.flatnonzero(free)
        correction[indices[trial < -limit]] = -limit
        correction[indices[trial > limit]] = limit
        free[indices[clipped]] = False
    return correction


def filter_plan(plan: np.ndarray, preserve_prefix_steps: int = 0) -> np.ndarray:
    """Apply the reference TTRC reversal filter to arm joints only."""
    result = np.asarray(plan, dtype=np.float32).copy()
    first_step = int(preserve_prefix_steps) + (2 if preserve_prefix_steps else 0)
    for joint in JOINT_INDICES:
        values = result[:, joint].copy()
        steps = np.diff(values)
        curvature = values[:-2] - 2.0 * values[1:-1] + values[2:]
        reversals = (steps[:-1] * steps[1:] < 0.0) & (np.abs(curvature) >= 2.5e-4)
        start = 0
        while start < len(reversals):
            if not reversals[start]:
                start += 1
                continue
            end = start
            while end + 1 < len(reversals) and reversals[end + 1]:
                end += 1
            centers = np.arange(start + 1, end + 2)
            centers = centers[centers >= first_step]
            if len(centers) >= 2:
                proposal = 0.25 * (
                    values[centers - 1] - 2.0 * values[centers] + values[centers + 1]
                )
                correction = _project_zero_sum_box(proposal, 2.0e-3)
                result[centers, joint] = values[centers] + correction
            start = end + 1
    return result


def prepare_basepolicy(actions: torch.Tensor, context: dict | None) -> torch.Tensor:
    """Prefix-condition and filter VLA output before it reaches the RL actor."""
    if not context:
        return actions
    output = actions.detach().cpu().to(torch.float32).numpy()
    previous = context.get("prev_base_action")
    prefix = int(context.get("predicted_delay", 0))
    if previous is not None:
        previous = np.asarray(previous, dtype=np.float32)
        if previous.ndim == 2:
            previous = previous[None]
        prefix = min(prefix, previous.shape[1], output.shape[1])
        output[:, :prefix] = previous[:, :prefix]
    if bool(context.get("filter_enabled", True)):
        output = np.stack([filter_plan(plan, prefix) for plan in output])
    return torch.from_numpy(output).to(device=actions.device, dtype=actions.dtype)


def pin_actor_prefix(actions: torch.Tensor, context: dict | None) -> torch.Tensor:
    """Keep actions already committed during inference identical to the old plan."""
    if not context or context.get("prev_actor_action") is None:
        return actions
    output = actions.clone()
    previous = torch.as_tensor(
        context["prev_actor_action"], device=output.device, dtype=output.dtype
    )
    if previous.dim() == 2:
        previous = previous.unsqueeze(0)
    prefix = min(
        int(context.get("predicted_delay", 0)), previous.shape[1], output.shape[1]
    )
    output[:, :prefix] = previous[:, :prefix]
    return output
