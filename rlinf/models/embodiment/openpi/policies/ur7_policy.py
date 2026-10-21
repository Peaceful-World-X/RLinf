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

"""UR7 v2.1 observations and absolute end-effector actions."""

import dataclasses
from collections.abc import Mapping
from io import BytesIO
from typing import Any

import numpy as np
from openpi import transforms

UR7_ROT6D_LAYOUT = (
    {"kind": "pose6d", "xyz": slice(0, 3), "rot6d": slice(3, 9)},
    {"kind": "scalar_abs", "idx": 9},
)


def canonicalize_state(state: Any) -> np.ndarray:
    """Select EEF pose, gripper, wrench and velocity, excluding joint state."""
    state = np.asarray(state, dtype=np.float32)
    if state.ndim != 1 or state.shape[0] not in (22, 34):
        raise ValueError(f"UR7 state must have shape (22,) or (34,), got {state.shape}")
    state = state[:22]
    if not np.isfinite(state).all():
        raise ValueError("UR7 state contains non-finite values")
    return state


def _image(value: Any) -> np.ndarray:
    if isinstance(value, Mapping) and isinstance(value.get("bytes"), bytes):
        from PIL import Image

        with Image.open(BytesIO(value["bytes"])) as decoded:
            value = np.asarray(decoded.convert("RGB"))
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    value = np.asarray(value)
    if value.ndim != 3:
        raise ValueError(f"UR7 image must have rank 3, got {value.shape}")
    if value.shape[0] in (1, 3, 4) and value.shape[-1] not in (1, 3, 4):
        value = np.moveaxis(value, 0, -1)
    if value.shape[-1] != 3 or not np.isfinite(value).all():
        raise ValueError("UR7 image must contain finite RGB values")
    if value.dtype != np.uint8:
        if np.issubdtype(value.dtype, np.floating) and value.max(initial=0) <= 1:
            value = value * 255
        value = np.clip(value, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(value)


@dataclasses.dataclass(frozen=True)
class UR7Inputs(transforms.DataTransformFn):
    """Map RLinf observations to the three-view OpenPI checkpoint contract."""

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        state = canonicalize_state(data["observation/state"])
        base = _image(data["observation/image"])
        wrist = _image(data["observation/wrist_image"])
        if base.shape != wrist.shape:
            raise ValueError(
                f"UR7 camera shapes must match: {base.shape} vs {wrist.shape}"
            )
        result = {
            "state": np.pad(state, (0, 10)),
            "image": {
                "base_0_rgb": base,
                "left_wrist_0_rgb": wrist,
                "right_wrist_0_rgb": np.zeros_like(base),
            },
            "image_mask": {
                "base_0_rgb": np.bool_(True),
                "left_wrist_0_rgb": np.bool_(True),
                "right_wrist_0_rgb": np.bool_(False),
            },
        }
        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            if actions.ndim != 2 or actions.shape[-1] != 10:
                raise ValueError(
                    f"UR7 actions must have shape (H, 10), got {actions.shape}"
                )
            if not np.isfinite(actions).all():
                raise ValueError("UR7 actions contain non-finite values")
            result["actions"] = np.pad(actions, ((0, 0), (0, 22)))
        if "prompt" in data:
            result["prompt"] = data["prompt"]
        return result


@dataclasses.dataclass(frozen=True)
class UR7Outputs(transforms.DataTransformFn):
    """Expose 50 absolute EEF poses with an absolute gripper command."""

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        return {"actions": np.asarray(data["actions"], dtype=np.float32)[..., :50, :10]}
