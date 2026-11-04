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

import dataclasses
import pathlib

import torch
from openpi import transforms
from openpi.models.model import BaseModelConfig
from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory

from rlinf.models.embodiment.mlp_policy.rlt_mlp_policy import RLTMLPPolicy
from rlinf.models.embodiment.openpi.policies.aloha_policy import AlohaInputs


@dataclasses.dataclass(frozen=True)
class G1Inputs(transforms.DataTransformFn):
    """Accept dataset images and the existing RLinf environment image contract."""

    def __call__(self, data: dict) -> dict:
        if "observation/image" in data:
            wrists = data["observation/wrist_image"]
            data = {
                "images": {
                    "cam_high": data["observation/image"],
                    "cam_left_wrist": wrists[0],
                    "cam_right_wrist": wrists[1],
                },
                "state": data["observation/state"],
                "prompt": data["prompt"],
            }
        return AlohaInputs(adapt_to_pi=False)(data)


@dataclasses.dataclass(frozen=True)
class G1Outputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": data["actions"][..., :16]}


@dataclasses.dataclass(frozen=True)
class LeRobotG1DataConfig(DataConfigFactory):
    """G1 checkpoint contract: three cameras, 14 joint deltas, two absolute grippers."""

    def create(
        self, assets_dirs: pathlib.Path, model_config: BaseModelConfig
    ) -> DataConfig:
        mask = transforms.make_bool_mask(14, -2)
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=transforms.Group(
                inputs=[
                    transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_left_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
            data_transforms=transforms.Group(
                inputs=[G1Inputs(), transforms.DeltaActions(mask)],
                outputs=[transforms.AbsoluteActions(mask), G1Outputs()],
            ),
            model_transforms=ModelTransformFactory()(model_config),
            action_sequence_keys=("action",),
        )


class G1ActionCoordinates:
    """Invertible Stage 2 coordinates: joint deltas and absolute 0..100 grippers."""

    def __init__(self, norm_stats_path: str | pathlib.Path):
        import json

        import torch

        stats = json.loads(pathlib.Path(norm_stats_path).read_text())["norm_stats"]
        self.low = torch.tensor(stats["actions"]["q01"][:14] + [0.0, 0.0])
        self.span = torch.cat(
            (
                torch.tensor(stats["actions"]["q99"][:14]) - self.low[:14] + 1e-6,
                torch.tensor([100.0, 100.0]),
            )
        )
        self.state_low = torch.tensor(stats["state"]["q01"][:16])
        self.state_span = (
            torch.tensor(stats["state"]["q99"][:16]) - self.state_low + 1e-6
        )
        if not (self.span > 0).all() or not (self.state_span > 0).all():
            raise ValueError("G1 quantile spans must be positive")

    def normalize(self, actions, state):
        delta = actions.float().clone()
        delta[..., :14] -= state.float().unsqueeze(-2)[..., :14]
        return 2 * (delta - self.low.to(delta)) / self.span.to(delta) - 1

    def denormalize(self, actions, state):
        physical = (actions.float() + 1) * self.span.to(actions) / 2 + self.low.to(
            actions
        )
        physical[..., :14] += state.float().unsqueeze(-2)[..., :14]
        return physical

    def proprio(self, state):
        return (
            2 * (state.float() - self.state_low.to(state)) / self.state_span.to(state)
            - 1
        )


class G1RLTMLPPolicy(RLTMLPPolicy):
    """Default to VLA residuals while keeping replay, Q and BC in one space."""

    def __init__(self, *args, coordinate_config, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_mode = coordinate_config.get("g1_action_mode", "base_residual")
        if self.action_mode not in ("state_delta", "base_residual", "absolute"):
            raise ValueError(f"Unknown G1 actor coordinates: {self.action_mode}")
        if self.step_action_dim != 14:
            raise ValueError("G1 coordinate ablations require 14 joint outputs")
        coordinates = G1ActionCoordinates(coordinate_config.g1_action_norm_stats_path)
        self.register_buffer("delta_low", coordinates.low[:14], persistent=False)
        self.register_buffer("delta_span", coordinates.span[:14], persistent=False)
        if self.action_mode == "absolute":
            center = torch.tensor(list(coordinate_config.g1_absolute_center))
            halfspan = torch.tensor(list(coordinate_config.g1_absolute_halfspan))
            if (
                center.shape != (14,)
                or halfspan.shape != (14,)
                or not (halfspan > 0).all()
            ):
                raise ValueError(
                    "Absolute joint coordinates require 14 finite positive spans"
                )
            if not torch.isfinite(center).all() or not torch.isfinite(halfspan).all():
                raise ValueError("Absolute joint coordinate statistics must be finite")
            self.register_buffer("absolute_center", center, persistent=False)
            self.register_buffer("absolute_halfspan", halfspan, persistent=False)

    def project_head(self, action: torch.Tensor, obs: dict) -> torch.Tensor:
        """Map the bounded actor head to the original normalized joint-delta space."""
        if self.action_mode == "state_delta":
            return action
        chunk = self._format_chunk_actions(action)
        if self.action_mode == "base_residual":
            # The anchor stays pure VLA even when conditioning uses human replacement/dropout.
            anchor = obs["base_action"].reshape_as(chunk)
            return (anchor + chunk).flatten(1)
        physical = self.absolute_center + self.absolute_halfspan * chunk
        state = obs["states"].reshape(-1, 1, self.step_action_dim)
        return (2 * (physical - state - self.delta_low) / self.delta_span - 1).flatten(
            1
        )

    def sac_forward(self, obs, **kwargs):
        action, logprobs, extra = super().sac_forward(obs, **kwargs)
        return self.project_head(action, obs), logprobs, extra
