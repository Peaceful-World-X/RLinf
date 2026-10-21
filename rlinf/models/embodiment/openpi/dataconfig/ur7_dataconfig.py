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

from openpi import transforms
from openpi.models.model import BaseModelConfig
from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory

from rlinf.models.embodiment.openpi.policies.ur7_policy import (
    UR7_ROT6D_LAYOUT,
    UR7Inputs,
    UR7Outputs,
)
from rlinf.models.embodiment.openpi.transforms.rigid_body_delta import (
    RigidBodyAbsoluteActions,
    RigidBodyDeltaActions,
)


@dataclasses.dataclass(frozen=True)
class LeRobotUR7DataConfig(DataConfigFactory):
    """Preserve the UR7 checkpoint's v2.1 data and SE(3) action semantics."""

    def create(
        self, assets_dirs: pathlib.Path, model_config: BaseModelConfig
    ) -> DataConfig:
        if model_config.action_dim != 32 or model_config.action_horizon != 50:
            raise ValueError("UR7 requires model action_dim=32 and action_horizon=50")
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=transforms.Group(
                inputs=[
                    transforms.RepackTransform(
                        {
                            "observation/state": "observation.state",
                            "observation/image": "observation.images.base",
                            "observation/wrist_image": "observation.images.wrist",
                            "actions": "action",
                            "prompt": "task",
                        }
                    )
                ]
            ),
            data_transforms=transforms.Group(
                inputs=[UR7Inputs(), RigidBodyDeltaActions(UR7_ROT6D_LAYOUT)],
                outputs=[RigidBodyAbsoluteActions(UR7_ROT6D_LAYOUT), UR7Outputs()],
            ),
            model_transforms=ModelTransformFactory()(model_config),
            action_sequence_keys=("action",),
        )
