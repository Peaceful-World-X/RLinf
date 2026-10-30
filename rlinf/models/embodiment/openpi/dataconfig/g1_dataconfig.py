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

from rlinf.models.embodiment.openpi.policies.aloha_policy import AlohaInputs


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
                inputs=[AlohaInputs(adapt_to_pi=False), transforms.DeltaActions(mask)],
                outputs=[transforms.AbsoluteActions(mask), G1Outputs()],
            ),
            model_transforms=ModelTransformFactory()(model_config),
            action_sequence_keys=("action",),
        )
