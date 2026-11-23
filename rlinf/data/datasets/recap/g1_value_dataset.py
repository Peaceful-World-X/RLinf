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

"""G1 camera packing and trajectory splits for the official RECAP value dataset."""

import json
from array import array
from pathlib import Path

from openpi import transforms

from rlinf.data.datasets.recap.value_dataset import ValueDataset
from rlinf.data.storage.lerobot import episode_boundaries
from rlinf.models.embodiment.openpi.dataconfig.g1_dataconfig import G1Inputs


class G1ValueDataset(ValueDataset):
    def __init__(
        self, dataset_path, split_manifest, repack_keys, split="train", **kwargs
    ):
        self.repack_keys = repack_keys
        super().__init__(dataset_path=dataset_path, split=split, **kwargs)

        manifest = json.loads(Path(split_manifest).read_text())
        source = manifest["sources"][Path(dataset_path).name]
        train, val = source["train"], source["val"]
        count = self.dataset_meta.total_episodes
        if len(train) + len(val) != count or set(train + val) != set(range(count)):
            raise ValueError(f"Invalid trajectory partition for {dataset_path}")
        episode_ids = source[split]
        stride = 1 if split == "train" else manifest["validation_frame_stride"]
        starts, ends = episode_boundaries(self._base)
        indices = array("q")
        for episode in episode_ids:
            indices.extend(range(starts[episode], ends[episode], stride))
        if not indices:
            raise ValueError(f"No frames in {split} for {dataset_path}")
        self._indices = indices

    def _build_transform(self, robot_type, model_type, action_dim, default_prompt):
        return transforms.compose(
            [
                transforms.RepackTransform(self.repack_keys),
                G1Inputs(),
                transforms.InjectDefaultPrompt(default_prompt),
                transforms.PadStatesAndActions(action_dim),
            ]
        )
