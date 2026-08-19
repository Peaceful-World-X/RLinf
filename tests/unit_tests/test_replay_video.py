# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

from rlinf.data.replay_video import (
    decode_h264_images_in_dict,
    encode_h264_tensor,
    is_h264_payload,
)


def test_h264_replay_tensor_round_trip() -> None:
    frames = torch.zeros((4, 1, 16, 16, 3), dtype=torch.uint8)
    for index in range(frames.shape[0]):
        frames[index].fill_(index * 50)

    payload = encode_h264_tensor(frames, time_dim=0)
    values = {"main_images": payload}
    decode_h264_images_in_dict(values)

    decoded = values["main_images"]
    assert is_h264_payload(payload)
    assert decoded.shape == frames.shape
    assert decoded.dtype == torch.uint8
    assert (decoded.float() - frames.float()).abs().mean() < 5


def test_h264_replay_decode_respects_cache_keys() -> None:
    frames = torch.zeros((2, 1, 16, 16, 3), dtype=torch.uint8)
    payload = encode_h264_tensor(frames, time_dim=0)
    values = {"main_images": payload}

    decode_h264_images_in_dict(values, {"states"})

    assert is_h264_payload(values["main_images"])
