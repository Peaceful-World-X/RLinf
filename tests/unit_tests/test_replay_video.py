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

from concurrent.futures import Future

import torch

from rlinf.data.replay_video import (
    DeferredH264Payload,
    StreamingH264TensorEncoder,
    decode_h264_images_in_dict,
    decode_h264_tensor,
    encode_h264_tensor,
    is_h264_payload,
    split_h264_payload,
    stack_h264_payloads,
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


def test_streaming_h264_encodes_frames_without_raw_chunk_stack() -> None:
    frames = torch.zeros((1, 4, 2, 16, 16, 3), dtype=torch.uint8)
    encoder = StreamingH264TensorEncoder(frames[:, 0])
    for index in range(frames.shape[1]):
        frame = frames[:, index].clone()
        frame.fill_(index * 50)
        frames[:, index] = frame
        encoder.append(frame)

    decoded = decode_h264_tensor(encoder.finish())

    assert decoded.shape == frames.shape
    assert (decoded.float() - frames.float()).abs().mean() < 5


def test_deferred_h264_payload_stacks_and_splits_without_decoding() -> None:
    frames = torch.zeros((1, 3, 1, 16, 16, 3), dtype=torch.uint8)
    payload = encode_h264_tensor(frames, time_dim=1)
    future = Future()
    future.set_result(payload)
    stacked = stack_h264_payloads(
        [DeferredH264Payload(future), payload], dim=0
    )

    split = split_h264_payload(stacked, split_size=1, dim=1)
    decoded = decode_h264_tensor(split[0])

    assert is_h264_payload(split[0])
    assert decoded.shape == (2, 1, 3, 1, 16, 16, 3)
