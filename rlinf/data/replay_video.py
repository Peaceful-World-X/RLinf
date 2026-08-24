# Copyright 2026 The GIGA Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""In-memory H.264 serialization for replay-buffer image tensors."""

from __future__ import annotations

import io
from concurrent.futures import Future
from typing import Any

import numpy as np
import torch

_H264_MARKER = "__rlinf_h264__"
_H264_STACK_MARKER = "__rlinf_h264_stack__"


class DeferredH264Payload:
    """A compressed replay payload produced outside the rollout control thread."""

    def __init__(self, future: Future):
        self.future = future

    def resolve(self) -> dict[str, Any]:
        return self.future.result()


class StreamingH264TensorEncoder:
    """Encode one RGB frame at a time while retaining only compressed bytes."""

    def __init__(
        self, frame: torch.Tensor, *, fps: int = 30, crf: int = 18, gop: int = 30
    ):
        import av

        frame = frame.detach().cpu().contiguous()
        if frame.dtype != torch.uint8 or frame.dim() < 3 or frame.shape[-1] != 3:
            raise ValueError("H.264 frames must be uint8 and end in [height, width, 3]")
        self.shape_prefix = list(frame.shape[:-3])
        self.height, self.width = map(int, frame.shape[-3:-1])
        if self.height % 2 or self.width % 2:
            raise ValueError("H.264 yuv420p requires even image dimensions")
        self.fps, self.crf, self.gop = int(fps), int(crf), int(gop)
        count = int(np.prod(self.shape_prefix or [1]))
        self._outputs = [io.BytesIO() for _ in range(count)]
        self._containers = []
        self._streams = []
        for output in self._outputs:
            container = av.open(output, mode="w", format="mp4")
            stream = container.add_stream("libx264", rate=self.fps)
            stream.width, stream.height = self.width, self.height
            stream.pix_fmt = "yuv420p"
            stream.codec_context.thread_count = 1
            stream.options = {
                "crf": str(self.crf),
                "preset": "veryfast",
                "g": str(self.gop),
                "keyint_min": str(self.gop),
                "sc_threshold": "0",
            }
            self._containers.append(container)
            self._streams.append(stream)
        self.frame_count = 0

    def append(self, frame: torch.Tensor) -> None:
        import av

        frame = frame.detach().cpu().contiguous()
        if list(frame.shape[:-3]) != self.shape_prefix:
            raise ValueError("streaming H.264 frame prefix shape changed")
        arrays = frame.reshape(-1, self.height, self.width, 3).numpy()
        for index, array in enumerate(arrays):
            video_frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            video_frame.pts = self.frame_count
            for packet in self._streams[index].encode(video_frame):
                self._containers[index].mux(packet)
        self.frame_count += 1

    def finish(self) -> dict[str, Any]:
        for container, stream in zip(self._containers, self._streams):
            for packet in stream.encode():
                container.mux(packet)
            container.close()
        shape = (
            [
                self.shape_prefix[0],
                self.frame_count,
                *self.shape_prefix[1:],
                self.height,
                self.width,
                3,
            ]
            if self.shape_prefix
            else [self.frame_count, self.height, self.width, 3]
        )
        return {
            _H264_MARKER: 1,
            "data": [
                torch.frombuffer(bytearray(output.getvalue()), dtype=torch.uint8)
                for output in self._outputs
            ],
            "shape": shape,
            "time_dim": 1 if self.shape_prefix else 0,
            "fps": self.fps,
            "crf": self.crf,
            "gop": self.gop,
        }


def is_h264_payload(value: Any) -> bool:
    """Return whether ``value`` is an encoded replay image payload."""
    return isinstance(value, DeferredH264Payload) or isinstance(value, dict) and (
        value.get(_H264_MARKER) == 1 or value.get(_H264_STACK_MARKER) == 1
    )


def _encode_stream(frames: np.ndarray, fps: int, crf: int, gop: int) -> torch.Tensor:
    try:
        import av
    except ImportError as exc:
        raise RuntimeError("image_encoding=h264 requires the 'av' package") from exc

    height, width = frames.shape[-3:-1]
    if height % 2 or width % 2:
        raise ValueError(
            f"H.264 yuv420p requires even image dimensions, got {height}x{width}"
        )
    output = io.BytesIO()
    container = av.open(output, mode="w", format="mp4")
    stream = container.add_stream("libx264", rate=fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    stream.codec_context.thread_count = 1
    stream.options = {
        "crf": str(crf),
        "preset": "veryfast",
        "g": str(gop),
        "keyint_min": str(gop),
        "sc_threshold": "0",
    }
    for frame in frames:
        video_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
        for packet in stream.encode(video_frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return torch.frombuffer(bytearray(output.getvalue()), dtype=torch.uint8)


def encode_h264_tensor(
    tensor: torch.Tensor,
    *,
    fps: int = 30,
    time_dim: int | None = None,
    crf: int = 18,
    gop: int = 30,
) -> dict[str, Any]:
    """Encode every RGB frame in a uint8 tensor without changing its logical shape."""
    if tensor.dtype != torch.uint8 or tensor.dim() < 3 or tensor.shape[-1] != 3:
        raise ValueError(
            "H.264 replay images must be uint8 tensors ending in [height, width, 3]"
        )
    tensor = tensor.detach().cpu().contiguous()
    leading_dims = tensor.dim() - 3
    if time_dim is not None:
        time_dim = time_dim % tensor.dim()
        if time_dim >= leading_dims:
            raise ValueError(
                "time_dim must identify a dimension before [height, width, 3]"
            )
        stream_axes = [axis for axis in range(leading_dims) if axis != time_dim]
        permutation = stream_axes + [time_dim, *range(leading_dims, tensor.dim())]
        ordered = tensor.permute(permutation).contiguous()
        frame_count = int(tensor.shape[time_dim])
    else:
        stream_axes = list(range(leading_dims))
        permutation = [*stream_axes, *range(leading_dims, tensor.dim())]
        ordered = tensor.permute(permutation).contiguous().unsqueeze(leading_dims)
        frame_count = 1

    height, width = tensor.shape[-3:-1]
    streams = ordered.reshape(-1, frame_count, height, width, 3).numpy()
    encoded = [_encode_stream(frames, fps, crf, gop) for frames in streams]
    return {
        _H264_MARKER: 1,
        "data": encoded,
        "shape": list(tensor.shape),
        "time_dim": time_dim,
        "fps": int(fps),
        "crf": int(crf),
        "gop": int(gop),
    }


def _decode_stream(data: torch.Tensor) -> torch.Tensor:
    try:
        import av
    except ImportError as exc:
        raise RuntimeError(
            "decoding H.264 replay images requires the 'av' package"
        ) from exc

    encoded = data.detach().cpu().contiguous().numpy().tobytes()
    container = av.open(io.BytesIO(encoded), mode="r", format="mp4")
    frames = [
        torch.from_numpy(frame.to_ndarray(format="rgb24"))
        for frame in container.decode(video=0)
    ]
    container.close()
    if not frames:
        raise ValueError("H.264 replay payload contains no decodable frames")
    return torch.stack(frames, dim=0)


def decode_h264_tensor(payload: dict[str, Any]) -> torch.Tensor:
    """Decode an H.264 replay payload back to its original tensor shape.

    Online RL sampling keeps images compressed and never calls this path. Full
    trajectory loads for offline consumers may be slower because decoding is
    intentionally not optimized at the expense of replay compression ratio.
    """
    if isinstance(payload, DeferredH264Payload):
        payload = payload.resolve()
    if payload.get(_H264_STACK_MARKER) == 1:
        values = [decode_h264_tensor(item) for item in payload["items"]]
        return torch.stack(values, dim=int(payload["dim"])).contiguous()
    if payload.get(_H264_MARKER) != 1:
        raise ValueError("value is not an H.264 replay payload")

    shape = [int(value) for value in payload["shape"]]
    leading_dims = len(shape) - 3
    time_dim = payload["time_dim"]
    decoded_streams = [_decode_stream(data) for data in payload["data"]]
    if time_dim is None:
        ordered = torch.stack([frames[0] for frames in decoded_streams], dim=0)
        return ordered.reshape(*shape).contiguous()

    time_dim = int(time_dim)
    stream_axes = [axis for axis in range(leading_dims) if axis != time_dim]
    stream_shape = [shape[axis] for axis in stream_axes]
    frame_count = shape[time_dim]
    if any(frames.shape[0] != frame_count for frames in decoded_streams):
        raise ValueError("decoded H.264 frame count does not match replay metadata")
    ordered = torch.stack(decoded_streams, dim=0).reshape(
        *stream_shape, frame_count, *shape[-3:]
    )
    permutation = stream_axes + [time_dim, *range(leading_dims, len(shape))]
    inverse = [permutation.index(axis) for axis in range(len(shape))]
    return ordered.permute(inverse).contiguous()


def resolve_h264_payload(value: Any) -> Any:
    if isinstance(value, DeferredH264Payload):
        return value.resolve()
    if isinstance(value, dict) and value.get(_H264_STACK_MARKER) == 1:
        value = dict(value)
        value["items"] = [resolve_h264_payload(item) for item in value["items"]]
    return value


def stack_h264_payloads(
    payloads: list[dict[str, Any]], dim: int = 0
) -> dict[str, Any]:
    """Represent a tensor stack while keeping every item H.264-compressed."""
    if not payloads or not all(is_h264_payload(payload) for payload in payloads):
        raise ValueError("all stacked values must be H.264 replay payloads")
    payloads = [resolve_h264_payload(payload) for payload in payloads]
    return {
        _H264_STACK_MARKER: 1,
        "dim": int(dim),
        "items": payloads,
    }


def split_h264_payload(
    payload: dict[str, Any], split_size: int, dim: int
) -> list[dict[str, Any]]:
    """Split a compressed payload along a non-image tensor dimension."""
    if payload.get(_H264_STACK_MARKER) == 1:
        stack_dim = int(payload["dim"])
        if dim == stack_dim:
            item_chunks = torch.chunk(
                torch.arange(len(payload["items"])), split_size
            )
            return [
                stack_h264_payloads(
                    [payload["items"][int(index)] for index in indices], stack_dim
                )
                for indices in item_chunks
            ]
        child_dim = dim - 1 if stack_dim < dim else dim
        split_items = [
            split_h264_payload(item, split_size, child_dim)
            for item in payload["items"]
        ]
        return [
            stack_h264_payloads([items[index] for items in split_items], stack_dim)
            for index in range(split_size)
        ]

    shape = [int(value) for value in payload["shape"]]
    leading_dims = len(shape) - 3
    dim = dim % len(shape)
    time_dim = payload["time_dim"]
    if dim >= leading_dims or (time_dim is not None and dim == int(time_dim)):
        raise ValueError("H.264 payloads can only be split across encoded stream axes")
    stream_axes = [
        axis
        for axis in range(leading_dims)
        if time_dim is None or axis != int(time_dim)
    ]
    stream_shape = [shape[axis] for axis in stream_axes]
    stream_dim = stream_axes.index(dim)
    data_grid = np.empty(len(payload["data"]), dtype=object)
    data_grid[:] = payload["data"]
    data_grid = data_grid.reshape(stream_shape)
    results = []
    index_chunks = np.array_split(np.arange(shape[dim]), split_size)
    for indices in index_chunks:
        selected = np.take(data_grid, indices, axis=stream_dim)
        split_payload = dict(payload)
        split_payload["data"] = selected.reshape(-1).tolist()
        split_payload["shape"] = list(shape)
        split_payload["shape"][dim] = int(len(indices))
        results.append(split_payload)
    return results


def decode_h264_images_in_dict(
    values: dict[str, Any], allowed_keys: set[str] | None = None
) -> dict[str, Any]:
    """Decode payloads selected by the replay cache key whitelist."""
    for key, value in list(values.items()):
        if is_h264_payload(value) and (allowed_keys is None or key in allowed_keys):
            values[key] = decode_h264_tensor(value)
    return values
