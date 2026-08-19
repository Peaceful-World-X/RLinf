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
from typing import Any

import numpy as np
import torch

_H264_MARKER = "__rlinf_h264__"


def is_h264_payload(value: Any) -> bool:
    """Return whether ``value`` is an encoded replay image payload."""
    return isinstance(value, dict) and value.get(_H264_MARKER) == 1


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


def decode_h264_images_in_dict(
    values: dict[str, Any], allowed_keys: set[str] | None = None
) -> dict[str, Any]:
    """Decode payloads selected by the replay cache key whitelist."""
    for key, value in list(values.items()):
        if is_h264_payload(value) and (allowed_keys is None or key in allowed_keys):
            values[key] = decode_h264_tensor(value)
    return values
