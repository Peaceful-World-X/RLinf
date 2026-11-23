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

"""Check allocation and one NCCL collective across the requested GPU group."""

import os

import torch
import torch.distributed as dist

rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(rank)
dist.init_process_group("nccl")
try:
    value = torch.ones(1, device=f"cuda:{rank}")
    dist.all_reduce(value)
    if value.item() != dist.get_world_size():
        raise RuntimeError("NCCL collective failed")
    print(
        f"rank={rank} device={torch.cuda.get_device_name(rank)} all_reduce={value.item()}",
        flush=True,
    )
finally:
    dist.destroy_process_group()
