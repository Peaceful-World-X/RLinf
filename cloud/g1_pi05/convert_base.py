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

"""Run the installed OpenPI converter with the G1 configuration and audit coverage."""

import argparse
from pathlib import Path
from unittest.mock import patch

import torch
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
from rlinf.utils.ckpt_convertor.convert_openpi_jax_to_python import (
    convert_pi0_checkpoint,
)


def convert(source, dest):
    original = PI0Pytorch.load_state_dict

    def load(model, state, strict=False):
        result = original(model, state, strict=strict)
        params = dict(model.named_parameters(remove_duplicate=False))
        loaded = {param.data_ptr() for key, param in params.items() if key in state}
        # The flow expert calls gemma_expert.model directly. Its unused vocabulary
        # projection has no JAX counterpart. PaliGemma's output embedding is tied.
        inactive = "paligemma_with_expert.gemma_expert.lm_head.weight"
        missing = [
            k
            for k in result.missing_keys
            if k != inactive and (k not in params or params[k].data_ptr() not in loaded)
        ]
        if missing or result.unexpected_keys:
            raise ValueError((missing, result.unexpected_keys))
        if inactive in result.missing_keys:
            with torch.no_grad():
                params[inactive].zero_()
        return result

    torch.manual_seed(0)
    with patch.object(PI0Pytorch, "load_state_dict", load):
        convert_pi0_checkpoint(
            str(source),
            "bfloat16",
            str(dest),
            get_openpi_config("pi05_g1_dualgripper").model,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("dest", type=Path)
    args = parser.parse_args()
    convert(args.source, args.dest)
