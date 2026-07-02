# Copyright 2026 The GIGA Authors.
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

import importlib.util
from pathlib import Path

import numpy as np


def _load_eval_module():
    repo = Path(__file__).resolve().parents[2]
    path = repo / "scripts" / "evaluate_online_openrlt_trajectories.py"
    spec = importlib.util.spec_from_file_location(
        "evaluate_online_openrlt_trajectories", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_tcp_error_metrics_use_only_valid_steps():
    module = _load_eval_module()
    pt_tcp = np.array(
        [
            [[0.0, 0.0, 0.0], [100.0, 100.0, 100.0]],
            [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
        ],
        dtype=np.float64,
    )
    actor_tcp = np.array(
        [
            [[1.0, 0.0, 0.0], [200.0, 200.0, 200.0]],
            [[1.0, 3.0, 1.0], [2.0, 2.0, 5.0]],
        ],
        dtype=np.float64,
    )
    valid = np.array([[True, False], [True, True]])

    metrics = module.tcp_error_metrics("actor_pt", actor_tcp, pt_tcp, valid)

    assert metrics["actor_pt_tcp_mse_m2"] == 14.0 / 9.0
    assert np.isclose(metrics["actor_pt_tcp_rmse_m"], np.sqrt(14.0 / 9.0))
    assert metrics["actor_pt_tcp_l2_mean_m"] == 2.0
    assert metrics["actor_pt_tcp_l2_max_m"] == 3.0


def test_tcp_error_metrics_return_zero_for_no_valid_steps():
    module = _load_eval_module()
    zeros = np.zeros((1, 2, 3), dtype=np.float64)

    metrics = module.tcp_error_metrics(
        "actor_pt", zeros, zeros, np.array([[False, False]])
    )

    assert metrics == {
        "actor_pt_tcp_mse_m2": 0.0,
        "actor_pt_tcp_rmse_m": 0.0,
        "actor_pt_tcp_l2_mean_m": 0.0,
        "actor_pt_tcp_l2_max_m": 0.0,
    }
