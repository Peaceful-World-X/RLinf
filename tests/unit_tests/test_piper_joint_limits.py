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

import ast
import importlib.util
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]


def _field_default_list(class_node: ast.ClassDef, field_name: str) -> list[float]:
    for stmt in class_node.body:
        if not isinstance(stmt, ast.AnnAssign):
            continue
        if not isinstance(stmt.target, ast.Name) or stmt.target.id != field_name:
            continue
        call = stmt.value
        assert isinstance(call, ast.Call)
        factory = next(
            keyword.value
            for keyword in call.keywords
            if keyword.arg == "default_factory"
        )
        assert isinstance(factory, ast.Lambda)
        return ast.literal_eval(factory.body)
    raise AssertionError(f"{field_name} not found")


def test_piper_env_uses_single_arm_seven_dim_joint_limits():
    source = (REPO_ROOT / "rlinf/envs/realworld/piper/piper_env.py").read_text()
    module_ast = ast.parse(source)
    config_cls = next(
        node
        for node in module_ast.body
        if isinstance(node, ast.ClassDef) and node.name == "PiperRobotConfig"
    )

    assert _field_default_list(config_cls, "min_qpos") == [
        -2.618,
        0.0,
        -2.967,
        -1.745,
        -1.22,
        -2.7925,
        0.0,
    ]
    assert _field_default_list(config_cls, "max_qpos") == [
        2.618,
        3.14,
        0.0,
        1.745,
        1.22,
        2.7925,
        0.07,
    ]


def test_piper_controller_clip_limits_match_collection_joint_limits():
    utils_path = REPO_ROOT / "rlinf/envs/realworld/piper/utils.py"
    spec = importlib.util.spec_from_file_location("piper_utils_for_test", utils_path)
    utils = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(utils)

    np.testing.assert_allclose(
        utils.PIPER_JOINT_LIMITS_LOW,
        np.array([-2.618, 0.0, -2.967, -1.745, -1.22, -2.7925]),
    )
    np.testing.assert_allclose(
        utils.PIPER_JOINT_LIMITS_HIGH,
        np.array([2.618, 3.14, 0.0, 1.745, 1.22, 2.7925]),
    )

    clipped = utils.clip_joint_positions(np.array([-9.0, -1.0, -9.0, -9.0, -9.0, -9.0]))
    np.testing.assert_allclose(
        clipped, np.array([-2.618, 0.0, -2.967, -1.745, -1.22, -2.7925])
    )
