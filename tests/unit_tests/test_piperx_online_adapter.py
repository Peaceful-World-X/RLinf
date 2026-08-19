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

"""Unit tests for the PiperX adapter without importing RealWorld dependencies."""

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_SOURCE = REPO_ROOT / "rlinf/envs/realworld/piperx/piperx_env.py"
CONTROLLER_SOURCE = REPO_ROOT / "rlinf/envs/realworld/piperx/piperx_controller.py"
ENTRY_SOURCE = REPO_ROOT / "examples/embodiment/train_piperx_openrlt.py"
WORKER_GROUP_SOURCE = REPO_ROOT / "rlinf/scheduler/worker/worker_group.py"


def _class_node(path: Path, name: str) -> ast.ClassDef:
    module = ast.parse(path.read_text(encoding="utf-8"))
    return next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == name
    )


def _annotated_default(class_node: ast.ClassDef, name: str):
    statement = next(
        node
        for node in class_node.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == name
    )
    return ast.literal_eval(statement.value)


def test_piperx_config_uses_authorized_ros_topics_and_zero_reward() -> None:
    config = _class_node(ENV_SOURCE, "PiperXRobotConfig")
    assert _annotated_default(config, "left_state_topic") == "/puppet/joint_left"
    assert _annotated_default(config, "right_state_topic") == "/puppet/joint_right"
    assert _annotated_default(config, "left_policy_topic") == "/policy/joint_left"
    assert _annotated_default(config, "right_policy_topic") == "/policy/joint_right"
    assert _annotated_default(config, "intervention_mode_topic") == (
        "/piper/intervene_mode"
    )
    assert _annotated_default(config, "subtask_reward_topic") == (
        "/piper/subtask_reward"
    )
    assert _annotated_default(config, "subtask_reward_value") == 0.0
    assert _annotated_default(config, "publish_policy_actions") is False


def test_reward_pedal_callback_has_visible_log_marker() -> None:
    source = ENV_SOURCE.read_text(encoding="utf-8")
    assert "[PIPERX REWARD PEDAL]" in source
    assert "self.config.subtask_reward_value" in source


def test_piperx_go_zero_matches_parent_call_signature() -> None:
    controller = _class_node(CONTROLLER_SOURCE, "PiperXController")
    method = next(
        node
        for node in controller.body
        if isinstance(node, ast.FunctionDef) and node.name == "go_zero"
    )
    assert [argument.arg for argument in method.args.args] == [
        "self",
        "is_mit_mode",
        "ns",
    ]


def test_piperx_policy_publish_is_explicitly_gated() -> None:
    source = CONTROLLER_SOURCE.read_text(encoding="utf-8")
    assert "if not self._publish_policy_actions:" in source
    assert "super()._publish_joint_ctrl(topic, action, speed_pct)" in source


def test_piperx_episode_keyboard_contract_is_local_to_adapter() -> None:
    source = ENV_SOURCE.read_text(encoding="utf-8")
    assert '("Key.enter",)' in source
    assert '("Key.space", "q", "x", "f")' in source
    assert '("1", "0")' in source
    assert '"piperx_break_chunk": True' in source
    assert '"piperx_action_executed": False' in source
    assert "home = np.asarray([0.0] * 6 + [0.3]" in source
    assert "self._move_to_home_continuous(home, home)" in source
    assert "max_step = np.asarray([0.01] * 6 + [0.2]" in source


def test_piperx_matches_reference_client_image_contract() -> None:
    source = ENV_SOURCE.read_text(encoding="utf-8")
    assert 'imgmsg_to_cv2(msg, "rgb8")' in source
    assert "cv2.resize" not in source
    assert "msg.header.stamp.to_sec()" in source
    assert "get_synchronized_qpos(frame_time)" in source


def test_piperx_controller_keeps_timestamped_joint_history() -> None:
    source = CONTROLLER_SOURCE.read_text(encoding="utf-8")
    assert "_left_joint_history" in source
    assert "_right_joint_history" in source
    assert "if stamp >= frame_time" in source


def test_piperx_operator_inputs_are_forwarded_to_ray_worker() -> None:
    entry = ENTRY_SOURCE.read_text(encoding="utf-8")
    worker_group = WORKER_GROUP_SOURCE.read_text(encoding="utf-8")
    assert 'b"\\n", b" ", b"q", b"x", b"f", b"0", b"1"' in entry
    assert '"/piper/intervene_mode"' in entry
    assert '"/piper/subtask_reward"' in entry
    assert '"RLINF_PIPERX_TERMINAL"' in worker_group
    assert '"RLINF_PIPERX_KEY_FIFO"' in worker_group
