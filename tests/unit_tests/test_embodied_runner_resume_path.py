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

from rlinf.runners.embodied_runner import EmbodiedRunner


def test_resolve_resume_checkpoint_path_uses_actor_critic_directory(tmp_path):
    resume_dir = tmp_path / "global_step_1500"
    resume_dir.mkdir()
    (resume_dir / "actor_critic.pt").write_bytes(b"checkpoint")

    assert EmbodiedRunner._resolve_actor_checkpoint_path(str(resume_dir)) == str(
        resume_dir
    )


def test_resolve_resume_checkpoint_path_keeps_legacy_actor_subdir(tmp_path):
    actor_dir = tmp_path / "global_step_1500" / "actor"
    actor_dir.mkdir(parents=True)

    assert EmbodiedRunner._resolve_actor_checkpoint_path(str(actor_dir.parent)) == str(
        actor_dir
    )


def test_resolve_resume_checkpoint_path_reports_expected_candidates(tmp_path):
    resume_dir = tmp_path / "global_step_1500"
    resume_dir.mkdir()

    try:
        EmbodiedRunner._resolve_actor_checkpoint_path(str(resume_dir))
    except FileNotFoundError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected missing resume checkpoint to raise")

    assert str(resume_dir / "actor") in message
    assert str(resume_dir / "actor_critic.pt") in message
