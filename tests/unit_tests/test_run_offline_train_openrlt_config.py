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

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "examples" / "embodiment" / "run_offline_train_openrlt.sh"


def test_offline_train_wrapper_keeps_explicit_paths(tmp_path):
    config = tmp_path / "offline.yaml"
    config.write_text(
        "\n".join(
            [
                "online_log_dir: /tmp/source-log",
                "output_name: run-out",
                "data_dir: /tmp/explicit-demos",
                "feature_cache: /tmp/explicit-features.pt",
                "output_dir: /tmp/explicit-output",
                "norm_stats_path: /tmp/norm.json",
                "urdf_path: /tmp/piper.urdf",
                "model_path: /tmp/openpi-model",
                "rl_token_path: /tmp/rltoken",
            ]
        ),
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PYTHON_BIN"] = "/opt/venv/piper/bin/python"

    proc = subprocess.run(
        ["bash", str(SCRIPT), str(config), "--help"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert proc.returncode == 0, proc.stdout
    assert "--data-dir /tmp/explicit-demos" in proc.stdout
    assert "--feature-cache /tmp/explicit-features.pt" in proc.stdout
    assert "--output-dir /tmp/explicit-output" in proc.stdout
    assert "--model-path /tmp/openpi-model" in proc.stdout
    assert "--rl-token-path /tmp/rltoken" in proc.stdout
    assert "/tmp/source-log/replay_buffer/rank_0" not in proc.stdout
