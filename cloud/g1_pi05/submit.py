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

"""Submit a minimal bootstrap to AIStudio; no training data or secrets in source_root."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from assets_io import extract, identifier

WORKFLOW = Path("cloud/g1_pi05")
# Only tracked source and configuration files needed by the offline trainers.
SOURCE_DIRS = (
    "rlinf",
    "examples/offline_rl",
    str(WORKFLOW),
    "pyproject.toml",
    "LICENSE",
)
SOURCE_TYPES = {
    ".py",
    ".yaml",
    ".yml",
    ".toml",
    ".json",
    ".sh",
    ".md",
    ".jinja",
    ".jinja2",
}


def stage_checkout(repo, dest):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args])

    commit = git("rev-parse", "--verify", "HEAD^{commit}").decode().strip()
    paths = git("ls-tree", "-r", "-z", "--name-only", commit, "--", *SOURCE_DIRS).split(
        b"\0"
    )
    files = [
        os.fsdecode(name)
        for name in paths
        if name
        and (
            Path(os.fsdecode(name)).suffix in SOURCE_TYPES
            or Path(os.fsdecode(name)).name in ("LICENSE", "cloud.env.example")
        )
    ]
    if str(WORKFLOW / "k8s_job.sh") not in files:
        raise ValueError(
            "The selected Git commit does not contain the G1 cloud workflow"
        )
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "checkout.tar"
        subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "archive",
                "--format=tar",
                "-o",
                str(archive),
                commit,
                "--",
                *files,
            ],
            check=True,
        )
        extract(archive, dest, commit)
        (dest / ".release.sha256").rename(dest / ".rlinf-git-commit")
    return commit


def main():
    p = argparse.ArgumentParser()
    p.add_argument("algorithm", choices=["recap", "steam"])
    p.add_argument("mode", choices=["env-only", "smoke", "train", "policy"])
    p.add_argument("--run-id", required=True, type=identifier)
    p.add_argument(
        "--manifest", type=Path, help="Override the committed assets/manifest.json"
    )
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    if os.getenv("RESUME", "0") not in ("0", "1"):
        raise ValueError("RESUME must be 0 or 1")
    env = {
        k: os.environ[k]
        for k in (
            "DATA_URI",
            "DATA_BYTES",
            "DATA_SHA256",
            "OSS_CONFIG",
            "CLOUD_WORK",
            "POLICY_STEPS",
            "VALUE_STEPS",
            "SAVE_INTERVAL",
            "RESUME",
            "POLICY_MICRO_BATCH",
            "VALUE_MICRO_BATCH",
            "POLICY_BATCH",
            "VALUE_BATCH",
        )
        if k in os.environ
    }
    n = int(os.getenv("GPU_COUNT", "8"))
    gpu_type = os.getenv("GPU_TYPE", "a100").lower()
    if n not in (2, 4, 8):
        raise ValueError("GPU_COUNT must be 2, 4, or 8")
    if a.mode == "policy":
        if a.algorithm != "recap":
            raise ValueError("Prelabelled policy mode requires RECAP")
        if a.manifest is None:
            a.manifest = (
                Path(__file__).resolve().parent / "assets" / "recap_g1_v4000.json"
            )
        release = json.loads(a.manifest.read_text())
        data = release["dataset"]
        for key, value in {
            "DATA_URI": data["uri"],
            "DATA_BYTES": str(data["bytes"]),
            "DATA_SHA256": data["sha256"],
        }.items():
            if key in env and env[key] != value:
                raise ValueError(f"{key} differs from the prelabelled release manifest")
            env[key] = value
        env.setdefault("POLICY_STEPS", "10000")
        env.setdefault("POLICY_BATCH", "256")
        env.setdefault("POLICY_MICRO_BATCH", "2")
        env.setdefault("SAVE_INTERVAL", "1000")
    if a.mode in ("train", "policy"):
        for k in (
            "DATA_URI",
            "DATA_BYTES",
            "DATA_SHA256",
            "POLICY_STEPS",
            *(("VALUE_STEPS",) if a.mode == "train" else ()),
        ):
            if not env.get(k):
                raise ValueError("Set " + k)
        import re

        if not env["DATA_URI"].startswith("oss://") or not re.fullmatch(
            "[a-f0-9]{64}", env["DATA_SHA256"]
        ):
            raise ValueError("Invalid DATA_URI or DATA_SHA256")
        for k in (
            "DATA_BYTES",
            "POLICY_STEPS",
            *(("VALUE_STEPS",) if a.mode == "train" else ()),
        ):
            if int(env[k]) <= 0:
                raise ValueError(k + " must be positive")
    env.update(
        ALGORITHM=a.algorithm, JOB_MODE=a.mode, RUN_ID=a.run_id, GPU_COUNT=str(n)
    )
    repo = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix="rlinf-submit-") as tmp:
        src = Path(tmp) / "checkout"
        env["RLINF_GIT_COMMIT"] = stage_checkout(repo, src)
        manifest_path = src / WORKFLOW / "assets/manifest.json"
        if a.manifest:
            shutil.copyfile(a.manifest, manifest_path)
        manifest = json.loads(manifest_path.read_text())
        if (
            a.mode == "policy"
            and manifest.get("prelabelled_tag") != "recap_g1_v4000_n10_q30"
        ):
            raise ValueError("Select the verified 20k RECAP release manifest")
        if manifest.get("schema") != 2 or "source" in manifest["objects"]:
            raise ValueError("Use a Git-workflow asset manifest (schema 2)")
        print(f"Submitting Git commit {env['RLINF_GIT_COMMIT']}", file=sys.stderr)
        if a.dry_run:
            print(
                json.dumps(
                    {
                        "source": "git",
                        "workflow": str(WORKFLOW),
                        "env": env,
                        "gpus": n,
                        "gpu_type": gpu_type,
                        "pi05_weights": manifest["objects"]["pi05/model.safetensors"][
                            "uri"
                        ],
                        "release": manifest["release"],
                    },
                    indent=2,
                )
            )
            return
        from pypai.conf import ExecConf, GpuType, KMConf, NasStoreConf
        from pypai.job import PythonJobBuilder

        job = PythonJobBuilder(
            source_root=str(src),
            command=f"bash {WORKFLOW}/k8s_job.sh",
            main_file="",
            km_conf=KMConf(
                image=os.getenv(
                    "AISTUDIO_IMAGE",
                    "reg.docker.alibaba-inc.com/aii/aistudio:12550155-20250530161247",
                )
            ),
            master=ExecConf(
                cpu=96,
                memory=1024 * 800,
                gpu_num=n,
                num=1,
                gpu_type=getattr(GpuType, gpu_type.upper(), gpu_type),
                disk_m=1024 * 500,
            ),
            k8s_app_name=os.getenv("RESOURCE_GROUP", "digitaltechnology"),
            k8s_priority="high",
            nas_storage=NasStoreConf(mount_path="/input"),
            tag=f"type=train,model=pi05,algo={a.algorithm}",
            platform="kubemaker",
            envs=env,
        )
        print("record_id=", job.run(enable_wait=False))


if __name__ == "__main__":
    main()
