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

"""Publish completed checkpoints and restore only fully published training steps."""

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from assets_io import oss, save_json, sha, verified_get
from loss_log import publish, restore_history, stage_history


def remote_json(uri):
    try:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "item"
            oss("cp", uri, path, capture=True)
            return json.loads(path.read_text())
    except subprocess.CalledProcessError as e:
        if "NoSuchKey" in (e.stdout or "") or "StatusCode=404" in (e.stdout or ""):
            return None
        raise


class RunStore:
    def __init__(self, job, identity):
        self.job = job
        self.uri = os.environ["RUN_URI"]
        self.sent = set()
        self.identity = identity
        self.record = job / "identity.json"

    def begin(self):
        prev = remote_json(self.uri + "/identity.json")
        if prev is not None and (
            prev != self.identity or os.getenv("RESUME", "0") != "1"
        ):
            raise ValueError(
                "Run ID already exists. Use a new ID or RESUME=1 with identical settings."
            )
        save_json(self.record, self.identity)
        oss("cp", self.record, self.uri + "/identity.json", "-f")
        save_json(self.job / "active-attempt.json", os.environ["ATTEMPT"])

    def restore(self, stage):
        record = remote_json(f"{self.uri}/{stage}/latest.json")
        if not record:
            return 0
        step = int(record["step"])
        root = self.job / "results" / stage / "checkpoints" / f"global_step_{step}"
        for name, obj in record["files"].items():
            dst = (root / name).resolve()
            if not dst.is_relative_to(root.resolve()):
                raise ValueError(name)
            verified_get(obj, dst)
        if len(record["loss_history"]) != step:
            raise ValueError(f"Incomplete loss history for {stage} checkpoint {step}")
        restore_history(self.job, stage, record["loss_history"])
        (root / ".complete").write_text(str(step))
        self.sent.add((stage, step))
        return step

    def sync(self, stage, committed_step=0):
        ckpts = self.job / "results" / stage / "checkpoints"
        for marker in sorted(
            ckpts.glob("global_step_*/.complete"),
            key=lambda m: int(m.parent.name.rsplit("_", 1)[1]),
        ):
            root = marker.parent
            step = int(root.name.rsplit("_", 1)[1])
            if (
                step <= committed_step
                or (stage, step) in self.sent
                or marker.read_text().strip() != str(step)
            ):
                continue
            files = {}
            for p in sorted(root.rglob("*")):
                if not p.is_file() or p == marker:
                    continue
                name = str(p.relative_to(root))
                uri = f"{self.uri}/{stage}/{root.name}/{name}"
                oss("cp", p, uri, "-f")
                files[name] = dict(uri=uri, bytes=p.stat().st_size, sha256=sha(p))
            if not any(n.startswith("actor/") for n in files):
                raise ValueError("Checkpoint has no actor files")
            index = self.job / f"{stage}-checkpoint.json"
            save_json(
                index,
                {
                    "step": step,
                    "files": files,
                    "loss_history": stage_history(self.job, stage, step),
                },
            )
            oss("cp", index, f"{self.uri}/{stage}/{root.name}/manifest.json", "-f")
            oss("cp", index, f"{self.uri}/{stage}/latest.json", "-f")
            self.sent.add((stage, step))
            # Keep the current complete checkpoint locally. Older copies are
            # already addressable through their own OSS manifests.
            for old in ckpts.glob("global_step_*/.complete"):
                old_step = int(old.parent.name.rsplit("_", 1)[1])
                if old_step < step and (stage, old_step) in self.sent:
                    shutil.rmtree(old.parent)


def finish_uploads():
    job = (
        Path(os.environ["RELEASE_DIR"])
        / "jobs"
        / os.environ["ALGORITHM"]
        / os.environ["RUN_ID"]
    )
    active = job / "active-attempt.json"
    if not active.is_file() or json.loads(active.read_text()) != os.environ["ATTEMPT"]:
        return
    store = RunStore(job, json.loads((job / "identity.json").read_text()))
    ready = job / "loss-ready.json"
    if not ready.is_file() or json.loads(ready.read_text()) != os.environ["ATTEMPT"]:
        return
    try:
        publish(job, store.uri)
    finally:
        sync_completed(store)


def sync_completed(store):
    for stage in ("value", "policy"):
        latest = remote_json(f"{store.uri}/{stage}/latest.json")
        store.sync(stage, int(latest["step"]) if latest else 0)


if __name__ == "__main__":
    finish_uploads()
