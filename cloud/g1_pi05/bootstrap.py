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

"""Restore the content-addressed release from OSS into worker-local storage."""

import json
import os
from pathlib import Path

from assets_io import extract, identifier, verified_get


def main():
    spec = json.loads(Path(os.environ["RELEASE_MANIFEST"]).read_text())
    if spec["schema"] != 2:
        raise ValueError("Unknown release schema")
    objects = spec["objects"]
    alg = os.environ["ALGORITHM"]
    if alg not in ("recap", "steam") or os.environ["JOB_MODE"] not in (
        "policy",
        "env-only",
        "smoke",
        "train",
    ):
        raise ValueError("Unknown algorithm or job mode")
    root = Path(os.environ["CLOUD_WORK"]).resolve()
    if root.parts[1:2] in (("input",), ("ossfs",)):
        raise ValueError("CLOUD_WORK must be worker-local")
    home = root / identifier(spec["release"])

    # File paths within a release are immutable. Distinct experiments get their own writable data.
    def get(name, target):
        obj = objects[name]
        return verified_get(obj, target)

    runtime_digest = objects["runtime"]["sha256"]
    target = get("runtime", root / "downloads" / runtime_digest)
    extract(target, home / "runtime", runtime_digest)
    get("tokenizer", home / "openpi-cache/big_vision/paligemma_tokenizer.model")
    # Each required CFG base file has its own immutable OSS URI and digest.
    for filename in ("model.safetensors", "config.json"):
        get("pi05/" + filename, home / "models/pi05" / filename)
    for key in (() if os.environ["JOB_MODE"] == "policy" else ("gemma", alg + "_vision")):
        for name in objects:
            if name.startswith(key + "/"):
                filename = name.split("/")[1]
                identifier(filename)
                get(name, home / "models" / key / filename)
    job = home / "jobs" / alg / identifier(os.environ["RUN_ID"])
    job.mkdir(parents=True, exist_ok=True)
    if os.environ["JOB_MODE"] in ("train", "policy"):
        uri = os.environ["DATA_URI"]
        if not uri.startswith("oss://"):
            raise ValueError("Expected explicit OSS DATA_URI")
        ds = {
            "uri": uri,
            "bytes": int(os.environ["DATA_BYTES"]),
            "sha256": os.environ["DATA_SHA256"],
        }
        if len(ds["sha256"]) != 64 or any(
            c not in "0123456789abcdef" for c in ds["sha256"]
        ):
            raise ValueError("Invalid DATA_SHA256")
        archive = verified_get(ds, root / "downloads" / ds["sha256"])
        extract(archive, job / "input", ds["sha256"])
    print(home)


if __name__ == "__main__":
    main()
