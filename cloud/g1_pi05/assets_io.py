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

"""Verified OSS objects and safe, atomic local archive extraction."""

import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

ROOT = "oss://tipalgog/sudu/rlinf-cloud"
ENDPOINT = "oss-cn-hangzhou-zmf.aliyuncs.com"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def oss(*args, capture=False):
    cmd = [
        os.environ.get("OSSUTIL", "ossutil64"),
        "--retry-times",
        "20",
        "-e",
        ENDPOINT,
        "-c",
        os.environ["OSS_CONFIG"],
        *map(str, args),
    ]
    return subprocess.run(
        cmd,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )


def identifier(name):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,100}", name):
        raise ValueError(f"Invalid identifier {name!r}")
    return name


def verified_get(spec, target):
    dest = Path(target)

    def ready(p):
        return (
            p.is_file()
            and p.stat().st_size == spec["bytes"]
            and sha(p) == spec["sha256"]
        )

    if ready(dest):
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
        p = Path(tmp) / "object"
        oss("cp", spec["uri"], p)
        if not ready(p):
            raise RuntimeError(f"Checksum mismatch: {spec['uri']}")
        p.replace(dest)
    return dest


def save_json(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=p.parent, mode="w", delete=False) as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(f.name, p)


def extract(archive, dest, digest):
    dest = Path(dest)
    marker = dest / ".release.sha256"
    if marker.is_file() and marker.read_text() == digest:
        return
    if dest.exists():
        raise RuntimeError(f"Existing directory does not match archive: {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
        staging = Path(tmp) / "content"
        staging.mkdir()
        with tarfile.open(archive, "r|*") as tar:
            for item in tar:
                rel = PurePosixPath(item.name)
                if (
                    rel.is_absolute()
                    or ".." in rel.parts
                    or not (item.isdir() or item.isfile())
                ):
                    raise ValueError(f"Unsafe archive entry {item.name}")
                path = staging / rel
                if item.isdir():
                    path.mkdir(parents=True, exist_ok=True)
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with path.open("xb") as f:
                        shutil.copyfileobj(tar.extractfile(item), f, 2 << 20)
                    path.chmod(0o755 if item.mode & 0o111 else 0o644)
        (staging / marker.name).write_text(digest)
        staging.rename(dest)
