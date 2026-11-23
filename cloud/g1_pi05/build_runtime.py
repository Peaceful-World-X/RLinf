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

"""Build an offline CPython runtime from the verified Beijing installation."""

import argparse
import importlib.metadata as md
import json
import os
import stat
import subprocess
import sys
import sysconfig
import tarfile
from pathlib import Path


def build(output):
    base = Path(sys.base_prefix).resolve()
    site = Path(sysconfig.get_paths()["purelib"])
    records = {}
    packages = {}
    for dist in md.distributions(path=[str(site)]):
        name = dist.metadata["Name"]
        packages[name] = dist.version
        for entry in dist.files or ():
            src = Path(os.path.abspath(site / entry))
            if not src.is_relative_to(site) or not src.is_file():
                continue
            if (
                any(x in src.parts for x in ("__pycache__", ".git"))
                or src.suffix == ".pyc"
            ):
                continue
            if src.name == "direct_url.json" or src.name.startswith("__editable__"):
                continue
            # RECORD contains package-owned files; never follow outside links.
            if src.is_symlink() and not src.resolve().is_relative_to(site):
                raise ValueError(f"External link in installed wheel: {src}")
            records[
                "python/lib/python3.11/site-packages/" + str(src.relative_to(site))
            ] = src
    for src in base.rglob("*"):
        if not src.is_file() or "__pycache__" in src.parts or src.suffix == ".pyc":
            continue
        if src.is_symlink() and not src.resolve().is_relative_to(base):
            raise ValueError(f"External interpreter link: {src}")
        records["python/" + str(src.relative_to(base))] = src
    # The venv's console scripts refer to its original absolute interpreter.
    # init_env replaces their shebangs after restoring the release.
    for name in ("ray", "torchrun", "accelerate", "huggingface-cli", "hf", "uv"):
        src = Path(sys.prefix) / "bin" / name
        if src.is_file():
            records["python/bin/" + name] = src
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".lock.json").write_text(
        json.dumps(packages, sort_keys=True, indent=2)
    )
    partial = output.with_suffix(".partial")
    with partial.open("wb") as dest:
        proc = subprocess.Popen(
            ["pigz", "-1", "-p", "4"], stdin=subprocess.PIPE, stdout=dest
        )
        try:
            with tarfile.open(fileobj=proc.stdin, mode="w|", dereference=True) as tar:
                for name, path in sorted(records.items()):
                    info = tar.gettarinfo(str(path), name)
                    # tarfile may coalesce hard links; store standalone regular files.
                    info.type = tarfile.REGTYPE
                    info.size = path.stat().st_size
                    info.mode = stat.S_IMODE(path.stat().st_mode) & 0o777
                    with path.open("rb") as stream:
                        tar.addfile(info, stream)
        finally:
            proc.stdin.close()
            code = proc.wait()
        if code:
            raise RuntimeError(f"compression failed: {code}")
    partial.replace(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    build(parser.parse_args().output)
