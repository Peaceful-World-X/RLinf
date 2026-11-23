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

"""Publish explicitly approved non-data assets; address objects by SHA-256."""

import argparse
import json
import subprocess
from pathlib import Path

from assets_io import ROOT, identifier, oss, save_json, sha

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MODELS = {
    "gemma": (
        "gemma-3-270m",
        [
            "model.safetensors",
            "config.json",
            "tokenizer.json",
            "tokenizer.model",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "added_tokens.json",
        ],
    ),
    "recap_vision": (
        "siglip2-so400m-patch14-224",
        ["config.json", "model.safetensors", "preprocessor_config.json"],
    ),
    "steam_vision": (
        "siglip-so400m-patch14-384",
        ["config.json", "model.safetensors", "preprocessor_config.json"],
    ),
    "pi05": ("pi05_base", ["model.safetensors", "config.json"]),
}


def put(path, prefix, execute):
    digest = sha(path)
    uri = ROOT + "/" + prefix + "/" + digest + "/" + path.name
    if execute:
        # Immutable content address; verify actual remote size and digest metadata.
        try:
            info = oss("stat", uri, capture=True).stdout
        except subprocess.CalledProcessError as e:
            if "NoSuchKey" not in (e.stdout or "") and "404" not in (e.stdout or ""):
                raise
            info = ""

        def matches(text):
            fields = {
                k.strip().lower(): v.strip()
                for k, _, v in (l.partition(":") for l in text.splitlines())
            }
            return (
                fields.get("content-length") == str(path.stat().st_size)
                and fields.get("x-oss-meta-sha256") == digest
            )

        if not matches(info):
            oss("cp", path, uri, "--meta", "x-oss-meta-sha256:" + digest)
            if not matches(oss("stat", uri, capture=True).stdout):
                raise ValueError("Post-upload verification failed: " + uri)
    return dict(uri=uri, bytes=path.stat().st_size, sha256=digest)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--inputs",
        type=Path,
        required=True,
        help="Private local JSON mapping asset names to paths",
    )
    p.add_argument("--release", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--upload", action="store_true")
    a = p.parse_args()
    identifier(a.release)
    paths = {k: Path(v) for k, v in json.loads(a.inputs.read_text()).items()}
    output = a.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if output.is_relative_to(REPO):
        raise ValueError("Place generated release outside the repo")
    objects = {}
    for key, (model, files) in MODELS.items():
        for name in files:
            path = paths[key] / name
            if path.is_symlink() or path.stat().st_size < (
                4096 if name.endswith(".safetensors") else 1
            ):
                raise ValueError(str(path))
            objects[f"{key}/{name}"] = put(path, "models/" + model, a.upload)
    for key in ("runtime", "runtime_lock", "tokenizer", "base_provenance", "ossutil"):
        objects[key] = put(
            paths[key],
            "models/tokenizer"
            if key == "tokenizer"
            else "runtime"
            if key in ("runtime", "runtime_lock", "ossutil")
            else "models/pi05_base",
            a.upload,
        )
    manifest = dict(schema=2, release=a.release, objects=objects)
    save_json(output / "manifest.json", manifest)
    if a.upload:
        ref = put(output / "manifest.json", "releases/" + a.release, True)
        save_json(output / "release.json", ref)
        print(ref["uri"], flush=True)


if __name__ == "__main__":
    main()
