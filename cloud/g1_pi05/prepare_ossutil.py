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

"""Bootstrap the pinned OSS client from one private object using worker-mounted credentials."""

import base64
import configparser
import hmac
import http.client
import json
import os
import shutil
import tempfile
import time
from email.utils import formatdate
from pathlib import Path
from urllib.parse import quote, urlsplit

from assets_io import ENDPOINT, ROOT, sha


def download(spec, target):
    uri = spec["uri"]
    if not uri.startswith(ROOT + "/runtime/") or not uri.endswith("/ossutil64"):
        raise ValueError("Unexpected OSS client URI")
    obj = urlsplit(uri)
    config = configparser.RawConfigParser()
    with open(os.environ["OSS_CONFIG"]) as stream:
        config.read_file(stream)
    creds = config["Credentials"]
    token = creds.get("stsToken") or creds.get("securityToken")
    for attempt in range(20):
        headers = {"Date": formatdate(usegmt=True)}
        extra = ""
        if token:
            headers["x-oss-security-token"] = token
            extra = f"x-oss-security-token:{token}\n"
        payload = f"GET\n\n\n{headers['Date']}\n{extra}/{obj.netloc}{obj.path}"
        signature = hmac.digest(
            creds["accessKeySecret"].encode(), payload.encode(), "sha1"
        )
        headers["Authorization"] = (
            "OSS " + creds["accessKeyID"] + ":" + base64.b64encode(signature).decode()
        )
        conn = http.client.HTTPSConnection(f"{obj.netloc}.{ENDPOINT}", timeout=90)
        try:
            conn.request("GET", quote(obj.path), headers=headers)
            response = conn.getresponse()
            if response.status == 429 or response.status >= 500:
                raise ConnectionError(f"OSS HTTP {response.status}")
            if response.status != 200:
                raise RuntimeError(
                    f"OSS client download failed (HTTP {response.status})"
                )
            with target.open("wb") as out:
                shutil.copyfileobj(response, out)
            if target.stat().st_size != spec["bytes"] or sha(target) != spec["sha256"]:
                raise OSError("OSS client checksum mismatch")
            return
        except (OSError, http.client.HTTPException):
            if attempt == 19:
                raise
            time.sleep(min(2**attempt, 10))
        finally:
            conn.close()


def main():
    spec = json.loads(Path(os.environ["RELEASE_MANIFEST"]).read_text())["objects"][
        "ossutil"
    ]
    cache = (
        Path(os.environ["CLOUD_WORK"]).resolve()
        / "tools"
        / spec["sha256"]
        / "ossutil64"
    )
    for path in (
        os.getenv("OSSUTIL"),
        cache,
        shutil.which("ossutil64"),
        shutil.which("ossutil"),
    ):
        if (
            path
            and os.access(path, os.X_OK)
            and Path(path).is_file()
            and sha(path) == spec["sha256"]
        ):
            print(Path(path).resolve())
            return
    cache.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=cache.parent) as tmp:
        tool = Path(tmp) / "ossutil64"
        download(spec, tool)
        tool.chmod(0o755)
        tool.replace(cache)
    print(cache)


if __name__ == "__main__":
    main()
