# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Give one unmodified RLinf training entry its own explicit Ray address."""

import os
import subprocess
import sys

import ray

from rlinf.utils.logging import get_logger

LOGGER = get_logger()

if __name__ == "__main__":
    context = ray.init(
        address="local",
        include_dashboard=False,
        num_cpus=16,
        object_store_memory=1024**3,
    )
    try:
        env = {**os.environ, "RAY_ADDRESS": context.address_info["gcs_address"]}
        LOGGER.info("Dedicated Ray address: %s", env["RAY_ADDRESS"])
        result = subprocess.run([sys.executable, "-u", *sys.argv[1:]], env=env)
    finally:
        ray.shutdown()
    raise SystemExit(result.returncode)
