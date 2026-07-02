#!/bin/bash
# Launch train_openrlt_right_arm_offline.py from an embodiment config name.
#
# Usage:
#   bash examples/embodiment/run_offline_train_openrlt.sh \
#     [config_name|config.yaml] [extra args...]
#
# Default config name:
#   offline_train_on_online122_residual_q010_new
#
# Config names are resolved under examples/embodiment/config/, matching
# run_piper_online_td3.sh. The YAML must contain `online_log_dir` and
# `output_name`. Unless explicitly set in YAML, the script derives:
#   --data-dir       <online_log_dir>/replay_buffer/rank_0
#   --feature-cache  <online_log_dir>/online_rltoken_features.pt
#   --output-dir     <online_log_dir>/<output_name>
# All other YAML keys map directly to argparse flags.

set -euo pipefail

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname "$(dirname "$EMBODIED_PATH")")
export SRC_FILE="${EMBODIED_PATH}/train_openrlt_right_arm_offline.py"
export PYTHONPATH="${REPO_PATH}:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-/opt/venv/piper/bin/python}"
CONFIG_NAME=${1:-offline_train_on_online122_residual_q010_new}
if [ "$#" -ge 1 ]; then
  shift
fi

if [[ "${CONFIG_NAME}" == */* || "${CONFIG_NAME}" == *.yaml || "${CONFIG_NAME}" == *.yml ]]; then
  CONFIG_FILE="${CONFIG_NAME}"
  CONFIG_LABEL=$(basename "${CONFIG_NAME}")
  CONFIG_LABEL="${CONFIG_LABEL%.yaml}"
  CONFIG_LABEL="${CONFIG_LABEL%.yml}"
else
  CONFIG_FILE="${EMBODIED_PATH}/config/${CONFIG_NAME}.yaml"
  CONFIG_LABEL="${CONFIG_NAME}"
fi

if [ ! -f "${CONFIG_FILE}" ]; then
  echo "Config not found: ${CONFIG_FILE}" >&2
  exit 1
fi

LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H:%M:%S')-${CONFIG_LABEL}"
MEGA_LOG_FILE="${LOG_DIR}/run_offline_train_openrlt.log"
mkdir -p "${LOG_DIR}"

CMD=("${PYTHON_BIN}" - "${CONFIG_FILE}" "${SRC_FILE}" "$@")
printf '%q ' "${CMD[@]}" > "${MEGA_LOG_FILE}"
printf '\n' >> "${MEGA_LOG_FILE}"
echo "Offline OpenRLT training from config: ${CONFIG_FILE}"

"${CMD[@]}" <<'PYEOF' 2>&1 | tee -a "${MEGA_LOG_FILE}"
import os
import pathlib
import shlex
import sys

import yaml

config_file = sys.argv[1]
train_script = pathlib.Path(sys.argv[2])
extra = sys.argv[3:]

with open(config_file, encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

if not isinstance(cfg, dict):
    raise TypeError(f"Expected a mapping in {config_file}, got {type(cfg).__name__}")

online_log_dir = (
    cfg.pop("online_log_dir", None)
    or cfg.pop("source_log_dir", None)
    or cfg.pop("log", None)
)
output_name = cfg.pop("output_name", None) or cfg.pop("out_suffix", None)
if online_log_dir is None:
    raise KeyError("Config must define `online_log_dir`.")
if output_name is None:
    raise KeyError("Config must define `output_name`.")

online_log_dir = pathlib.Path(os.path.expanduser(str(online_log_dir)))

def expand_path(value):
    return str(pathlib.Path(os.path.expanduser(str(value))))

data_dir = cfg.pop("data_dir", None)
feature_cache = cfg.pop("feature_cache", None)
output_dir = cfg.pop("output_dir", None)
cfg["data_dir"] = (
    expand_path(data_dir)
    if data_dir is not None
    else str(online_log_dir / "replay_buffer" / "rank_0")
)
cfg["feature_cache"] = (
    expand_path(feature_cache)
    if feature_cache is not None
    else str(online_log_dir / "online_rltoken_features.pt")
)
cfg["output_dir"] = (
    expand_path(output_dir)
    if output_dir is not None
    else str(online_log_dir / str(output_name))
)

cmd = [sys.executable, str(train_script)]
for k, v in cfg.items():
    if v is None:
        continue
    flag = "--" + k.replace("_", "-")
    if isinstance(v, bool):
        cmd.append(flag if v else f"--no-{k.replace('_', '-')}")
    elif isinstance(v, list):
        cmd += [flag] + [str(x) for x in v]
    else:
        cmd += [flag, str(v)]
cmd.extend(extra)

print("CMD:", shlex.join(cmd), flush=True)
os.execv(sys.executable, cmd)
PYEOF
