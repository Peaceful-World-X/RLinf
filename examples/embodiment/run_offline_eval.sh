#! /bin/bash

set -euo pipefail

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))

export PYTHONPATH="${REPO_PATH}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1
export RLINF_SKIP_ROS_CLEANUP=1
export RAY_memory_usage_threshold=0.99
if [ $# -lt 1 ]; then
    echo "Usage: $0 <config_name>" >&2
    exit 1
fi

CONFIG_NAME=$1
CONFIG_FILE="${EMBODIED_PATH}/config/${CONFIG_NAME}.yaml"

if [ ! -f "${CONFIG_FILE}" ]; then
    echo "Config file not found: ${CONFIG_FILE}" >&2
    exit 1
fi

echo "Using Python at $(which python)"
python "${EMBODIED_PATH}/offline_eval.py" --config-name "${CONFIG_NAME}"
