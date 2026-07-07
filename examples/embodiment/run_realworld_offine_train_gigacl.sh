#!/bin/bash
# Launch the real-world charge-insert offline GigaCL pipeline.
#
# This wrapper runs the joint offline OpenRLT actor/critic stage and the
# intervention-classifier stage from one config. By default the OpenRLT stage
# uses the image_last_linear prefix-token training config.
#
# Usage:
#   bash examples/embodiment/run_realworld_offine_train_gigacl.sh \
#     [realworld_charge_insert_offline|realworld_charge_insert_offline.yaml] \
#     [extra args passed to train_joint_offline_openrlt_classifier.py]

set -euo pipefail

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname "$(dirname "$EMBODIED_PATH")")
export PYTHONPATH="${REPO_PATH}:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-/opt/venv/piper/bin/python}"
CONFIG_NAME=${1:-realworld_charge_insert_offline}
if [ "$#" -ge 1 ]; then
  shift
fi

if [[ "${CONFIG_NAME}" == */* ]]; then
  CONFIG_FILE="${CONFIG_NAME}"
else
  CONFIG_LABEL="${CONFIG_NAME}"
  CONFIG_LABEL="${CONFIG_LABEL%.yaml}"
  CONFIG_LABEL="${CONFIG_LABEL%.yml}"
  CONFIG_FILE="${EMBODIED_PATH}/config/${CONFIG_LABEL}.yaml"
fi

if [ ! -f "${CONFIG_FILE}" ]; then
  echo "Config not found: ${CONFIG_FILE}" >&2
  exit 1
fi

echo "Real-world offline GigaCL training from config: ${CONFIG_FILE}"
"${PYTHON_BIN}" "${EMBODIED_PATH}/train_joint_offline_openrlt_classifier.py" \
  --config "${CONFIG_FILE}" "$@"
