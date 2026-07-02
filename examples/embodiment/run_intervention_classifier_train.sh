#!/bin/bash
set -euo pipefail

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname "$(dirname "$EMBODIED_PATH")")
export PYTHONPATH="${REPO_PATH}:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-/opt/venv/piper/bin/python}"
CONFIG_NAME=${1:-intervention_classifier_online122_residual_q010_new}
if [ "$#" -ge 1 ]; then
  shift
fi

if [[ "${CONFIG_NAME}" == */* || "${CONFIG_NAME}" == *.yaml || "${CONFIG_NAME}" == *.yml ]]; then
  CONFIG_FILE="${CONFIG_NAME}"
else
  CONFIG_FILE="${EMBODIED_PATH}/config/${CONFIG_NAME}.yaml"
fi

if [ ! -f "${CONFIG_FILE}" ]; then
  echo "Config not found: ${CONFIG_FILE}" >&2
  exit 1
fi

echo "Intervention classifier training from config: ${CONFIG_FILE}"
"${PYTHON_BIN}" "${EMBODIED_PATH}/train_intervention_classifier.py" --config "${CONFIG_FILE}" "$@"
