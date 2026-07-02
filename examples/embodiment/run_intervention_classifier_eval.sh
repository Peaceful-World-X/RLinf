#!/bin/bash
set -euo pipefail

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname "$(dirname "$EMBODIED_PATH")")
export PYTHONPATH="${REPO_PATH}:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-/opt/venv/piper/bin/python}"
CONFIG_NAME=${1:-intervention_classifier_online122_residual_q010_new}
CHECKPOINT=${2:-/home/focal/shared_disk/users/kwj/RLinf/logs/intervention_classifier_online122_residual_q010_new/best_intervention_classifier.pt}
if [ "$#" -ge 1 ]; then shift; fi
if [ "$#" -ge 1 ]; then shift; fi

if [[ "${CONFIG_NAME}" == */* || "${CONFIG_NAME}" == *.yaml || "${CONFIG_NAME}" == *.yml ]]; then
  CONFIG_FILE="${CONFIG_NAME}"
else
  CONFIG_FILE="${EMBODIED_PATH}/config/${CONFIG_NAME}.yaml"
fi

"${PYTHON_BIN}" "${EMBODIED_PATH}/eval_intervention_classifier.py" --config "${CONFIG_FILE}" --checkpoint "${CHECKPOINT}" "$@"
