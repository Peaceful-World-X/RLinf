#! /bin/bash
set -euo pipefail

export RLINF_SKIP_ROS_CLEANUP=1
export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname "$(dirname "$EMBODIED_PATH")")
export SRC_FILE="${EMBODIED_PATH}/train_async.py"
export PYTHONPATH="${REPO_PATH}:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-/opt/venv/piper/bin/python}"
CONFIG_NAME=${1:-realworld_piper_peginsertion_td3_rl_token_online}
if [ "$#" -ge 1 ]; then
  shift
fi

LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H:%M:%S')-${CONFIG_NAME}"
MEGA_LOG_FILE="${LOG_DIR}/run_piper_online_td3.log"
mkdir -p "${LOG_DIR}"

CMD=(
  "${PYTHON_BIN}" "${SRC_FILE}"
  --config-path "${EMBODIED_PATH}/config/"
  --config-name "${CONFIG_NAME}"
  "runner.logger.log_path=${LOG_DIR}"
  "$@"
)

printf '%q ' "${CMD[@]}" > "${MEGA_LOG_FILE}"
printf '\n' >> "${MEGA_LOG_FILE}"
echo "Piper online TD3/RLToken run. This entrypoint can command the real robot."
"${CMD[@]}" 2>&1 | tee -a "${MEGA_LOG_FILE}"
