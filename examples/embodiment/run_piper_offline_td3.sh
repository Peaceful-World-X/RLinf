#! /bin/bash
set -euo pipefail

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}")" && pwd )"
export REPO_PATH=$(dirname "$(dirname "$EMBODIED_PATH")")
export SRC_FILE="${EMBODIED_PATH}/train_offline_rl.py"
export PYTHONPATH="${REPO_PATH}:${PYTHONPATH:-}"
export RLINF_SKIP_ROS_CLEANUP=1

CONFIG_NAME=${1:-realworld_piper_peginsertion_td3_rl_token_real}
MAX_STEPS=${2:-200}
PYTHON_BIN=${PYTHON_BIN:-/opt/venv/piper/bin/python}
shift $(( $# >= 1 ? 1 : 0 ))
shift $(( $# >= 1 ? 1 : 0 ))

LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H%M%S')-${CONFIG_NAME}-offline"
MEGA_LOG_FILE="${LOG_DIR}/run_piper_offline_td3.log"
mkdir -p "${LOG_DIR}"

CMD=(
  "${PYTHON_BIN}" "${SRC_FILE}"
  --config-path "${EMBODIED_PATH}/config/"
  --config-name "${CONFIG_NAME}"
  "runner.logger.log_path=${LOG_DIR}"
  "runner.logger.experiment_name=${CONFIG_NAME}-offline"
  "runner.max_steps=${MAX_STEPS}"
  "runner.val_check_interval=-1"
  "runner.only_eval=False"
  "$@"
)

printf '%q ' "${CMD[@]}" > "${MEGA_LOG_FILE}"
printf '\n' >> "${MEGA_LOG_FILE}"
echo "Offline-only Piper TD3 run. No env/rollout workers should be created."
"${CMD[@]}" 2>&1 | tee -a "${MEGA_LOG_FILE}"
