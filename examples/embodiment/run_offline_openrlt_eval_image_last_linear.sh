#!/bin/bash
set -euo pipefail

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH="$(dirname "$(dirname "$EMBODIED_PATH")")"
export PYTHONPATH="${REPO_PATH}:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-/opt/venv/piper/bin/python}"
CONFIG_NAME="${1:-offline_train_on_online122_residual_q010_new_image_last_linear}"
STEP="${2:-latest}"
if [ "$#" -ge 1 ]; then shift; fi
if [ "$#" -ge 1 ]; then shift; fi

if [[ "${CONFIG_NAME}" == */* || "${CONFIG_NAME}" == *.yaml || "${CONFIG_NAME}" == *.yml ]]; then
  CONFIG_FILE="${CONFIG_NAME}"
else
  CONFIG_FILE="${EMBODIED_PATH}/config/${CONFIG_NAME}.yaml"
fi

if [ ! -f "${CONFIG_FILE}" ]; then
  echo "Config not found: ${CONFIG_FILE}" >&2
  exit 1
fi

read_config() {
  "${PYTHON_BIN}" - "$CONFIG_FILE" "$1" <<'PYEOF'
import sys
import yaml
path, key = sys.argv[1], sys.argv[2]
with open(path, encoding='utf-8') as f:
    cfg = yaml.safe_load(f) or {}
value = cfg.get(key)
if value is None:
    value = ''
print(value)
PYEOF
}

ONLINE_LOG_DIR="$(read_config online_log_dir)"
OUTPUT_NAME="$(read_config output_name)"
DATA_DIR="$(read_config data_dir)"
NORM_STATS_PATH="$(read_config norm_stats_path)"
URDF_PATH="$(read_config urdf_path)"
MODEL_CONFIG="$(read_config feature_cache_model_config)"
FEATURE_CACHE="$(read_config feature_cache)"
GAMMA="$(read_config gamma)"

if [ -z "${ONLINE_LOG_DIR}" ] || [ -z "${OUTPUT_NAME}" ]; then
  echo "Config must define online_log_dir and output_name: ${CONFIG_FILE}" >&2
  exit 1
fi
if [ -z "${DATA_DIR}" ]; then
  DATA_DIR="${ONLINE_LOG_DIR}/replay_buffer/rank_0"
fi
if [ -z "${GAMMA}" ]; then
  GAMMA="0.94"
fi

CHECKPOINT_ROOT="${ONLINE_LOG_DIR}/${OUTPUT_NAME}/checkpoints"
if [ "${STEP}" = "latest" ] || [ "${STEP}" = "final" ]; then
  CHECKPOINT="$(find "${CHECKPOINT_ROOT}" -mindepth 2 -maxdepth 2 -path '*/actor_critic.pt' | sed -E 's#(.*/global_step_([0-9]+)/actor_critic.pt)#\2 \1#' | sort -n | tail -n 1 | cut -d' ' -f2-)"
elif [[ "${STEP}" == global_step_* ]]; then
  CHECKPOINT="${CHECKPOINT_ROOT}/${STEP}/actor_critic.pt"
elif [[ "${STEP}" =~ ^[0-9]+$ ]]; then
  CHECKPOINT="${CHECKPOINT_ROOT}/global_step_${STEP}/actor_critic.pt"
else
  CHECKPOINT="${STEP}"
fi

if [ -z "${CHECKPOINT}" ] || [ ! -f "${CHECKPOINT}" ]; then
  echo "Checkpoint not found for STEP=${STEP} under ${CHECKPOINT_ROOT}" >&2
  exit 1
fi

STEP_NAME="$(basename "$(dirname "${CHECKPOINT}")")"
OUT="${ONLINE_LOG_DIR}/eval_${OUTPUT_NAME}_${STEP_NAME}_all_trajs"

CMD=(
  "${PYTHON_BIN}" "${REPO_PATH}/scripts/evaluate_online_openrlt_trajectories.py"
  --log-dir "${ONLINE_LOG_DIR}"
  --data-dir "${DATA_DIR}"
  --checkpoint "${CHECKPOINT}"
  --output-dir "${OUT}"
  --model-config "${MODEL_CONFIG}"
  --feature-cache "${FEATURE_CACHE}"
  --norm-stats-path "${NORM_STATS_PATH}"
  --urdf-path "${URDF_PATH}"
  --gamma "${GAMMA}"
  --batch-size 8
  --device cuda:0
)

if [ -n "${RL_TOKEN_PATH:-}" ]; then
  CMD+=(--rl-token-path "${RL_TOKEN_PATH}")
fi

printf 'Offline OpenRLT eval from config: %s\n' "${CONFIG_FILE}"
printf 'Checkpoint: %s\n' "${CHECKPOINT}"
printf 'Output: %s\n' "${OUT}"
printf 'CMD:'
printf ' %q' "${CMD[@]}" "$@"
printf '\n'
exec "${CMD[@]}" "$@"
