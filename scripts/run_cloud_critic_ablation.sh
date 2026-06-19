#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 EXP_NAME GPU_RANK MAX_STEPS DATA_DIR [hydra overrides...]" >&2
  exit 2
fi

EXP_NAME="$1"
GPU_RANK="$2"
MAX_STEPS="$3"
DATA_DIR="$4"
shift 4

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_PATH="$(dirname "${SCRIPT_DIR}")"

DATA_ROOT="${GIGARLINF_DATA_ROOT:-/shared_disk/users/angen.ye/data/gigarlinf/rlinf_cubeinsert_20260615}"
LOG_ROOT="${GIGARLINF_CRITIC_LOG_ROOT:-/shared_disk/users/angen.ye/code/giga_rlinf/train_logs/critic_ablations}"
MODEL_PATH="${GIGARLINF_PI05_PATH:-${DATA_ROOT}/pi05_cube_insert}"
RL_TOKEN_PATH="${GIGARLINF_RLTOKEN_PATH:-${DATA_ROOT}/rltoken_pi05_cube_insert_10000/10000}"
NORM_STATS_PATH="${GIGARLINF_NORM_STATS_PATH:-${MODEL_PATH}/norm_stats.json}"
URDF_PATH="${GIGARLINF_URDF_PATH:-${REPO_PATH}/assets/piper_local_assets/piper.urdf}"
SAVE_INTERVAL="${GIGARLINF_SAVE_INTERVAL:-500}"
VAL_NUM_CHUNKS="${GIGARLINF_VAL_NUM_CHUNKS:-1000}"
KEY_SEGMENT_SUMMARY="${GIGARLINF_KEY_SEGMENT_SUMMARY:-}"

SAFE_EXP_NAME="$(echo "${EXP_NAME}" | tr -c 'A-Za-z0-9_' '_')"
export RLINF_RAY_NAMESPACE="${RLINF_RAY_NAMESPACE:-rlinf_${SAFE_EXP_NAME}}"
export PYTHON_BIN="${PYTHON_BIN:-/opt/venv/piper/bin/python}"

mkdir -p "${LOG_ROOT}"

COMMON_OVERRIDES=(
  "runner.logger.log_path=${LOG_ROOT}/${EXP_NAME}"
  "runner.logger.experiment_name=${EXP_NAME}"
  "runner.save_initial_checkpoint=True"
  "runner.save_interval=${SAVE_INTERVAL}"
  "runner.offline_validation_visualization.data_paths=[${DATA_DIR}]"
  "runner.offline_validation_visualization.urdf_path=${URDF_PATH}"
  "runner.offline_validation_visualization.action_norm_stats_path=${NORM_STATS_PATH}"
  "runner.offline_validation_visualization.num_chunks_per_trajectory=${VAL_NUM_CHUNKS}"
  "runner.offline_validation_visualization.index_mode=terminal_window"
  "+runner.offline_validation_visualization.whole_trajectory=True"
  "+runner.offline_validation_visualization.timeline_only=True"
  "runner.offline_validation_visualization.num_success_trajectories=5"
  "runner.offline_validation_visualization.num_failure_trajectories=5"
  "runner.offline_validation_visualization.split_success_failure=True"
  "+runner.offline_validation_visualization.image_thumb_size=64"
  "cluster.component_placement.actor=${GPU_RANK}"
  "cluster.component_placement.env=${GPU_RANK}"
  "cluster.component_placement.rollout=${GPU_RANK}"
  "actor.model.model_path=${MODEL_PATH}"
  "actor.model.rl_token_path=${RL_TOKEN_PATH}"
  "actor.model.action_norm_stats_path=${NORM_STATS_PATH}"
  "algorithm.offline_data_paths=[${DATA_DIR}]"
  "algorithm.stage_actor_bc_only=False"
  "algorithm.stage_freeze_actor=True"
  "algorithm.actor_q_coef=0.0"
  "algorithm.bc_coef=0.0"
  "algorithm.critic_td_coef=1.0"
  "algorithm.critic_mc_return_coef=0.0"
  "algorithm.training_stages.actor_critic_coupling_start_step=-1"
  "algorithm.training_stages.coupled_actor_q_coef=0.0"
  "algorithm.training_stages.coupled_critic_target_action_source=data_next"
  "algorithm.tail_curriculum.enabled=False"
)

if [[ -n "${KEY_SEGMENT_SUMMARY}" ]]; then
  COMMON_OVERRIDES+=(
    "+runner.offline_validation_visualization.key_segment_summary_path=${KEY_SEGMENT_SUMMARY}"
  )
fi

cd "${REPO_PATH}"
exec bash examples/embodiment/run_piper_offline_td3.sh \
  realworld_piper_peginsertion_td3_rl_token_real \
  "${MAX_STEPS}" \
  "${COMMON_OVERRIDES[@]}" \
  "$@"
