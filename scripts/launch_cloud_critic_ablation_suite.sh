#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_PATH="$(dirname "${SCRIPT_DIR}")"

DATA_ROOT="${GIGARLINF_DATA_ROOT:-/shared_disk/users/angen.ye/data/gigarlinf/rlinf_cubeinsert_20260615}"
FULL_DATA="${DATA_ROOT}/demos_fail_reward0_normdelta"
KEY_DATA="${DATA_ROOT}/demos_fail_reward0_normdelta_keysegment"
KEY_SEGMENT_SUMMARY="${KEY_DATA}/key_segment_summary.json"
LOG_ROOT="${GIGARLINF_CRITIC_LOG_ROOT:-/shared_disk/users/angen.ye/code/giga_rlinf/train_logs/critic_ablations}"
LONG_ACTOR_RESUME="${GIGARLINF_LONG_ACTOR_RESUME:-/shared_disk/users/angen.ye/code/giga_rlinf/train_logs/actor_bc_ablations/bc_long100k_gpu0_v2/bc_long100k_gpu0_v2/checkpoints/global_step_1000}"
MAX_STEPS="${GIGARLINF_CRITIC_MAX_STEPS:-5000}"
SAVE_INTERVAL="${GIGARLINF_SAVE_INTERVAL:-500}"
FREE_MEM_MB="${GIGARLINF_QUEUE_FREE_MEM_MB:-1000}"
GPU_POOL=(${GIGARLINF_QUEUE_GPUS:-1 2 3 4 5 6 7})

mkdir -p "${LOG_ROOT}"
cd "${REPO_PATH}"

launch_detached() {
  local exp="$1"
  local gpu="$2"
  local data_dir="$3"
  shift 3
  if pgrep -af "${exp}" >/dev/null 2>&1; then
    echo "[skip] ${exp} already has a running process"
    return 0
  fi
  if [[ -d "${LOG_ROOT}/${exp}/${exp}/checkpoints/global_step_${MAX_STEPS}" ]]; then
    echo "[skip] ${exp} already has global_step_${MAX_STEPS}"
    return 0
  fi
  echo "[launch] ${exp} on GPU ${gpu}"
  nohup env \
    GIGARLINF_SAVE_INTERVAL="${SAVE_INTERVAL}" \
    GIGARLINF_VAL_NUM_CHUNKS=1000 \
    GIGARLINF_CRITIC_LOG_ROOT="${LOG_ROOT}" \
    bash scripts/run_cloud_critic_ablation.sh \
      "${exp}" "${gpu}" "${MAX_STEPS}" "${data_dir}" "$@" \
      > "${LOG_ROOT}/${exp}.launch.log" 2>&1 &
  echo "[pid] ${exp} $!"
}

launch_first_wave() {
  launch_detached critic_tail_td_data_next_full_5000_gpu1 1 "${FULL_DATA}" \
    +algorithm.use_n_step_target=False \
    algorithm.n_step=1 \
    algorithm.critic_target_action_source=data_next \
    algorithm.tail_curriculum.enabled=True \
    algorithm.tail_curriculum.start_window=1 \
    algorithm.tail_curriculum.hold_steps=0 \
    algorithm.tail_curriculum.end_window=96 \
    algorithm.tail_curriculum.warmup_steps=4000

  launch_detached critic_nstep1_data_next_full_5000_gpu2 2 "${FULL_DATA}" \
    +algorithm.use_n_step_target=True \
    algorithm.n_step=1 \
    algorithm.critic_target_action_source=data_next

  launch_detached critic_nstep3_data_next_full_5000_gpu3 3 "${FULL_DATA}" \
    +algorithm.use_n_step_target=True \
    algorithm.n_step=3 \
    algorithm.critic_target_action_source=data_next

  launch_detached critic_nstep10_data_next_full_5000_gpu4 4 "${FULL_DATA}" \
    +algorithm.use_n_step_target=True \
    algorithm.n_step=10 \
    algorithm.critic_target_action_source=data_next

  launch_detached critic_nstep20_data_next_full_5000_gpu6 6 "${FULL_DATA}" \
    +algorithm.use_n_step_target=True \
    algorithm.n_step=20 \
    algorithm.critic_target_action_source=data_next

  launch_detached critic_mc_return_full_5000_gpu7 7 "${FULL_DATA}" \
    +algorithm.use_n_step_target=False \
    algorithm.n_step=1 \
    algorithm.critic_target_action_source=data_next \
    algorithm.critic_td_coef=0.0 \
    algorithm.critic_mc_return_coef=1.0
}

gpu_mem_used() {
  nvidia-smi --id="$1" --query-gpu=memory.used --format=csv,noheader,nounits \
    | tr -d '[:space:]'
}

wait_for_free_gpu() {
  while true; do
    for gpu in "${GPU_POOL[@]}"; do
      local used
      used="$(gpu_mem_used "${gpu}")"
      if [[ -n "${used}" && "${used}" -lt "${FREE_MEM_MB}" ]]; then
        echo "${gpu}"
        return 0
      fi
    done
    echo "[queue] no free GPU yet; sleeping 120s" >&2
    sleep 120
  done
}

launch_queue() {
  # Give freshly launched Ray workers time to allocate memory before polling.
  sleep "${GIGARLINF_QUEUE_INITIAL_SLEEP:-300}"

  local gpu
  gpu="$(wait_for_free_gpu)"
  launch_detached critic_actor_target_tail_from_bc1000_full_5000_gpu"${gpu}" "${gpu}" "${FULL_DATA}" \
    runner.resume_dir="${LONG_ACTOR_RESUME}" \
    +runner.reset_global_step_on_resume=True \
    +algorithm.reset_update_step_on_resume=True \
    +algorithm.reset_critic_on_resume=True \
    +algorithm.use_n_step_target=False \
    algorithm.n_step=1 \
    algorithm.critic_target_action_source=actor \
    algorithm.tail_curriculum.enabled=True \
    algorithm.tail_curriculum.start_window=1 \
    algorithm.tail_curriculum.hold_steps=0 \
    algorithm.tail_curriculum.end_window=96 \
    algorithm.tail_curriculum.warmup_steps=4000

  sleep 300
  gpu="$(wait_for_free_gpu)"
  launch_detached critic_keysegment_data_next_fullviz_5000_gpu"${gpu}" "${gpu}" "${KEY_DATA}" \
    runner.offline_validation_visualization.data_paths="[${FULL_DATA}]" \
    +runner.offline_validation_visualization.key_segment_summary_path="${KEY_SEGMENT_SUMMARY}" \
    +algorithm.use_n_step_target=False \
    algorithm.n_step=1 \
    algorithm.critic_target_action_source=data_next
}

case "${1:-start}" in
  start)
    launch_first_wave
    nohup bash "$0" queue > "${LOG_ROOT}/critic_ablation_queue_manager.log" 2>&1 &
    echo "[queue-pid] $!"
    ;;
  queue)
    launch_queue
    ;;
  first-wave)
    launch_first_wave
    ;;
  *)
    echo "Usage: $0 {start|queue|first-wave}" >&2
    exit 2
    ;;
esac
