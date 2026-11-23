#!/usr/bin/env bash
set -euo pipefail
: "${OSS_CONFIG:=/input/workspace/.secrets/rlinf-oss.conf}"
: "${CLOUD_WORK:=/tmp/rlinf-g1}"
: "${GPU_COUNT:=8}"
: "${JOB_MODE:=env-only}"
: "${ALGORITHM:=recap}"
: "${RUN_ID:?Set RUN_ID}"
: "${RELEASE_MANIFEST:=$PWD/assets/manifest.json}"
[[ -r "$OSS_CONFIG" ]] || { echo "Missing mounted private OSS config: $OSS_CONFIG" >&2; exit 2; }
export OSS_CONFIG CLOUD_WORK GPU_COUNT JOB_MODE ALGORITHM RUN_ID RELEASE_MANIFEST
export OSSUTIL
OSSUTIL="$(python3 prepare_ossutil.py)"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline PYTHONUNBUFFERED=1
