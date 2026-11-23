#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ "${RLINF_GIT_COMMIT:?Missing submission commit}" != "$(cat ../../.rlinf-git-commit)" ]]; then
  echo "Submitted source and job commit differ" >&2
  exit 2
fi
source ./init_cfg.sh
[[ "$RUN_ID" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]] || exit 2
[[ "$ALGORITHM" == recap || "$ALGORITHM" == steam ]] || exit 2
ATTEMPT="$(date -u +%Y%m%dT%H%M%SZ)-$$"
export ATTEMPT
RUN_URI="oss://tipalgog/sudu/rlinf-cloud/runs/$ALGORITHM/$RUN_ID"
export RUN_URI
pid=''
workflow_dir=''
finish() {
  rc=$?
  trap - EXIT TERM INT
  if [[ -n "$pid" ]]; then kill -TERM -- "-$pid" 2>/dev/null || true; wait "$pid" || true; fi
  if ((rc != 0)) && [[ -n "$workflow_dir" && -x "${TRAIN_PY:-}" ]]; then
    if ! "$TRAIN_PY" "$workflow_dir/sync_runs.py"; then echo "Final synchronization failed" >&2; fi
  fi
  exit "$rc"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
python3 bootstrap.py
source ./init_env.sh
cd "$REPO_PATH"
workflow_dir="$REPO_PATH/cloud/g1_pi05"
set -m
"$TRAIN_PY" "$workflow_dir/run_pipeline.py" &
pid=$!
set +m
wait "$pid"
pid=''
